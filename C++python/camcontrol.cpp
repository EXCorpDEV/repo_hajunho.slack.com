#include "framework.h"
#include "EXCAM2.h"

// Media Foundation 및 기타 헤더들
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <shlwapi.h>
#include <strsafe.h>

// 라이브러리 링크
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "mf.lib")

#define MAX_LOADSTRING 100

// 전역 변수:
HINSTANCE hInst;                                // 현재 인스턴스입니다.
WCHAR szTitle[MAX_LOADSTRING];                  // 제목 표시줄 텍스트입니다.
WCHAR szWindowClass[MAX_LOADSTRING];            // 기본 창 클래스 이름입니다.

// 캡처 스레드 관련 전역 변수
HANDLE g_hCaptureThread = NULL;
volatile bool g_bExitCapture = false;

// 프레임 데이터를 위한 전역 변수
BYTE* g_pFrameBuffer = nullptr;
UINT g_Width = 0;
UINT g_Height = 0;
CRITICAL_SECTION g_FrameCS;

// 이 코드 모듈에 포함된 함수의 선언을 전달합니다:
ATOM                MyRegisterClass(HINSTANCE hInstance);
BOOL                InitInstance(HINSTANCE, int);
LRESULT CALLBACK    WndProc(HWND, UINT, WPARAM, LPARAM);
INT_PTR CALLBACK    About(HWND, UINT, WPARAM, LPARAM);

//---------------------------------------------------------------------------
// Webcam 캡처 스레드 함수
//---------------------------------------------------------------------------

DWORD WINAPI WebcamCaptureThread(LPVOID lpParam)
{
    HRESULT hr = S_OK;

    // COM 초기화 (멀티스레드)
    hr = CoInitializeEx(NULL, COINIT_MULTITHREADED);
    if (FAILED(hr))
    {
        OutputDebugString(_T("CoInitializeEx 실패\n"));
        return 0;
    }

    // Media Foundation 초기화
    hr = MFStartup(MF_VERSION);
    if (FAILED(hr))
    {
        OutputDebugString(_T("MFStartup 실패\n"));
        CoUninitialize();
        return 0;
    }

    // 웹캠 장치 열거를 위한 속성 객체 생성
    IMFAttributes* pAttributes = nullptr;
    hr = MFCreateAttributes(&pAttributes, 1);
    if (FAILED(hr))
    {
        OutputDebugString(_T("MFCreateAttributes 실패\n"));
        MFShutdown();
        CoUninitialize();
        return 0;
    }

    hr = pAttributes->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
        MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
    if (FAILED(hr))
    {
        OutputDebugString(_T("SetGUID 실패\n"));
        pAttributes->Release();
        MFShutdown();
        CoUninitialize();
        return 0;
    }

    IMFActivate** ppDevices = nullptr;
    UINT32 count = 0;
    hr = MFEnumDeviceSources(pAttributes, &ppDevices, &count);
    pAttributes->Release();
    if (FAILED(hr) || count == 0)
    {
        OutputDebugString(_T("비디오 캡처 장치를 찾을 수 없음\n"));
        if (ppDevices) {
            CoTaskMemFree(ppDevices);
        }
        MFShutdown();
        CoUninitialize();
        return 0;
    }

    // 첫 번째 장치를 사용합니다.
    IMFMediaSource* pSource = nullptr;
    hr = ppDevices[0]->ActivateObject(IID_PPV_ARGS(&pSource));
    for (UINT32 i = 0; i < count; i++)
    {
        ppDevices[i]->Release();
    }
    CoTaskMemFree(ppDevices);
    if (FAILED(hr))
    {
        OutputDebugString(_T("ActivateObject 실패\n"));
        MFShutdown();
        CoUninitialize();
        return 0;
    }

    // Source Reader 생성
    IMFAttributes* pReaderAttributes = nullptr;
    hr = MFCreateAttributes(&pReaderAttributes, 1);
    if (SUCCEEDED(hr))
    {
        hr = pReaderAttributes->SetUINT32(MF_READWRITE_DISABLE_CONVERTERS, TRUE);
    }
    if (FAILED(hr))
    {
        OutputDebugString(_T("Reader 속성 생성 실패\n"));
        pSource->Release();
        MFShutdown();
        CoUninitialize();
        return 0;
    }

    IMFSourceReader* pReader = nullptr;
    hr = MFCreateSourceReaderFromMediaSource(pSource, pReaderAttributes, &pReader);
    pReaderAttributes->Release();
    pSource->Release();
    if (FAILED(hr))
    {
        OutputDebugString(_T("MFCreateSourceReaderFromMediaSource 실패\n"));
        MFShutdown();
        CoUninitialize();
        return 0;
    }

    // 비디오 스트림의 미디어 타입을 RGB32로 설정
    IMFMediaType* pType = nullptr;
    hr = MFCreateMediaType(&pType);
    if (SUCCEEDED(hr))
    {
        hr = pType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    }
    if (SUCCEEDED(hr))
    {
        hr = pType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
    }
    if (SUCCEEDED(hr))
    {
        hr = pReader->SetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, pType);
    }

    // 미디어 타입에서 해상도 정보 가져오기
    IMFMediaType* pCurrentType = nullptr;
    hr = pReader->GetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, &pCurrentType);
    if (SUCCEEDED(hr))
    {
        UINT32 width, height;
        MFGetAttributeSize(pCurrentType, MF_MT_FRAME_SIZE, &width, &height);

        EnterCriticalSection(&g_FrameCS);
        g_Width = width;
        g_Height = height;
        if (g_pFrameBuffer) delete[] g_pFrameBuffer;
        g_pFrameBuffer = new BYTE[width * height * 4];
        LeaveCriticalSection(&g_FrameCS);

        pCurrentType->Release();
    }

    pType->Release();
    if (FAILED(hr))
    {
        OutputDebugString(_T("미디어 타입 설정 실패\n"));
        pReader->Release();
        MFShutdown();
        CoUninitialize();
        return 0;
    }

    OutputDebugString(_T("웹캠 캡처 시작\n"));

    // 프레임 캡처 루프
    while (!g_bExitCapture)
    {
        DWORD streamIndex = 0, flags = 0;
        LONGLONG llTimestamp = 0;
        IMFSample* pSample = nullptr;

        hr = pReader->ReadSample(
            (DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM,
            0,
            &streamIndex,
            &flags,
            &llTimestamp,
            &pSample
        );

        if (FAILED(hr))
        {
            OutputDebugString(_T("ReadSample 실패\n"));
            break;
        }

        if (flags & MF_SOURCE_READERF_STREAMTICK)
        {
            if (pSample)
                pSample->Release();
            continue;
        }

        if (pSample)
        {
            IMFMediaBuffer* pBuffer = nullptr;
            hr = pSample->ConvertToContiguousBuffer(&pBuffer);
            if (SUCCEEDED(hr))
            {
                BYTE* pData = nullptr;
                DWORD maxLength, currentLength;
                hr = pBuffer->Lock(&pData, &maxLength, &currentLength);
                if (SUCCEEDED(hr))
                {
                    EnterCriticalSection(&g_FrameCS);
                    memcpy(g_pFrameBuffer, pData, min(currentLength, g_Width * g_Height * 4));
                    LeaveCriticalSection(&g_FrameCS);

                    pBuffer->Unlock();
                    InvalidateRect((HWND)lpParam, NULL, FALSE);  // 화면 갱신 요청
                }
                pBuffer->Release();
            }
            pSample->Release();
        }

        Sleep(30); // 프레임 레이트 조절
    }

    // 자원 정리
    pReader->Release();
    MFShutdown();
    CoUninitialize();
    OutputDebugString(_T("웹캠 캡처 종료\n"));
    return 0;
}

//---------------------------------------------------------------------------
// InitInstance 함수 수정
//---------------------------------------------------------------------------

BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
    hInst = hInstance;

    InitializeCriticalSection(&g_FrameCS);  // Critical Section 초기화

    HWND hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, nullptr, nullptr, hInstance, nullptr);

    if (!hWnd)
    {
        return FALSE;
    }

    ShowWindow(hWnd, nCmdShow);
    UpdateWindow(hWnd);

    // 캡처 스레드 생성
    g_hCaptureThread = CreateThread(NULL, 0, WebcamCaptureThread, hWnd, 0, NULL);
    if (g_hCaptureThread == NULL)
    {
        MessageBox(hWnd, _T("캡처 스레드 생성 실패"), _T("오류"), MB_OK);
    }

    return TRUE;
}

//---------------------------------------------------------------------------
// WndProc 함수 수정
//---------------------------------------------------------------------------

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_COMMAND:
    {
        int wmId = LOWORD(wParam);
        switch (wmId)
        {
        case IDM_ABOUT:
            DialogBox(hInst, MAKEINTRESOURCE(IDD_ABOUTBOX), hWnd, About);
            break;
        case IDM_EXIT:
            DestroyWindow(hWnd);
            break;
        default:
            return DefWindowProc(hWnd, message, wParam, lParam);
        }
    }
    break;
    case WM_PAINT:
    {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hWnd, &ps);

        EnterCriticalSection(&g_FrameCS);
        if (g_pFrameBuffer && g_Width > 0 && g_Height > 0)
        {
            BITMAPINFO bmi = { 0 };
            bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
            bmi.bmiHeader.biWidth = g_Width;
            bmi.bmiHeader.biHeight = -(LONG)g_Height;  // Top-down DIB
            bmi.bmiHeader.biPlanes = 1;
            bmi.bmiHeader.biBitCount = 32;
            bmi.bmiHeader.biCompression = BI_RGB;

            SetStretchBltMode(hdc, COLORONCOLOR);

            // 창 크기에 맞게 스트레치하여 그리기
            RECT rc;
            GetClientRect(hWnd, &rc);
            StretchDIBits(hdc,
                0, 0, rc.right, rc.bottom,
                0, 0, g_Width, g_Height,
                g_pFrameBuffer,
                &bmi,
                DIB_RGB_COLORS,
                SRCCOPY);
        }
        LeaveCriticalSection(&g_FrameCS);

        EndPaint(hWnd, &ps);
    }
    break;
    case WM_DESTROY:
        // 애플리케이션 종료 전 캡처 스레드 종료 요청
        g_bExitCapture = true;
        if (g_hCaptureThread)
        {
            WaitForSingleObject(g_hCaptureThread, INFINITE);
            CloseHandle(g_hCaptureThread);
            g_hCaptureThread = NULL;
        }

        EnterCriticalSection(&g_FrameCS);
        if (g_pFrameBuffer)
        {
            delete[] g_pFrameBuffer;
            g_pFrameBuffer = nullptr;
        }
        LeaveCriticalSection(&g_FrameCS);
        DeleteCriticalSection(&g_FrameCS);

        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

// WinMain 함수 추가
int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPWSTR    lpCmdLine,
    _In_ int       nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    // 전역 문자열을 초기화합니다.
    LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
    LoadStringW(hInstance, IDC_EXCAM2, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);

    // 애플리케이션 초기화를 수행합니다:
    if (!InitInstance(hInstance, nCmdShow))
    {
        return FALSE;
    }

    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_EXCAM2));

    MSG msg;

    // 기본 메시지 루프입니다:
    while (GetMessage(&msg, nullptr, 0, 0))
    {
        if (!TranslateAccelerator(msg.hwnd, hAccelTable, &msg))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
    }

    return (int)msg.wParam;
}

// About 대화 상자 메시지 처리기입니다.
INT_PTR CALLBACK About(HWND hDlg, UINT message, WPARAM wParam, LPARAM lParam)
{
    UNREFERENCED_PARAMETER(lParam);
    switch (message)
    {
    case WM_INITDIALOG:
        return (INT_PTR)TRUE;

    case WM_COMMAND:
        if (LOWORD(wParam) == IDOK || LOWORD(wParam) == IDCANCEL)
        {
            EndDialog(hDlg, LOWORD(wParam));
            return (INT_PTR)TRUE;
        }
        break;
    }
    return (INT_PTR)FALSE;
}

//
//  함수: MyRegisterClass()
//
//  용도: 창 클래스를 등록합니다.
//
ATOM MyRegisterClass(HINSTANCE hInstance)
{
    WNDCLASSEXW wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);

    wcex.style = CS_HREDRAW | CS_VREDRAW;
    wcex.lpfnWndProc = WndProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = hInstance;
    wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_EXCAM2));
    wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wcex.lpszMenuName = MAKEINTRESOURCEW(IDC_EXCAM2);
    wcex.lpszClassName = szWindowClass;
    wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

    return RegisterClassExW(&wcex);
}


// EXCAM.cpp : 애플리케이션에 대한 진입점을 정의합니다.
//
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0601 // Windows 7 이상
#endif

#include "framework.h"
#include "EXCAM.h"

// Media Foundation 및 기타 헤더들
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <shlwapi.h>
#include <strsafe.h>
#include <tchar.h>
#include <windows.h>

// 라이브러리 링크
#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "shlwapi.lib")


#define MAX_LOADSTRING 100

// 전역 변수:
HINSTANCE hInst;                                // 현재 인스턴스입니다.
WCHAR szTitle[MAX_LOADSTRING];                  // 제목 표시줄 텍스트입니다.
WCHAR szWindowClass[MAX_LOADSTRING];            // 기본 창 클래스 이름입니다.

// 캡처 스레드 관련 전역 변수
HANDLE g_hCaptureThread = NULL;
volatile bool g_bExitCapture = false;

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

    // Source Reader 생성 (공유 모드를 유도하기 위해 변환 비활성화)
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

    // 비디오 스트림의 미디어 타입을 RGB32로 설정합니다.
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

        // MF_SOURCE_READERF_STREAMTICK은 실제 프레임이 아닌 스트림 타이밍 정보입니다.
        if (flags & MF_SOURCE_READERF_STREAMTICK)
        {
            if (pSample)
                pSample->Release();
            continue;
        }

        if (pSample)
        {
            // 예제에서는 단순히 타임스탬프를 출력합니다.
            wchar_t debugMsg[128];
            StringCchPrintf(debugMsg, 128, L"프레임 캡처됨 - 타임스탬프: %lld\n", llTimestamp);
            OutputDebugString(debugMsg);

            // 여기서 pSample의 버퍼에 접근하여 영상 데이터를 처리할 수 있습니다.

            pSample->Release();
        }

        Sleep(30);
    }

    // 자원 정리
    pReader->Release();
    MFShutdown();
    CoUninitialize();
    OutputDebugString(_T("웹캠 캡처 종료\n"));
    return 0;
}

//---------------------------------------------------------------------------
// wWinMain 및 나머지 기존 코드
//---------------------------------------------------------------------------

int APIENTRY wWinMain(_In_ HINSTANCE hInstance,
    _In_opt_ HINSTANCE hPrevInstance,
    _In_ LPWSTR    lpCmdLine,
    _In_ int       nCmdShow)
{
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);

    // 전역 문자열 초기화
    LoadStringW(hInstance, IDS_APP_TITLE, szTitle, MAX_LOADSTRING);
    LoadStringW(hInstance, IDC_EXCAM, szWindowClass, MAX_LOADSTRING);
    MyRegisterClass(hInstance);

    // 애플리케이션 초기화를 수행합니다.
    if (!InitInstance(hInstance, nCmdShow))
    {
        return FALSE;
    }

    HACCEL hAccelTable = LoadAccelerators(hInstance, MAKEINTRESOURCE(IDC_EXCAM));

    MSG msg;

    // 기본 메시지 루프
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

//
//  함수: MyRegisterClass()
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
    wcex.hIcon = LoadIcon(hInstance, MAKEINTRESOURCE(IDI_EXCAM));
    wcex.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wcex.lpszMenuName = MAKEINTRESOURCEW(IDC_EXCAM);
    wcex.lpszClassName = szWindowClass;
    wcex.hIconSm = LoadIcon(wcex.hInstance, MAKEINTRESOURCE(IDI_SMALL));

    return RegisterClassExW(&wcex);
}

//
//   함수: InitInstance(HINSTANCE, int)
//
BOOL InitInstance(HINSTANCE hInstance, int nCmdShow)
{
    hInst = hInstance; // 인스턴스 핸들을 전역 변수에 저장합니다.

    HWND hWnd = CreateWindowW(szWindowClass, szTitle, WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, 0, CW_USEDEFAULT, 0, nullptr, nullptr, hInstance, nullptr);

    if (!hWnd)
    {
        return FALSE;
    }

    ShowWindow(hWnd, nCmdShow);
    UpdateWindow(hWnd);

    // 캡처 스레드 생성: 창 생성 후 별도의 스레드에서 웹캠 캡처를 시작합니다.
    g_hCaptureThread = CreateThread(NULL, 0, WebcamCaptureThread, NULL, 0, NULL);
    if (g_hCaptureThread == NULL)
    {
        MessageBox(hWnd, _T("캡처 스레드 생성 실패"), _T("오류"), MB_OK);
    }

    return TRUE;
}

//
//  함수: WndProc(HWND, UINT, WPARAM, LPARAM)
//
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    switch (message)
    {
    case WM_COMMAND:
    {
        int wmId = LOWORD(wParam);
        // 메뉴 선택을 파싱합니다.
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
        // TODO: 필요한 그리기 작업 수행 (예: 영상 데이터를 창에 출력)
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
        PostQuitMessage(0);
        break;
    default:
        return DefWindowProc(hWnd, message, wParam, lParam);
    }
    return 0;
}

// 정보 대화 상자의 메시지 처리기입니다.
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

"""
Google Gemini AI를 사용한 개선된 챗봇
설치 필요: pip install google-generativeai python-dotenv
"""
import os
import sys
from dotenv import load_dotenv
import google.generativeai as genai


def setup_api():
    """API 설정 및 초기화"""
    try:
        load_dotenv('gemini.env')
        api_key = os.getenv('API_KEY')

        if not api_key:
            print("오류: API_KEY가 설정되지 않았습니다. gemini.env 파일을 확인하세요.")
            return None

        genai.configure(api_key=api_key)
        return genai.GenerativeModel('models/gemini-1.5-flash-latest')

    except Exception as e:
        print(f"API 설정 오류: {e}")
        return None


def get_available_models():
    """사용 가능한 모델 목록 출력"""
    try:
        print("\n사용 가능한 모델 목록:")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"- {m.name}")
        print()
    except Exception as e:
        print(f"모델 목록 조회 오류: {e}")


def main():
    """메인 함수"""
    print("=== Gemini AI 챗봇 ===")
    print("종료하려면 'quit', 'exit', '종료'를 입력하세요.")
    print("모델 목록을 보려면 'models'를 입력하세요.\n")

    # API 설정
    model = setup_api()
    if not model:
        return

    # 대화 루프
    while True:
        try:
            user_input = input("질문: ").strip()

            # 종료 명령 확인
            if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                print("챗봇을 종료합니다.")
                break

            # 빈 입력 확인
            if not user_input:
                print("질문을 입력해주세요.")
                continue

            # 모델 목록 요청 확인
            if user_input.lower() == 'models':
                get_available_models()
                continue

            # AI 응답 생성
            print("응답 생성 중...")
            response = model.generate_content(user_input)
            print(f"\nGemini: {response.text}\n")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\n\n프로그램이 중단되었습니다.")
            break

        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            print("다시 시도해주세요.\n")


if __name__ == "__main__":
    main()
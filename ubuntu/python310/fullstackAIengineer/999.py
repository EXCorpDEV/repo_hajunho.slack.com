"""
새로운 Google Gen AI SDK를 사용한 개선된 챗봇
설치 필요: pip install google-genai python-dotenv
"""
import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

def setup_client():
    """클라이언트 설정 및 초기화"""
    try:
        load_dotenv('gemini.env')
        api_key = os.getenv('API_KEY') or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')

        if not api_key:
            print("오류: API_KEY가 설정되지 않았습니다.")
            print("gemini.env 파일에 API_KEY 또는 환경변수 GEMINI_API_KEY/GOOGLE_API_KEY를 설정하세요.")
            return None

        # 새로운 SDK의 중앙 집중식 클라이언트 생성
        client = genai.Client(api_key=api_key)
        return client

    except Exception as e:
        print(f"클라이언트 설정 오류: {e}")
        return None

def get_available_models(client):
    """사용 가능한 모델 목록 출력"""
    try:
        print("\n사용 가능한 모델 목록:")
        for model in client.models.list():
            print(f"- {model.name}")
            # 지원하는 액션도 표시
            if hasattr(model, 'supported_actions') and model.supported_actions:
                actions = ', '.join(model.supported_actions)
                print(f"  지원 기능: {actions}")
        print()
    except Exception as e:
        print(f"모델 목록 조회 오류: {e}")

def main():
    """메인 함수"""
    print("=== 새로운 Google Gen AI SDK 챗봇 ===")
    print("새로운 통합 SDK를 사용하여 Gemini 2.0 모델과 대화하세요!")
    print("종료하려면 'quit', 'exit', '종료'를 입력하세요.")
    print("모델 목록을 보려면 'models'를 입력하세요.")
    print("스트리밍 모드로 응답받으려면 'stream: 질문'을 입력하세요.\n")

    # 클라이언트 설정
    client = setup_client()
    if not client:
        return

    # 기본 모델 설정 (Gemini 2.0 Flash)
    default_model = 'gemini-2.0-flash'

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
                get_available_models(client)
                continue

            # 스트리밍 모드 확인
            if user_input.lower().startswith('stream:'):
                query = user_input[7:].strip()
                if not query:
                    print("스트리밍할 질문을 입력해주세요. 예: stream: 안녕하세요")
                    continue

                print("스트리밍 응답 생성 중...\n")
                print("Gemini: ", end="", flush=True)

                # 스트리밍 응답 생성
                try:
                    for chunk in client.models.generate_content_stream(
                        model=default_model,
                        contents=query
                    ):
                        if chunk.text:
                            print(chunk.text, end="", flush=True)
                    print("\n")
                except Exception as e:
                    print(f"\n스트리밍 오류: {e}")

            else:
                # 일반 응답 생성
                print("응답 생성 중...")

                # 새로운 SDK 방식으로 컨텐츠 생성
                response = client.models.generate_content(
                    model=default_model,
                    contents=user_input,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        top_p=0.9,
                        max_output_tokens=1000,
                        # 안전 설정 예시
                        safety_settings=[
                            types.SafetySetting(
                                category='HARM_CATEGORY_HATE_SPEECH',
                                threshold='BLOCK_ONLY_HIGH'
                            ),
                        ]
                    )
                )

                print(f"\nGemini: {response.text}\n")

                # 토큰 사용량 정보 표시 (있는 경우)
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    print(f"토큰 사용량: {response.usage_metadata}")

            print("-" * 50)

        except KeyboardInterrupt:
            print("\n\n프로그램이 중단되었습니다.")
            break

        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            print("새로운 SDK를 사용하는지 확인하세요: pip install google-genai")
            print("다시 시도해주세요.\n")

if __name__ == "__main__":
    main()
"""
새로운 Google Gen AI SDK를 사용한 수정된 챗봇
실제 API 구조에 맞게 업데이트됨
설치 필요: pip install google-genai python-dotenv pillow
"""
import os
import json
import pathlib
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

def setup_client():
    """클라이언트 설정 및 초기화"""
    try:
        load_dotenv('gemini.env')
        api_key = os.getenv('API_KEY') or os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')

        if not api_key:
            print("오류: API_KEY가 설정되지 않았습니다.")
            return None

        client = genai.Client(api_key=api_key)
        return client

    except Exception as e:
        print(f"클라이언트 설정 오류: {e}")
        return None

# 함수 호출을 위한 도구 함수들
def get_current_weather(location: str) -> str:
    """현재 날씨 정보를 가져옵니다.

    Args:
        location: 도시명, 예: 서울, 부산
    """
    print(f"날씨 정보 조회: {location}")
    # 실제로는 날씨 API를 호출하겠지만, 여기서는 예시 데이터 반환
    return f"{location}의 현재 날씨: 맑음, 기온 22도, 습도 65%"

def get_current_time() -> str:
    """현재 시간을 반환합니다."""
    return datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")

def calculate_math(expression: str) -> str:
    """간단한 수학 계산을 수행합니다.

    Args:
        expression: 계산할 수식 (예: "2+3*4")
    """
    try:
        # 보안을 위해 제한된 계산만 허용
        allowed_chars = set('0123456789+-*/().')
        if not all(c in allowed_chars for c in expression.replace(' ', '')):
            return "안전하지 않은 수식입니다."

        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"계산 오류: {e}"

class AdvancedChatbot:
    def __init__(self, client):
        self.client = client
        self.chat = None
        self.model = 'gemini-2.0-flash'

        # 함수 호출을 위한 도구 정의
        self.tools = [get_current_weather, get_current_time, calculate_math]

    def start_chat(self):
        """새로운 채팅 세션을 시작합니다."""
        try:
            # 실제 API에 맞는 간단한 구조 사용
            self.chat = self.client.chats.create(model=self.model)
            print("새로운 채팅 세션이 시작되었습니다.")
        except Exception as e:
            print(f"채팅 시작 오류: {e}")

    def send_message(self, message, use_tools=False):
        """메시지를 보내고 응답을 받습니다."""
        try:
            if not self.chat:
                self.start_chat()

            if use_tools:
                # 함수 호출 기능을 사용한 단일 API 호출
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=message,
                    config=types.GenerateContentConfig(
                        tools=self.tools
                    )
                )
                return response.text
            else:
                # 일반 채팅 메시지
                response = self.chat.send_message(message)
                return response.text

        except Exception as e:
            return f"오류 발생: {e}"

    def upload_and_analyze_file(self, file_path):
        """파일을 업로드하고 분석합니다."""
        try:
            if not os.path.exists(file_path):
                return "파일을 찾을 수 없습니다."

            # 파일 업로드
            uploaded_file = self.client.files.upload(file=file_path)
            print(f"파일 업로드 완료: {uploaded_file.name}")

            # 파일 분석 요청
            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    "이 파일의 내용을 분석하고 요약해주세요:",
                    uploaded_file
                ]
            )

            return response.text

        except Exception as e:
            return f"파일 처리 오류: {e}"

    def analyze_image(self, image_path):
        """이미지를 분석합니다."""
        try:
            if not os.path.exists(image_path):
                return "이미지 파일을 찾을 수 없습니다."

            # PIL Image로 로드 (새 SDK는 자동 변환 지원)
            image = Image.open(image_path)

            response = self.client.models.generate_content(
                model=self.model,
                contents=[
                    "이 이미지에 대해 자세히 설명해주세요:",
                    image
                ]
            )

            return response.text

        except Exception as e:
            return f"이미지 분석 오류: {e}"

    def stream_response(self, message):
        """스트리밍 응답을 제공합니다."""
        try:
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=message
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            yield f"스트리밍 오류: {e}"

def print_help():
    """도움말을 출력합니다."""
    help_text = """
=== 사용 가능한 명령어 ===
• chat: <메시지>       - 연속 대화 모드
• tool: <메시지>       - 함수 호출 기능 사용
• stream: <메시지>     - 스트리밍 응답
• file: <파일경로>     - 파일 업로드 및 분석
• image: <이미지경로>  - 이미지 분석
• newchat              - 새 채팅 시작
• models               - 사용 가능한 모델 목록
• help                 - 이 도움말 보기
• quit/exit/종료       - 프로그램 종료

=== 함수 호출 예시 ===
• tool: 서울 날씨 어때?
• tool: 지금 몇 시야?
• tool: 2+3*4 계산해줘

=== 파일 분석 예시 ===
• file: document.txt
• image: photo.jpg

=== 스트리밍 예시 ===
• stream: 긴 이야기를 들려줘
    """
    print(help_text)

def main():
    """메인 함수"""
    print("=== Google Gen AI SDK 챗봇 (수정됨) ===")
    print("새로운 통합 SDK의 기능을 체험해보세요!")
    print("도움말을 보려면 'help'를 입력하세요.\n")

    # 클라이언트 설정
    client = setup_client()
    if not client:
        return

    # 챗봇 인스턴스 생성
    chatbot = AdvancedChatbot(client)

    while True:
        try:
            user_input = input("\n명령: ").strip()

            if not user_input:
                continue

            # 종료 명령
            if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                print("챗봇을 종료합니다.")
                break

            # 도움말
            elif user_input.lower() == 'help':
                print_help()

            # 모델 목록
            elif user_input.lower() == 'models':
                try:
                    print("\n사용 가능한 모델:")
                    for model in client.models.list():
                        print(f"- {model.name}")
                except Exception as e:
                    print(f"모델 목록 조회 오류: {e}")

            # 새 채팅 시작
            elif user_input.lower() == 'newchat':
                chatbot.start_chat()

            # 채팅 모드
            elif user_input.lower().startswith('chat:'):
                message = user_input[5:].strip()
                if message:
                    print("응답 생성 중...")
                    response = chatbot.send_message(message)
                    print(f"\nGemini: {response}")

            # 함수 호출 모드
            elif user_input.lower().startswith('tool:'):
                message = user_input[5:].strip()
                if message:
                    print("함수 호출 기능으로 응답 생성 중...")
                    response = chatbot.send_message(message, use_tools=True)
                    print(f"\nGemini: {response}")

            # 스트리밍 모드
            elif user_input.lower().startswith('stream:'):
                message = user_input[7:].strip()
                if message:
                    print("스트리밍 응답 생성 중...\n")
                    print("Gemini: ", end="", flush=True)
                    for chunk in chatbot.stream_response(message):
                        print(chunk, end="", flush=True)
                    print("\n")

            # 파일 분석
            elif user_input.lower().startswith('file:'):
                file_path = user_input[5:].strip()
                if file_path:
                    print("파일 분석 중...")
                    response = chatbot.upload_and_analyze_file(file_path)
                    print(f"\nGemini: {response}")

            # 이미지 분석
            elif user_input.lower().startswith('image:'):
                image_path = user_input[6:].strip()
                if image_path:
                    print("이미지 분석 중...")
                    response = chatbot.analyze_image(image_path)
                    print(f"\nGemini: {response}")

            # 일반 질문 (단일 응답)
            else:
                print("응답 생성 중...")
                response = client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=user_input
                )
                print(f"\nGemini: {response.text}")

        except KeyboardInterrupt:
            print("\n\n프로그램이 중단되었습니다.")
            break

        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            print("SDK 설치를 확인하세요: pip install google-genai")

if __name__ == "__main__":
    main()
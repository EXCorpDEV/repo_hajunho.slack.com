from typing import Optional
from langchain_community.callbacks.manager import get_openai_callback
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain.tools import Tool


def load_api_key(key_path: str = "secret.key") -> str:
    with open(key_path, 'r') as f:
        return f.read().strip()


class AdvancedChatAssistant:
    def __init__(self, key_path: str = "secret.key"):
        """
        고급 기능을 갖춘 ChatAssistant 초기화
        - 대화 기억
        - 웹 검색 기능
        - Wikipedia 검색 기능
        """
        self.api_key = load_api_key(key_path)
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=self.api_key,
            temperature=0.7
        )

        # 대화 기록 초기화
        self.message_history = ChatMessageHistory()

        # 프롬프트 템플릿 설정
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(
                content="You are a helpful AI assistant. Please respond in the same language as the user's input. If the user speaks Korean, respond in Korean. If they speak English, respond in English. Always be clear and easy to understand."),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content="{input}")
        ])

        # 검색 도구 설정
        search = DuckDuckGoSearchAPIWrapper()
        wikipedia = WikipediaAPIWrapper()

        self.tools = {
            "웹검색": search.run,
            "위키백과": wikipedia.run
        }

    def _should_use_search(self, query: str) -> tuple[bool, str]:
        """검색 도구 사용 여부와 도구 종류 결정"""
        search_keywords = ['찾아', '검색', '알려줘', '최근', '뉴스', '날씨']
        wiki_keywords = ['역사', '누구야', '개념', '정의', '설명']

        if any(keyword in query for keyword in search_keywords):
            return True, "웹검색"
        elif any(keyword in query for keyword in wiki_keywords):
            return True, "위키백과"
        return False, ""

    def chat(self, user_input: str) -> str:
        """
        사용자 입력에 따라 적절한 도구를 사용하여 응답
        """
        try:
            with get_openai_callback() as cb:
                use_search, tool_name = self._should_use_search(user_input)

                if use_search:
                    # 검색 도구 사용
                    search_result = self.tools[tool_name](user_input)
                    augmented_input = f"다음 정보를 참고하여 답변해주세요: {search_result}\n\n질문: {user_input}"
                else:
                    augmented_input = user_input

                # 대화 기록에 사용자 입력 추가
                self.message_history.add_user_message(augmented_input)

                # 프롬프트 생성 및 LLM 실행
                messages = self.prompt.format_messages(
                    input=augmented_input,
                    history=self.message_history.messages[:-1]  # 현재 입력을 제외한 이전 대화 기록
                )
                response = self.llm.invoke(messages)

                # 대화 기록에 AI 응답 추가
                self.message_history.add_ai_message(response.content)

                print(f"\n토큰 사용량: {cb.total_tokens} (프롬프트: {cb.prompt_tokens}, 생성: {cb.completion_tokens})")
                print(f"예상 비용: ${cb.total_cost:.5f}")

                return response.content

        except Exception as e:
            import traceback
            print(f"Error details: {traceback.format_exc()}")
            return f"오류가 발생했습니다: {str(e)}"


def main():
    try:
        assistant = AdvancedChatAssistant()
        print("AI 어시스턴트와 대화를 시작합니다. (종료: 'quit' 또는 'exit')")
        print("기능:")
        print("1. 대화 기억")
        print("2. 실시간 웹 검색")
        print("3. Wikipedia 정보 검색")

        while True:
            user_input = input("\n질문: ").strip()

            if user_input.lower() in ['quit', 'exit']:
                print("대화를 종료합니다.")
                break

            if not user_input:
                continue

            response = assistant.chat(user_input)
            print(f"\n답변: {response}")

    except Exception as e:
        print(f"프로그램 실행 중 오류 발생: {str(e)}")
        print("필요한 패키지를 설치했는지 확인해주세요:")
        print("pip install langchain langchain-openai langchain-community wikipedia-api duckduckgo-search")


if __name__ == "__main__":
    main()
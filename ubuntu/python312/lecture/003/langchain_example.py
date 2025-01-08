from typing import Optional
from langchain_community.callbacks.manager import get_openai_callback
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
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
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="당신은 도움이 되는 AI 어시스턴트입니다. 명확하고 이해하기 쉽게 답변해주세요."),
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

        # 대화 체인 설정
        chain = prompt | self.llm

        self.chain_with_history = RunnableWithMessageHistory(
            chain,
            lambda session_id: self.message_history,
            input_messages_key="input",
            history_messages_key="history"
        )

    def _should_use_search(self, query: str) -> tuple[bool, str]:
        """검색 도구 사용 여부와 도구 종류 결정"""
        search_keywords = ['찾아', '검색', '알려줘', '최근', '뉴스']
        wiki_keywords = ['역사', '누구야', '개념', '정의', '설명']

        if any(keyword in query for keyword in search_keywords):
            return True, "웹검색"
        elif any(keyword in query for keyword in wiki_keywords):
            return True, "위키백과"
        return False, ""

    async def chat(self, user_input: str) -> str:
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

                # 대화 처리
                response = await self.chain_with_history.invoke(
                    {"input": augmented_input},
                    {"session_id": "default"}
                )

                print(f"\n토큰 사용량: {cb.total_tokens} (프롬프트: {cb.prompt_tokens}, 생성: {cb.completion_tokens})")
                print(f"예상 비용: ${cb.total_cost:.5f}")

                return response.content

        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"


def main():
    try:
        import asyncio
        assistant = AdvancedChatAssistant()
        print("AI 어시스턴트와 대화를 시작합니다. (종료: 'quit' 또는 'exit')")
        print("기능:")
        print("1. 대화 기억")
        print("2. 실시간 웹 검색")
        print("3. Wikipedia 정보 검색")

        async def chat_loop():
            while True:
                user_input = input("\n질문: ").strip()

                if user_input.lower() in ['quit', 'exit']:
                    print("대화를 종료합니다.")
                    break

                if not user_input:
                    continue

                response = await assistant.chat(user_input)
                print(f"\n답변: {response}")

        asyncio.run(chat_loop())

    except Exception as e:
        print(f"프로그램 실행 중 오류 발생: {str(e)}")
        print("필요한 패키지를 설치했는지 확인해주세요:")
        print("pip install langchain langchain-openai langchain-community wikipedia-api duckduckgo-search")


if __name__ == "__main__":
    main()
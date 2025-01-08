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
        self.api_key = load_api_key(key_path)
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=self.api_key,
            temperature=0.7
        )

        self.message_history = ChatMessageHistory()

        # 개선된 시스템 프롬프트
        system_prompt = """당신은 친절하고 도움이 되는 AI 어시스턴트입니다.
        - 사용자의 언어로 응답하세요 (한국어면 한국어로, 영어면 영어로)
        - 인사에는 친근하게 응답하세요
        - 검색 결과가 제공되면 그 내용을 바탕으로 자연스럽게 설명해주세요
        - 검색 결과를 그대로 복사하지 말고, 이해하기 쉽게 설명해주세요
        - 이전 대화 내용을 참고하여 문맥을 이해하세요"""

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content="{input}")
        ])

        # 검색 도구 설정
        self.search = DuckDuckGoSearchAPIWrapper()
        self.wikipedia = WikipediaAPIWrapper()

    def _process_query(self, query: str) -> tuple[str, str]:
        """
        쿼리 처리 및 검색 방법 결정
        반환: (처리된 쿼리, 검색 방법)
        """
        # 간단한 인사말 처리
        greetings = ['안녕', 'hi', 'hello', '반가워', '잘 지내']
        if any(greeting in query.lower() for greeting in greetings):
            return query, "none"

        # 명시적 검색 요청 처리
        if any(word in query for word in ['찾아', '검색', '알려줘']):
            search_term = query.replace('찾아', '').replace('검색', '').replace('알려줘', '').strip()
            return search_term, "web"

        # 질문형 검색 처리
        question_keywords = ['뭐야', '무엇', '누구', '어떻게', '언제', '어디', '방법']
        if any(keyword in query for keyword in question_keywords):
            return query, "web"

        # 위키백과 검색이 적합한 경우
        wiki_keywords = ['란', '의미', '정의', '설명', '역사']
        if any(keyword in query for keyword in wiki_keywords):
            return query, "wiki"

        return query, "none"

    def chat(self, user_input: str) -> str:
        try:
            with get_openai_callback() as cb:
                # 쿼리 처리
                processed_query, search_type = self._process_query(user_input)

                # 검색 수행
                if search_type == "web":
                    search_result = self.search.run(processed_query)
                    augmented_input = f"다음 정보를 참고하여 자연스럽게 설명해주세요: {search_result}\n\n질문: {user_input}"
                elif search_type == "wiki":
                    search_result = self.wikipedia.run(processed_query)
                    augmented_input = f"다음 Wikipedia 정보를 참고하여 설명해주세요: {search_result}\n\n질문: {user_input}"
                else:
                    augmented_input = user_input

                # 사용자 메시지 저장
                self.message_history.add_user_message(augmented_input)

                # 응답 생성
                messages = self.prompt.format_messages(
                    input=augmented_input,
                    history=self.message_history.messages[:-1]
                )
                response = self.llm.invoke(messages)

                # AI 응답 저장
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
        print("\n사용 방법:")
        print("- 일반적인 대화: 평소처럼 자연스럽게 대화하세요")
        print("- 정보 검색: '찾아줘', '검색해줘', '알려줘' 등을 포함하여 질문하세요")
        print("- 개념 설명: '란?', '의미', '정의' 등을 포함하여 질문하세요")
        print("이전 대화 내용을 기억하며 대화합니다.")

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
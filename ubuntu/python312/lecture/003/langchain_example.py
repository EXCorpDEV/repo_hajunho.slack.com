from typing import Optional
from langchain_community.callbacks.manager import get_openai_callback
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper


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

        # 시스템 프롬프트 설정
        system_prompt = """당신은 신중하고 정확한 정보를 제공하는 AI 어시스턴트입니다.

1. 정보 제공 원칙:
   - 공식적이고 검증된 정보를 우선으로 제공하세요
   - 논란이 있는 내용은 제외하고 객관적 사실만 전달하세요
   - 최신 정보의 경우 "검색 결과에 따르면" 등의 한정어를 사용하세요
   - 불확실한 정보는 제공하지 마세요

2. 인물 정보 제공 시:
   - 현재 공식 직위나 역할을 먼저 언급하세요
   - 주요 경력이나 이력을 간단히 설명하세요
   - 논란이 될 수 있는 내용은 제외하세요
   - 개인적인 평가나 의견은 배제하세요

3. 응답 형식:
   - 한국어로 명확하고 간단히 답변하세요
   - 정보의 출처나 시점을 명시하세요
   - 불확실한 부분은 "확인이 필요합니다"라고 하세요
   - 검색 결과가 없으면 솔직히 모른다고 하세요"""

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content="{input}")
        ])

        # 검색 도구 초기화
        self.search = DuckDuckGoSearchAPIWrapper(region="kr-kr", time='y')
        self.wikipedia = WikipediaAPIWrapper(lang="ko")

    def perform_search(self, query: str, search_type: str = "web") -> str:
        """검색 실행"""
        try:
            # 검색어 전처리
            replacements = [
                "란?", "이란?", "가 뭐야?", "이 뭐야?", "은 뭐야?",
                "는 뭐야?", "에 대해 알려줘", "를 알려줘", "을 알려줘",
                "이 누구야?", "가 누구야?", "은 누구야?", "는 누구야?"
            ]
            search_query = query
            for rep in replacements:
                search_query = search_query.replace(rep, "")
            search_query = search_query.strip()

            print(f"\n[검색 디버그] 원본 쿼리: {query}")
            print(f"[검색 디버그] 전처리된 쿼리: {search_query}")

            if not search_query:
                return ""

            if search_type == "web":
                search_results = []

                # 기본 정보 검색
                official_query = f"{search_query} 프로필 위키백과"
                result1 = self.search.run(official_query)
                if result1:
                    search_results.append(f"기본 정보: {result1}")

                # 직위/직책 검색
                position_query = f"{search_query} 현재 직책 직위"
                result2 = self.search.run(position_query)
                if result2 and result2 != result1:
                    search_results.append(f"현재 직위: {result2}")

                # 공식 발표 검색
                news_query = f"{search_query} 공식 발표 보도자료"
                result3 = self.search.run(news_query)
                if result3 and result3 not in [result1, result2]:
                    search_results.append(f"공식 발표: {result3}")

                return "\n".join(search_results) if search_results else ""

            elif search_type == "wiki":
                result = self.wikipedia.run(search_query)
                if result and "Page id" not in result:
                    return f"Wikipedia: {result}"
                return ""

            return ""

        except Exception as e:
            print(f"[검색 실패] {search_type}: {str(e)}")
            return ""

    def chat(self, user_input: str) -> str:
        try:
            with get_openai_callback() as cb:
                print("\n[처리 시작] 사용자 입력:", user_input)

                # 검색 필요성 확인
                should_search = not any(greeting in user_input.lower()
                                        for greeting in ['안녕', 'hi', 'hello'])

                augmented_input = user_input
                if should_search:
                    # 웹 검색 시도
                    web_result = self.perform_search(user_input, "web")
                    if web_result:
                        augmented_input = (
                            f"다음 정보를 참고하여 자연스럽게 답변해주세요:\n"
                            f"{web_result}\n"
                            f"질문: {user_input}\n"
                            f"위 정보를 바탕으로 쉽게 설명해주세요."
                        )
                    else:
                        # Wikipedia 검색 시도
                        wiki_result = self.perform_search(user_input, "wiki")
                        if wiki_result:
                            augmented_input = (
                                f"다음 Wikipedia 정보를 참고하여 자연스럽게 답변해주세요:\n"
                                f"{wiki_result}\n"
                                f"질문: {user_input}\n"
                                f"위 정보를 바탕으로 쉽게 설명해주세요."
                            )
                        else:
                            augmented_input = (
                                f"다음 질문에 대해 알고 있는 정보를 바탕으로 답변해주세요: {user_input}\n"
                                f"검색 결과가 없다면 솔직히 모른다고 말씀해주세요."
                            )

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

                # 대화 기록 출력
                print("\n=== 현재 대화 기록 ===")
                for i, msg in enumerate(self.message_history.messages[-4:], 1):
                    print(f"{i}. {'사용자' if isinstance(msg, HumanMessage) else 'AI'}: {msg.content[:50]}...")

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
        print("\n질문 방법:")
        print("1. 개념 설명: '...란?', '...이 뭐야?', '...에 대해 알려줘'")
        print("2. 일반 검색: 궁금한 것을 자연스럽게 물어보세요")
        print("3. 이전 대화: '아까 무슨 얘기했지?', '이전 답변이 뭐였어?' 등")

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
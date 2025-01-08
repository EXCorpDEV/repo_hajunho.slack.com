from typing import Optional
from langchain_community.callbacks.manager import get_openai_callback
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
import time


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
        system_prompt = """당신은 친절하고 도움이 되는 한국어 AI 어시스턴트입니다.

1. 기본 응답:
   - 항상 한국어로 답변하세요
   - 친절하고 자연스러운 어투를 사용하세요
   - 전문용어는 쉽게 풀어서 설명하세요

2. 검색 결과 활용:
   - 검색 결과가 있다면 그 내용을 바탕으로 종합적으로 설명해주세요
   - 검색 결과를 그대로 복사하지 말고 자연스럽게 재구성하세요
   - 최신 정보와 관련된 질문에는 검색 결과의 시점을 언급해주세요

3. 대화 관리:
   - 이전 대화 내용을 참고하여 문맥을 유지하세요
   - 이전 응답에 오류가 있었다면 새로운 정보로 수정하세요
   - 모르는 내용은 솔직히 모른다고 말씀하세요

4. 특별 지침:
   - 인물 설명 시 주요 경력과 현재 역할을 중심으로 설명하세요
   - 개념 설명 시 정의, 특징, 예시 순으로 구성하세요
   - 검색 실패 시 일반적인 설명이라도 제공하도록 노력하세요"""

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content="{input}")
        ])

        # 검색 도구 초기화
        self.search = DuckDuckGoSearchAPIWrapper(
            region="kr-kr",
            time='y',
            max_results=3  # 최대 3개의 결과만 가져오기
        )
        self.wikipedia = WikipediaAPIWrapper(
            lang="ko",
            top_k_results=1,  # 가장 관련성 높은 결과 1개만 가져오기
            load_max_docs=1
        )

    def safe_search(self, query: str, search_type: str = "web") -> str:
        """안전한 검색 수행"""
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

            if not search_query:
                return ""

            if search_type == "web":
                # 한국어 검색을 위한 키워드 추가
                return self.search.run(f"{search_query} 정의 설명 개념")
            else:  # wiki
                try:
                    # Wikipedia 검색 시도
                    result = self.wikipedia.run(search_query)
                    if not result or "Page id" in result:  # 검색 실패 시
                        # 웹 검색으로 폴백
                        return self.search.run(f"{search_query} wikipedia")
                    return result
                except Exception as wiki_error:
                    print(f"Wikipedia 검색 실패: {str(wiki_error)}")
                    # 웹 검색으로 폴백
                    return self.search.run(f"{search_query} wikipedia")
        except Exception as e:
            print(f"검색 실패 ({search_type}): {str(e)}")
            return ""

    def chat(self, user_input: str) -> str:
        try:
            with get_openai_callback() as cb:
                # 검색 시도
                web_result = self.safe_search(user_input, "web")
                time.sleep(1)  # API 호출 간 간격 추가
                wiki_result = self.safe_search(user_input, "wiki")

                # 검색 결과 조합
                search_results = []
                if web_result:
                    search_results.append(f"웹 검색 결과: {web_result}")
                if wiki_result:
                    search_results.append(f"Wikipedia 결과: {wiki_result}")

                if search_results:
                    augmented_input = (
                        f"다음 정보를 참고하여 질문에 답변해주세요:\n"
                        f"{'\n'.join(search_results)}\n"
                        f"질문: {user_input}"
                    )
                else:
                    augmented_input = user_input

                # 사용자 메시지 저장
                self.message_history.add_user_message(user_input)  # 원래 질문 저장

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
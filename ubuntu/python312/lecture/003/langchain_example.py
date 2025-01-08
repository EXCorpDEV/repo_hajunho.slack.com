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

            print(f"\n[검색 디버그] 원본 쿼리: {query}")
            print(f"[검색 디버그] 전처리된 쿼리: {search_query}")

            if not search_query:
                print("[검색 디버그] 검색어가 비어있음")
                return ""

            if search_type == "web":
                try:
                    # 한국어 검색을 위한 키워드 추가
                    search_results = []

                    # 일반 검색
                    result1 = self.search.run(f"{search_query}")
                    if result1:
                        search_results.append(f"기본 검색: {result1}")

                    # 개념 검색
                    result2 = self.search.run(f"{search_query} 정의 설명 개념")
                    if result2 and result2 != result1:
                        search_results.append(f"개념 검색: {result2}")

                    # 최신 정보 검색
                    result3 = self.search.run(f"{search_query} 현재 최신")
                    if result3 and result3 not in [result1, result2]:
                        search_results.append(f"최신 정보: {result3}")

                    print(f"[검색 디버그] 웹 검색 결과 수: {len(search_results)}")
                    return "\n".join(search_results)

                except Exception as web_error:
                    print(f"[검색 디버그] 웹 검색 실패: {str(web_error)}")
                    return ""

            else:  # wiki
                try:
                    # Wikipedia 검색 시도
                    print("[검색 디버그] Wikipedia 검색 시도")
                    result = self.wikipedia.run(search_query)

                    if not result or "Page id" in result:  # 검색 실패 시
                        print("[검색 디버그] Wikipedia 검색 실패, 웹 검색으로 전환")
                        web_result = self.search.run(f"{search_query} wikipedia")
                        if web_result:
                            return f"Wikipedia 대체 검색: {web_result}"
                        return ""

                    print("[검색 디버그] Wikipedia 검색 성공")
                    return f"Wikipedia: {result}"

                except Exception as wiki_error:
                    print(f"[검색 디버그] Wikipedia 오류: {str(wiki_error)}")
                    return ""

        except Exception as e:
            print(f"[검색 디버그] 전체 검색 실패: {str(e)}")
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
                    print("\n[검색 시작] 웹 검색")
                    web_result = self.safe_search(user_input, "web")
                    if web_result:
                        print("[검색 성공] 웹 검색 결과 있음")
                        augmented_input = (
                            f"다음 정보를 참고하여 자연스럽게 답변해주세요:\n"
                            f"{web_result}\n"
                            f"질문: {user_input}\n"
                            f"위 정보를 바탕으로 쉽게 설명해주세요."
                        )
                    else:
                        print("[검색 실패] 웹 검색 결과 없음")

                        # Wikipedia 검색 시도
                        print("\n[검색 시작] Wikipedia 검색")
                        wiki_result = self.safe_search(user_input, "wiki")
                        if wiki_result:
                            print("[검색 성공] Wikipedia 검색 결과 있음")
                            augmented_input = (
                                f"다음 Wikipedia 정보를 참고하여 자연스럽게 답변해주세요:\n"
                                f"{wiki_result}\n"
                                f"질문: {user_input}\n"
                                f"위 정보를 바탕으로 쉽게 설명해주세요."
                            )
                        else:
                            print("[검색 실패] Wikipedia 검색 결과 없음")
                            augmented_input = (
                                f"다음 질문에 대해 알고 있는 정보를 바탕으로 답변해주세요: {user_input}\n"
                                f"검색 결과가 없다면 솔직히 모른다고 말씀해주세요."
                            )

                # 검색 결과를 포함한 전체 메시지 저장
                self.message_history.add_user_message(augmented_input)

                # 최근 5개의 메시지만 유지 (2-3번의 대화)
                if len(self.message_history.messages) > 10:
                    # 메시지 목록 초기화 후 최근 메시지만 유지
                    recent_messages = self.message_history.messages[-10:]
                    self.message_history = ChatMessageHistory()
                    for msg in recent_messages:
                        if isinstance(msg, HumanMessage):
                            self.message_history.add_user_message(msg.content)
                        else:
                            self.message_history.add_ai_message(msg.content)

                # 응답 생성
                messages = self.prompt.format_messages(
                    input=augmented_input,
                    history=self.message_history.messages[:-1]
                )
                response = self.llm.invoke(messages)

                # AI 응답 저장
                self.message_history.add_ai_message(response.content)

                # 디버깅용 메시지 히스토리 출력
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
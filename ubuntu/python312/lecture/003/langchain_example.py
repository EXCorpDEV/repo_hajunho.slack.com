from typing import Optional
import os
import traceback

# === (1) langchain & community/openai 패키지 가져오기 ===
# 공식 LangChain
from langchain.schema import HumanMessage, SystemMessage

# langchain_community (DuckDuckGoSearchAPIWrapper, ChatMessageHistory 등)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

# langchain_openai (최신 ChatOpenAI)
from langchain_openai import ChatOpenAI

# === (2) Prompt 관련 ===
# 공식 LangChain에서 제공하는 ChatPromptTemplate, MessagesPlaceholder
# (만약 langchain_core가 아닌, langchain.prompts.chat 쪽에서 불러오는 버전이 맞다면 해당 경로 사용)
from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder


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

        # --- 시스템 프롬프트 설정 ---
        system_prompt = """당신은 신중하고 정확한 정보를 제공하는 AI 어시스턴트입니다.

1. 검색 결과 처리:
   - 제공된 검색 결과를 반드시 활용하여 답변을 구성하세요
   - 검색 결과에서 가장 관련성 높은 정보를 선택하세요
   - 검색 결과가 불충분하면 추가 검색을 요청하세요
   - 검색 결과를 이해하기 쉽게 재구성하여 설명하세요

2. 인물 정보 제공 시:
   - 현재 공식 직위나 역할을 먼저 언급하세요
   - 주요 경력이나 이력을 간단히 설명하세요
   - 논란이 될 수 있는 내용은 제외하세요
   - 개인적인 평가나 의견은 배제하세요

3. 응답 형식:
   - 한국어로 명확하고 간단히 답변하세요
   - 정보의 출처나 시점을 명시하세요
   - 불확실한 부분은 "확인이 필요합니다"라고 하세요
   - 검색 결과가 없으면 솔직히 모른다고 하세요
        """

        # ChatPromptTemplate.from_messages 구성
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="history"),
            # 이 위치에서 {input}을 그대로 쓰려면 HumanMessage(content="{input}")를 써야 하는데,
            # 아래 chat() 메서드에서 augmented_input을 user 메시지로 추가하기 때문에
            # 실질적으로는 PromptTemplate의 human prompt 부분이 잘 활용되는지 확인 필요
            HumanMessage(content="{input}")
        ])

        # --- 검색 도구 초기화 ---
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

                # 위키백과 정보 검색
                wiki_query = f"{search_query} site:wikipedia.org"
                result1 = self.search.run(wiki_query)
                if result1:
                    search_results.append(f"위키백과 정보: {result1}")

                # 공식 프로필 검색
                profile_query = f"{search_query} 프로필 경력 이력"
                result2 = self.search.run(profile_query)
                if result2 and result2 != result1:
                    search_results.append(f"프로필 정보: {result2}")

                # 현재 정보 검색
                current_query = f"{search_query} 현재 소속 직책 2024"
                result3 = self.search.run(current_query)
                if result3 and result3 not in [result1, result2]:
                    search_results.append(f"현재 정보: {result3}")

                print(f"\n[검색 결과]")
                for idx, res in enumerate(search_results, start=1):
                    print(f"{idx}. {res[:100]}...")

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
        # 여기서 get_openai_callback()은 langchain_community.callbacks.manager에 있는 것으로 가정
        from langchain_community.callbacks.manager import get_openai_callback

        try:
            with get_openai_callback() as cb:
                print("\n[처리 시작] 사용자 입력:", user_input)

                # 검색 여부 결정 (간단 조건 예시)
                should_search = not any(greeting in user_input.lower() for greeting in ['안녕', 'hi', 'hello'])

                augmented_input = user_input
                if should_search:
                    # 1) 웹 검색 시도
                    web_result = self.perform_search(user_input, "web")
                    if web_result:
                        augmented_input = (
                            f"다음 정보를 바탕으로 답변을 작성해주세요. 반드시 제공된 정보를 활용하여 응답하세요:\n"
                            f"{web_result}\n\n"
                            f"질문: {user_input}\n\n"
                            f"답변 형식:\n"
                            f"1. 현재 직위/역할\n"
                            f"2. 주요 경력\n"
                            f"3. 최근 활동/현재 상황\n\n"
                            f"위 정보를 바탕으로 객관적이고 명확하게 설명해주세요."
                        )
                    else:
                        # 2) 위키백과 검색 시도
                        wiki_result = self.perform_search(user_input, "wiki")
                        if wiki_result:
                            augmented_input = (
                                f"다음 Wikipedia 정보를 참고하여 자연스럽게 답변해주세요:\n"
                                f"{wiki_result}\n\n"
                                f"질문: {user_input}\n"
                                f"위 정보를 바탕으로 쉽게 설명해주세요."
                            )
                        else:
                            # 3) 검색 결과도 없을 시
                            augmented_input = (
                                f"다음 질문에 대해 알고 있는 정보를 바탕으로 답변해주세요: {user_input}\n"
                                f"검색 결과가 없다면 솔직히 모른다고 말씀해주세요."
                            )

                # 사용자 메시지 기록
                self.message_history.add_user_message(augmented_input)

                # 메시지 변환: history는 직전에 추가한 메시지를 제외하기 위해 [:-1]
                messages = self.prompt.format_messages(
                    input=augmented_input,
                    history=self.message_history.messages[:-1]
                )
                response = self.llm.invoke(messages)

                # AI 응답 기록
                self.message_history.add_ai_message(response.content)

                # 대화 기록 확인
                print("\n=== 현재 대화 기록 (최근 4개) ===")
                for i, msg in enumerate(self.message_history.messages[-4:], 1):
                    who = "사용자" if isinstance(msg, HumanMessage) else "AI"
                    print(f"{i}. {who}: {msg.content[:50]}...")

                print(f"\n토큰 사용량: {cb.total_tokens} "
                      f"(프롬프트: {cb.prompt_tokens}, 생성: {cb.completion_tokens})")
                print(f"예상 비용: ${cb.total_cost:.5f}")

                return response.content

        except Exception as e:
            print(f"Error details: {traceback.format_exc()}")
            return f"오류가 발생했습니다: {str(e)}"


def main():
    try:
        assistant = AdvancedChatAssistant()
        print("AI 어시스턴트와 대화를 시작합니다. (종료: 'quit' 또는 'exit')")

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

from typing import Optional
import os
import traceback

# === (1) langchain & community/openai 패키지 가져오기 ===
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI

# prompt 관련
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
        self.current_subject = None  # <--- 이번 예시에서 '윤석열' 같은 대상 보관

        system_prompt = """당신은 정확한 정보를 제공하는 AI 어시스턴트입니다.

1. 검색 결과:
   - 제공된 검색 결과를 반드시 활용하고, 그 내용을 답변에 전부 노출하세요.
   - 검색 결과가 길 경우 핵심 요약을 함께 해주세요.

2. 맥락 이해:
   - 직전에 언급된 주제가 있으면, "그의", "그녀의" 등의 대명사가 해당 주제를 가리킨다고 가정하세요.

3. 응답 형식:
   - 한국어로 간결하고 정확하게 답하세요.
   - 검색 결과가 전혀 없을 경우, 솔직히 모른다고 해주세요.
        """

        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="history"),
            # 주의: 아래 "{input}" 부분이 실제로 user 메시지를 전달받을 자리입니다.
            HumanMessage(content="{input}")
        ])

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
                    search_results.append(f"[위키백과 정보]\n{result1}")

                # 공식 프로필 검색
                profile_query = f"{search_query} 프로필 경력 이력"
                result2 = self.search.run(profile_query)
                if result2 and result2 != result1:
                    search_results.append(f"[프로필 정보]\n{result2}")

                # 현재 정보 검색
                current_query = f"{search_query} 현재 소속 직책 2024"
                result3 = self.search.run(current_query)
                if result3 and result3 not in [result1, result2]:
                    search_results.append(f"[현재 정보]\n{result3}")

                print(f"\n[검색 결과 목록]")
                for idx, res in enumerate(search_results, start=1):
                    preview = (res[:200] + "...") if len(res) > 200 else res
                    print(f"{idx}.\n{preview}\n")

                return "\n\n".join(search_results) if search_results else ""

            elif search_type == "wiki":
                result = self.wikipedia.run(search_query)
                if result and "Page id" not in result:
                    return f"[Wikipedia]\n{result}"
                return ""

            return ""

        except Exception as e:
            print(f"[검색 실패] {search_type}: {str(e)}")
            return ""

    def handle_pronouns(self, user_input: str) -> str:
        """
        사용자가 '그의', '그녀의' 등 대명사를 썼을 때,
        직전에 저장된 self.current_subject가 있으면 치환해줌.
        """
        # 단순 예시: "그의" -> "{self.current_subject}의"
        # 복잡한 대명사는 여기서 추가 처리
        if self.current_subject:
            user_input = user_input.replace("그의", f"{self.current_subject}의")
            user_input = user_input.replace("그녀의", f"{self.current_subject}의")
        return user_input

    def parse_subject(self, user_input: str):
        """
        사용자가 새롭게 특정 인물을 언급했다면 그 인물을 찾아 self.current_subject에 저장.
        여기서는 아주 단순하게 '윤석열'처럼 고유명사 하나를 찾는 예시.
        실제로는 NER(Named Entity Recognition) 등을 활용해야 더 정확합니다.
        """
        possible_subjects = ["윤석열", "이재명", "문재인", "트럼프", "바이든"]  # 예시
        for subj in possible_subjects:
            if subj in user_input:
                self.current_subject = subj
                break

    def chat(self, user_input: str) -> str:
        from langchain_community.callbacks.manager import get_openai_callback

        try:
            print("\n[처리 시작] 사용자 입력:", user_input)

            # 1) 사용자 입력에서 대명사(그의, 그녀의)가 있으면 이전 subject로 치환
            processed_input = self.handle_pronouns(user_input)

            # 2) 새 subject가 있는지 파악
            self.parse_subject(processed_input)

            # 3) 검색 필요 여부(간단 규칙)
            should_search = not any(greeting in processed_input.lower() for greeting in ['안녕', 'hi', 'hello'])

            augmented_input = processed_input
            if should_search:
                # 3-1) 웹 검색
                web_result = self.perform_search(processed_input, "web")
                if web_result:
                    # 검색 결과를 전부 답변에 포함시키라고 지시
                    augmented_input = (
                        f"아래는 검색 결과입니다:\n\n"
                        f"{web_result}\n\n"
                        f"위 검색 결과 내용을 전부 인용해 답변해 주세요. "
                        f"질문: {processed_input}\n\n"
                        f"1) 검색 결과를 전부 보여주고\n"
                        f"2) 추가 설명이 있다면 꼭 작성\n"
                        f"3) 만약 부족하면 wiki 검색도 할 것."
                    )
                else:
                    # 3-2) 위키백과 검색
                    wiki_result = self.perform_search(processed_input, "wiki")
                    if wiki_result:
                        augmented_input = (
                            f"아래는 위키백과 검색 결과입니다:\n\n"
                            f"{wiki_result}\n\n"
                            f"위 정보를 전부 보여주고, 요약해서 답변해 주세요.\n"
                            f"질문: {processed_input}"
                        )
                    else:
                        # 검색 결과 없을 때
                        augmented_input = (
                            f"검색 결과가 전혀 없습니다. 알고 있는 정보만으로 답변해 주세요.\n"
                            f"질문: {processed_input}"
                        )

            # 4) 메시지 기록(대화 히스토리)
            self.message_history.add_user_message(augmented_input)

            # 5) LLM 호출
            with get_openai_callback() as cb:
                messages = self.prompt.format_messages(
                    input=augmented_input,
                    history=self.message_history.messages[:-1]
                )
                response = self.llm.invoke(messages)

                # 6) AI 응답 기록
                self.message_history.add_ai_message(response.content)

                # 7) 디버그 출력
                print("\n=== 대화 기록 (최근 4개 메시지) ===")
                for i, msg in enumerate(self.message_history.messages[-4:], 1):
                    who = "사용자" if isinstance(msg, HumanMessage) else "AI"
                    snippet = (msg.content[:60] + "...") if len(msg.content) > 60 else msg.content
                    print(f"{i}. [{who}] {snippet}")

                print(f"\n[토큰 사용량] 총 {cb.total_tokens} "
                      f"(프롬프트: {cb.prompt_tokens}, 생성: {cb.completion_tokens})")
                print(f"[예상 비용] ${cb.total_cost:.5f}")

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
            print(f"\n[AI 답변]\n{response}")

    except Exception as e:
        print(f"프로그램 실행 중 오류 발생: {str(e)}")
        print("아래 명령어로 필요한 패키지를 설치했는지 확인해주세요:\n")
        print("pip install langchain langchain-openai langchain-community wikipedia-api duckduckgo-search")


if __name__ == "__main__":
    main()

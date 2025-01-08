import os
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.callbacks import get_openai_callback


def load_api_key(key_path: str = "secret.key") -> str:
    """
    secret.key 파일에서 API 키를 로드
    Args:
        key_path: API 키가 저장된 파일 경로
    Returns:
        str: API 키
    Raises:
        FileNotFoundError: 키 파일이 없는 경우
        ValueError: 키 파일이 비어있는 경우
    """
    try:
        with open(key_path, 'r') as f:
            api_key = f.read().strip()
            if not api_key:
                raise ValueError("API 키 파일이 비어있습니다.")
            return api_key
    except FileNotFoundError:
        raise FileNotFoundError(f"'{key_path}' 파일을 찾을 수 없습니다. API 키를 파일에 저장해주세요.")


class ChatAssistant:
    def __init__(self, key_path: str = "secret.key"):
        """
        ChatAssistant 초기화
        Args:
            key_path: API 키가 저장된 파일 경로
        """
        try:
            self.api_key = load_api_key(key_path)
            self._initialize_chain()
        except (FileNotFoundError, ValueError) as e:
            raise RuntimeError(f"초기화 실패: {str(e)}")

    def _initialize_chain(self):
        """체인 초기화"""
        # 시스템 & 사용자 프롬프트 설정
        system_prompt = SystemMessagePromptTemplate.from_template(
            "당신은 도움이 되는 AI 어시스턴트입니다. 명확하고 이해하기 쉽게 답변해주세요."
        )
        human_prompt = HumanMessagePromptTemplate.from_template(
            "{question}"
        )

        # ChatPromptTemplate 생성
        self.chat_prompt = ChatPromptTemplate.from_messages([
            system_prompt,
            human_prompt
        ])

        # LLM 설정
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            openai_api_key=self.api_key,
            temperature=0.7,
            request_timeout=30
        )

        # 체인 구성
        self.chain = self.chat_prompt | self.llm

    def ask(self, question: str) -> str:
        """
        질문하고 답변 받기
        Args:
            question: 사용자 질문
        Returns:
            str: AI의 답변
        """
        try:
            # OpenAI 콜백을 통해 토큰 사용량 추적
            with get_openai_callback() as cb:
                response = self.chain.invoke({"question": question})
                print(f"\n토큰 사용량: {cb.total_tokens} (프롬프트: {cb.prompt_tokens}, 생성: {cb.completion_tokens})")
                print(f"예상 비용: ${cb.total_cost:.5f}")

            return response.content

        except Exception as e:
            print(f"Error: {str(e)}")
            return f"오류가 발생했습니다: {str(e)}"


def main():
    try:
        # ChatAssistant 인스턴스 생성
        assistant = ChatAssistant()

        # 대화 루프
        print("AI 어시스턴트와 대화를 시작합니다. 종료하려면 'quit' 또는 'exit'를 입력하세요.")

        while True:
            question = input("\n질문: ").strip()

            if question.lower() in ['quit', 'exit']:
                print("대화를 종료합니다.")
                break

            if not question:
                continue

            answer = assistant.ask(question)
            print(f"\n답변: {answer}")

    except Exception as e:
        print(f"프로그램 실행 중 오류 발생: {str(e)}")


if __name__ == "__main__":
    main()
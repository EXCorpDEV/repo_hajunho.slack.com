import openai
import os


# API 키 파일에서 읽기
def load_api_key(key_file):
    try:
        with open(key_file, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: {key_file} 파일을 찾을 수 없습니다.")
        return None


# OpenAI API 초기화
def initialize_openai():
    api_key = load_api_key('ex.key')
    if api_key:
        openai.api_key = api_key
        return True
    return False


# GPT-4에게 질문하기
def ask_gpt4(question):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"오류가 발생했습니다: {str(e)}"


def main():
    if not initialize_openai():
        return

    print("GPT-4 질문 프로그램 (종료하려면 'quit' 또는 'exit' 입력)")
    print("-" * 50)

    while True:
        question = input("\n질문을 입력하세요: ")
        if question.lower() in ['quit', 'exit']:
            print("프로그램을 종료합니다.")
            break

        print("\nGPT-4의 응답:")
        print("-" * 50)
        response = ask_gpt4(question)
        print(response)
        print("-" * 50)


if __name__ == "__main__":
    main()
import requests

def list_models():
    api_key = input("Enter your OpenAI API key: ")

    url = "https://api.openai.com/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        for model in data["data"]:
            print(model["id"])
    else:
        print("Failed to retrieve models:")
        print(response.status_code, response.text)

if __name__ == "__main__":
    list_models()

# dall-e-2
# gpt-4o-mini-2024-07-18
# gpt-4o-mini
# gpt-3.5-turbo
# gpt-3.5-turbo-0125
# gpt-3.5-turbo-instruct
# babbage-002
# o1-mini
# o1-mini-2024-09-12
# whisper-1
# dall-e-3
# gpt-3.5-turbo-16k
# omni-moderation-latest
# o1-preview-2024-09-12
# omni-moderation-2024-09-26
# tts-1-hd-1106
# o1-preview
# tts-1-hd
# davinci-002
# text-embedding-ada-002
# tts-1
# tts-1-1106
# gpt-3.5-turbo-instruct-0914
# text-embedding-3-small
# text-embedding-3-large
# gpt-3.5-turbo-1106
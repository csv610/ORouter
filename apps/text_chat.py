import os
import sys
import random

from openai import OpenAI

class OpenRouterClient:
    """
    A client for interacting with the OpenRouter API.
    """
    MODELS = [
        "deepseek/deepseek-chat-v3.1:free",
        "mistralai/mistral-small-3.2-24b-instruct:free",
        "moonshotai/kimi-dev-72b:free",
        "meta-llama/llama-3.3-8b-instruct:free",
        "nvidia/nemotron-nano-9b-v2:free",
        "openai/gpt-oss-20b:free",
        "qwen/qwen3-14b:free",
        "qwen/qwen3-30b-a3b:free",
        "qwen/qwen3-235b-a22b:free",
        "tencent/hunyuan-a13b-instruct:free",
        "x-ai/grok-4-fast:free",
        "z-ai/glm-4.5-air:free",
    ]

    def __init__(self):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )

    def get_models(self):
        """
        Returns the list of available models.
        """
        return self.MODELS

    def generate_text(self, user_query, model=None):
        """
        Gets a completion from the OpenRouter API.
        """
        if model is None:
            model = random.choice(self.get_models())
        print(f"Using model: {model}")

        completion = self.client.chat.completions.create(
            extra_body={},
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": user_query
                }
            ]
        )
        return completion.choices[0].message.content

def main():
    """
    Main function to get a response from OpenRouter.
    """
    if len(sys.argv) > 1:
        user_query = sys.argv[1]
    else:
        user_query = "Hello, who are you?"

    client = OpenRouterClient()
    response = client.generate_text(user_query)
    print(response)

if __name__ == "__main__":
    main()

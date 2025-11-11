
from openai import OpenAI
import os
import argparse
import sys

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key= os.getenv("OPENROUTER_API_KEY")
)

MODELS = {
    "deepseek": "deepseek/deepseek-chat-v3.1:free",
    "deepseek-r1": "deepseek/deepseek-r1:free",
    "gemma27b": "google/gemma-3-27b-it:free",
    "glm": "z-ai/glm-4.5-air:free",
    "gpt-oss": "openai/gpt-oss-20b:free",
    "gemma3n": "google/gemma-3n-e4b-it:free",
    "grok": "x-ai/grok-4-fast:free",
    "haiku": "anthropic/claude-haiku-4.5",
    "hunyuan": "tencent/hunyuan-a13b-instruct:free",
    "kimi": "moonshotai/kimi-dev-72b:free",
    "llama3": "meta-llama/llama-3.3-8b-instruct:free",
    "llama4": "meta-llama/llama-4-maverick:free",
    "mistral": "mistralai/mistral-small-3.2-24b-instruct:free",
    "minimax": "minimax/minimax-m2:free",
    "nemotron9b": "nvidia/nemotron-nano-9b-v2:free",
    "nemotron12b": "nvidia/nemotron-nano-12b-v2-vl:free",
    "qwen-14b": "qwen/qwen3-14b:free",
    "qwen-30b": "qwen/qwen3-30b-a3b:free",
    "qwen-235b": "qwen/qwen3-235b-a22b:free",
    "sonnet"   : "anthropic/claude-sonnet-4.5",
    "sonar": "perplexity/sonar", 
    "sonar-pro": "perplexity/sonar-pro", 
    "sonar-research": "perplexity/sonar-deep-research",
    "sonar-search": "perplexity/sonar-pro-search",
    "sonar-reason": "perplexity/sonar-reasoning-pro",
}

def ask_llm(query, model_name="deepseek"):
    """
    Query the LLM with a given prompt.

    Args:
        query (str): The query to send to the model
        model_name (str): The model alias (default: deepseek)

    Returns:
        str: The response from the LLM
    """
    if model_name not in MODELS:
        raise ValueError(f"Model '{model_name}' not found. Available models: {', '.join(MODELS.keys())}")

    model = MODELS[model_name]

    # Define text and image parts separately
    text_part = {
      "type": "text",
      "text": query
    }
    content = [text_part]

    """
    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"

    # Build content array - only add image if URL is present
    if image_url:
      image_part = {
        "type": "image_url",
        "image_url": {
          "url": image_url
        }
      }
      content.append(image_part)
    """

    completion = client.chat.completions.create(
      extra_body={},
      model=model,
      messages=[
        {
          "role": "user",
          "content": content
        }
      ]
    )

    return completion.choices[0].message.content


def main():
    """Main function to parse arguments and query the LLM."""
    parser = argparse.ArgumentParser(description="Query OpenRouter API")

    parser.add_argument("-q", "--query", help="The query to send to the model")

    parser.add_argument(
        "-m", "--model",
        choices=list(MODELS.keys()),
        default="deepseek",
        help="Model to use (default: deepseek)"
    )

    args = parser.parse_args()
    response = ask_llm(args.query, args.model)
    print(response)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Test script for text generation with OpenRouterClient."""

import sys
from pathlib import Path

# Add parent directory to path so we can import orouter
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from orouter import OpenRouterClient, ModelConfig, ModelInput

def query_model(question: str, model: str = None) -> bool:
    try:
        client = OpenRouterClient()
        input_data = ModelInput(user_prompt=question)
        config = ModelConfig(model=model or "openai/gpt-oss-20b:free")

        response = client.generate_content(input_data, config)
        print(f"\nQuestion: {question}")
        if model:
            print(f"Model: {model}")
        print(f"Response: {response}\n")
        return True
    except Exception as e:
        print(f"ERROR: {e}\n")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OpenRouter client for text generation"
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        help="Question to ask the model"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default='gpt-oss',
        help="Model to use (e.g., 'haiku', 'opus')"
    )

    args = parser.parse_args()

    answer = query_model(args.question, args.model)
    print(answer)

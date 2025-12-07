#!/usr/bin/env python3
"""Test script for vision capabilities with OpenRouterClient."""

import sys
from pathlib import Path

# Add parent directory to path so we can import orouter
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from orouter import OpenRouterClient, ModelConfig, ModelInput

def query_model(prompt: str, image_source: str = None, model: str = None) -> bool:
    try:
        client = OpenRouterClient()
        input_data = ModelInput(user_prompt=prompt, image_source=image_source)
        config = ModelConfig(model=model or "google/gemma-3-27b-it:free")

        response = client.generate_content(input_data, config)
        print(f"\nPrompt: {prompt}")
        if image_source:
            print(f"Image: {image_source[:50]}...")
        if model:
            print(f"Model: {model}")
        print(f"Response: {response}\n")
        return True
    except Exception as e:
        print(f"ERROR: {e}\n")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OpenRouter client for vision queries"
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default='Describe the image in detail',
        help="Prompt/question for the vision model"
    )
    parser.add_argument(
        "-i", "--image",
        type=str,
        help="Image source (URL, file path, or data URI)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default='gemma27b',
        help="Model to use (e.g., 'gemma27b')"
    )

    args = parser.parse_args()

    answer = query_model(args.prompt, args.image, args.model)
    print(answer)

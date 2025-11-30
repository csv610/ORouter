#!/usr/bin/env python3
"""Test script for text generation with OpenRouterClient."""

from openrouter_client import OpenRouterClient, ModelConfig, ModelInput

def test_basic_text():
    """Test basic text generation with a simple prompt."""
    print("=" * 60)
    print("TEST 1: Basic Text Generation")
    print("=" * 60)

    try:
        client = OpenRouterClient()
        input_data = ModelInput(prompt="What is 2+2?")
        response = client.generate_text(input_data)
        print(f"\nPrompt: What is 2+2?")
        print(f"Response: {response}\n")
        return True
    except Exception as e:
        print(f"ERROR: {e}\n")
        return False

def test_with_model_alias():
    """Test text generation with a specific model alias."""
    print("=" * 60)
    print("TEST 2: Text Generation with Model Alias")
    print("=" * 60)

    try:
        client = OpenRouterClient()
        input_data = ModelInput(
            prompt="Explain quantum computing in one sentence.",
            model="haiku"
        )
        response = client.generate_text(input_data)
        print(f"Model used: haiku")
        print(f"Response: {response}\n")
        return True
    except Exception as e:
        print(f"ERROR: {e}\n")
        return False

def test_with_system_prompt():
    """Test text generation with system prompt."""
    print("=" * 60)
    print("TEST 3: Text Generation with System Prompt")
    print("=" * 60)

    try:
        client = OpenRouterClient()
        input_data = ModelInput(
            prompt="What is Python?",
            system_prompt="You are a helpful assistant."
        )
        response = client.generate_text(input_data)
        print(f"Response: {response}\n")
        return True
    except Exception as e:
        print(f"ERROR: {e}\n")
        return False

def test_with_config():
    """Test text generation with custom config."""
    print("=" * 60)
    print("TEST 4: Text Generation with Custom Config")
    print("=" * 60)

    try:
        config = ModelConfig(
            temperature=0.5,
            max_tokens=100
        )
        client = OpenRouterClient(config=config)
        input_data = ModelInput(prompt="Tell me about AI.")
        response = client.generate_text(input_data)
        print(f"Temperature: 0.5, Max tokens: 100")
        print(f"Response: {response}\n")
        return True
    except Exception as e:
        print(f"ERROR: {e}\n")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("OPENROUTER CLIENT TEXT GENERATION TESTS")
    print("=" * 60 + "\n")

    results = []
    results.append(("Basic Text", test_basic_text()))
    results.append(("Model Alias", test_with_model_alias()))
    results.append(("System Prompt", test_with_system_prompt()))
    results.append(("Custom Config", test_with_config()))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    total_passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed\n")

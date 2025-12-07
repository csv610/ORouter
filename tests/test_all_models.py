#!/usr/bin/env python3
"""Test script to verify all available models work with a test prompt."""

from openrouter_client import OpenRouterClient, ModelInput

def test_all_models():
    """Test all text and vision models with a simple prompt."""
    client = OpenRouterClient()
    prompt = "What is metformin?"

    # Combine all text and vision model aliases
    all_models = {**client.TEXT_MODELS, **client.VISION_MODELS}

    print("\n" + "=" * 80)
    print("TESTING ALL AVAILABLE MODELS")
    print("=" * 80)
    print(f"Prompt: {prompt}\n")

    results = []

    for alias, full_model_id in sorted(all_models.items()):
        print(f"Testing: {alias} ({full_model_id})")
        print("-" * 80)

        try:
            input_data = ModelInput(
                prompt=prompt,
                model=alias
            )
            response = client.generate_text(input_data)

            # Show first 150 characters of response
            truncated_response = response[:150].replace("\n", " ")
            if len(response) > 150:
                truncated_response += "..."

            print(f"✓ SUCCESS")
            print(f"  Response: {truncated_response}")
            print(f"  Total length: {len(response)} characters\n")
            results.append((alias, True, None))

        except Exception as e:
            error_msg = str(e)[:100]
            print(f"✗ FAILED")
            print(f"  Error: {error_msg}\n")
            results.append((alias, False, error_msg))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    print(f"\nTotal Models: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}\n")

    # Detailed results
    print("Model Results:")
    print("-" * 80)
    for alias, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        if success:
            print(f"{status:8} {alias:20} {client.TEXT_MODELS.get(alias) or client.VISION_MODELS.get(alias)}")
        else:
            print(f"{status:8} {alias:20} Error: {error[:40]}")

    print("\n" + "=" * 80)
    print(f"Overall: {passed}/{total} models working\n")

    return passed, total

if __name__ == "__main__":
    try:
        passed, total = test_all_models()
        exit(0 if passed == total else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        exit(1)

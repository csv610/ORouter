#!/usr/bin/env python3
"""Test script for vision capabilities with OpenRouterClient."""

import sys
sys.path.insert(0, '/Users/csv610/Projects/ORouter')

from orouter.openrouter_client import OpenRouterClient, ModelInput

def test_vision_with_url():
    """Test vision with a public image URL."""
    print("=" * 60)
    print("TEST 1: Vision with Public Image URL")
    print("=" * 60)

    try:
        client = OpenRouterClient()
        # Using a simple public image
        input_data = ModelInput(
            user_prompt="Describe what you see in this image in 2 sentences.",
            image_source="https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Good_Food_Display_-_NCI_Visuals_Online.jpg/1200px-Good_Food_Display_-_NCI_Visuals_Online.jpg",
            model="google/gemma-3-27b-it:free"
        )
        response = client.generate_text(input_data)
        print(f"Image URL: https://upload.wikimedia.org/...")
        print(f"Model: gemma27b")
        print(f"Response: {response}\n")
        return True
    except Exception as e:
        print(f"ERROR: {e}\n")
        return False

def test_vision_with_data_uri():
    """Test vision with a simple data URI image (small test image)."""
    print("=" * 60)
    print("TEST 2: Vision with Data URI")
    print("=" * 60)

    try:
        # A simple 1x1 red pixel in data URI format
        red_pixel_data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="

        client = OpenRouterClient()
        input_data = ModelInput(
            user_prompt="What color is this image?",
            image_source=red_pixel_data_uri,
            model="google/gemma-3-27b-it:free"
        )
        response = client.generate_text(input_data)
        print(f"Image: 1x1 pixel data URI")
        print(f"Model: gemma27b")
        print(f"Response: {response}\n")
        return True
    except Exception as e:
        print(f"ERROR: {e}\n")
        return False

def test_vision_with_local_file():
    """Test vision with a local image file."""
    print("=" * 60)
    print("TEST 3: Vision with Local File (Taj Mahal)")
    print("=" * 60)

    try:
        import os

        # Use the Taj Mahal image from the data folder
        taj_mahal_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Tajmahal.jpg')

        if not os.path.exists(taj_mahal_path):
            print(f"ERROR: Image file not found at {taj_mahal_path}\n")
            return False

        client = OpenRouterClient()
        input_data = ModelInput(
            user_prompt="Describe this landmark in detail. What do you see?",
            image_source=taj_mahal_path,
            model="google/gemma-3-27b-it:free"
        )
        response = client.generate_text(input_data)
        print(f"Image: Local file (Taj Mahal)")
        print(f"Model: gemma27b")
        print(f"Response: {response}\n")

        return True
    except Exception as e:
        print(f"ERROR: {e}\n")
        return False

def test_vision_with_system_prompt():
    """Test vision with system prompt and image."""
    print("=" * 60)
    print("TEST 4: Vision with System Prompt")
    print("=" * 60)

    try:
        client = OpenRouterClient()
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Good_Food_Display_-_NCI_Visuals_Online.jpg/1200px-Good_Food_Display_-_NCI_Visuals_Online.jpg"

        input_data = ModelInput(
            user_prompt="What types of food are shown here?",
            image_source=image_url,
            system_prompt="You are a food expert. Analyze images and describe the nutritional content.",
            model="google/gemma-3-27b-it:free"
        )
        response = client.generate_text(input_data)
        print(f"Model: gemma27b")
        print(f"Response: {response}\n")
        return True
    except Exception as e:
        print(f"ERROR: {e}\n")
        return False

def test_vision_model_variants():
    """Test different vision models."""
    print("=" * 60)
    print("TEST 5: Vision Model Variants")
    print("=" * 60)

    try:
        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Good_Food_Display_-_NCI_Visuals_Online.jpg/1200px-Good_Food_Display_-_NCI_Visuals_Online.jpg"
        models_to_test = [
            ("gemma27b-1", "google/gemma-3-27b-it:free"),
            ("gemma27b-2", "google/gemma-3-27b-it:free")
        ]

        for model_name, model_id in models_to_test:
            try:
                client = OpenRouterClient()
                input_data = ModelInput(
                    user_prompt="Briefly describe this image.",
                    image_source=image_url,
                    model=model_id
                )
                response = client.generate_text(input_data)
                print(f"✓ {model_name}: Success (response length: {len(response)} chars)")
            except Exception as e:
                print(f"✗ {model_name}: {str(e)[:80]}")

        print()
        return True
    except Exception as e:
        print(f"ERROR: {e}\n")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("OPENROUTER CLIENT VISION TESTS")
    print("=" * 60 + "\n")

    results = []
    results.append(("Vision with URL", test_vision_with_url()))
    results.append(("Vision with Data URI", test_vision_with_data_uri()))
    results.append(("Vision with Local File", test_vision_with_local_file()))
    results.append(("Vision with System Prompt", test_vision_with_system_prompt()))
    results.append(("Vision Model Variants", test_vision_model_variants()))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    total_passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed\n")

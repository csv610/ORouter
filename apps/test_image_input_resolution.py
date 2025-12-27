#!/usr/bin/env python3
"""Test the image input resolution functionality."""

import sys
from pathlib import Path

sys.path.insert(0, '.')
from nejm_image_challenges_cli import resolve_image_sources

def test_single_file():
    """Test single file resolution."""
    print("Test 1: Single file (tajmahal.jpg)")
    result = resolve_image_sources("/Users/csv610/Projects/ORouter/data/tajmahal.jpg")
    print(f"  Input: /Users/csv610/Projects/ORouter/data/tajmahal.jpg")
    print(f"  Result: {result}")
    assert len(result) == 1
    print("  ✓ PASSED\n")


def test_directory():
    """Test directory resolution."""
    print("Test 2: Directory with images")
    result = resolve_image_sources("/tmp/test_images")
    print(f"  Input: /tmp/test_images")
    print(f"  Found {len(result)} images:")
    for img in result:
        print(f"    - {Path(img).name}")
    assert len(result) == 3
    print("  ✓ PASSED\n")


def test_range():
    """Test range resolution."""
    print("Test 3: Image range (scan_001.jpg:scan_003.jpg)")
    result = resolve_image_sources("/tmp/test_images/scan_001.jpg:/tmp/test_images/scan_003.jpg")
    print(f"  Input: /tmp/test_images/scan_001.jpg:/tmp/test_images/scan_003.jpg")
    print(f"  Found {len(result)} images:")
    for img in result:
        print(f"    - {Path(img).name}")
    assert len(result) == 3
    print("  ✓ PASSED\n")


def test_range_partial():
    """Test range resolution with partial range."""
    print("Test 4: Partial range (scan_001.jpg:scan_002.jpg)")
    result = resolve_image_sources("/tmp/test_images/scan_001.jpg:/tmp/test_images/scan_002.jpg")
    print(f"  Input: /tmp/test_images/scan_001.jpg:/tmp/test_images/scan_002.jpg")
    print(f"  Found {len(result)} images:")
    for img in result:
        print(f"    - {Path(img).name}")
    assert len(result) == 2
    print("  ✓ PASSED\n")


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING IMAGE INPUT RESOLUTION")
    print("=" * 70 + "\n")

    try:
        test_single_file()
        test_directory()
        test_range()
        test_range_partial()

        print("=" * 70)
        print("ALL TESTS PASSED!")
        print("=" * 70)

    except AssertionError as e:
        print(f"  ✗ FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"  ✗ ERROR: {e}\n")
        sys.exit(1)

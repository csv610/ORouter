#!/usr/bin/env python3
"""Test script for structured output with Pydantic models."""

from typing import List, Optional
from pydantic import BaseModel
from openrouter_client import OpenRouterClient

# Define Pydantic models for structured output
class Person(BaseModel):
    """A person with basic information."""
    name: str
    age: int
    occupation: str
    email: Optional[str] = None

class Recipe(BaseModel):
    """A recipe with ingredients and instructions."""
    name: str
    servings: int
    ingredients: List[str]
    instructions: List[str]
    prep_time_minutes: int
    cook_time_minutes: int

class MovieReview(BaseModel):
    """A movie review with structured feedback."""
    title: str
    rating: float  # 1.0 to 10.0
    genres: List[str]
    director: str
    summary: str
    strengths: List[str]
    weaknesses: List[str]
    would_recommend: bool

class ContactInfo(BaseModel):
    """Contact information for a person."""
    full_name: str
    email: str
    phone: str
    address: str
    city: str
    state: str
    zip_code: str

class SentimentAnalysis(BaseModel):
    """Analysis of sentiment in text."""
    text: str
    sentiment: str  # positive, negative, neutral
    confidence: float  # 0.0 to 1.0
    key_phrases: List[str]

def test_simple_structured():
    """Test simple structured output with Person model."""
    print("=" * 60)
    print("TEST 1: Simple Structured Output (Person)")
    print("=" * 60)

    try:
        client = OpenRouterClient()
        prompt = "Create a person named Alice who is a 28-year-old software engineer. Include her email as alice.smith@company.com"

        result = client.generate_structured(
            user_prompt=prompt,
            response_model=Person,
            model="sonnet"
        )

        print(f"Prompt: {prompt}")
        print(f"\nStructured Output:")
        print(f"  Name: {result.name}")
        print(f"  Age: {result.age}")
        print(f"  Occupation: {result.occupation}")
        print(f"  Email: {result.email}\n")

        # Validate structure
        assert isinstance(result, Person)
        assert result.name == "Alice"
        assert result.age == 28
        assert result.occupation.lower() in ["software engineer", "software engineer"]

        return True
    except Exception as e:
        print(f"ERROR: {e}\n")
        return False

def test_complex_structured():
    """Test complex structured output with Recipe model."""
    print("=" * 60)
    print("TEST 2: Complex Structured Output (Recipe)")
    print("=" * 60)

    try:
        client = OpenRouterClient()
        prompt = "Provide a recipe for chocolate chip cookies"

        result = client.generate_structured(
            user_prompt=prompt,
            response_model=Recipe,
            model="sonnet"
        )

        print(f"Prompt: {prompt}")
        print(f"\nStructured Output:")
        print(f"  Name: {result.name}")
        print(f"  Servings: {result.servings}")
        print(f"  Prep Time: {result.prep_time_minutes} minutes")
        print(f"  Cook Time: {result.cook_time_minutes} minutes")
        print(f"  Ingredients ({len(result.ingredients)}):")
        for ing in result.ingredients[:3]:
            print(f"    - {ing}")
        if len(result.ingredients) > 3:
            print(f"    ... and {len(result.ingredients) - 3} more")
        print(f"  Instructions ({len(result.instructions)} steps)\n")

        # Validate structure
        assert isinstance(result, Recipe)
        assert len(result.ingredients) > 0
        assert len(result.instructions) > 0
        assert result.prep_time_minutes >= 0
        assert result.cook_time_minutes >= 0

        return True
    except Exception as e:
        print(f"ERROR: {e}\n")
        return False

def test_nested_structured():
    """Test nested structured output with MovieReview model."""
    print("=" * 60)
    print("TEST 3: Nested Structured Output (Movie Review)")
    print("=" * 60)

    try:
        client = OpenRouterClient()
        prompt = "Review the movie 'The Matrix' from 1999. Rate it and provide analysis."

        result = client.generate_structured(
            user_prompt=prompt,
            response_model=MovieReview,
            model="sonnet"
        )

        print(f"Prompt: {prompt}")
        print(f"\nStructured Output:")
        print(f"  Title: {result.title}")
        print(f"  Rating: {result.rating}/10.0")
        print(f"  Director: {result.director}")
        print(f"  Genres: {', '.join(result.genres)}")
        print(f"  Summary: {result.summary[:100]}...")
        print(f"  Strengths: {', '.join(result.strengths[:2])}")
        print(f"  Weaknesses: {', '.join(result.weaknesses[:2])}")
        print(f"  Recommend: {'Yes' if result.would_recommend else 'No'}\n")

        # Validate structure
        assert isinstance(result, MovieReview)
        assert 0 <= result.rating <= 10
        assert len(result.genres) > 0
        assert len(result.strengths) > 0
        assert len(result.weaknesses) > 0
        assert isinstance(result.would_recommend, bool)

        return True
    except Exception as e:
        print(f"ERROR: {e}\n")
        return False

def test_list_structured():
    """Test generating a list of structured outputs with wrapper model."""
    print("=" * 60)
    print("TEST 4: List of Structured Output (People)")
    print("=" * 60)

    try:
        client = OpenRouterClient()
        prompt = "Create 3 different people: a doctor, a teacher, and a chef. Each should have realistic names, ages, and occupations."

        class PeopleList(BaseModel):
            items: List[Person]

        result = client.generate_structured(
            user_prompt=prompt,
            response_model=PeopleList,
            model="sonnet"
        )
        results = result.items

        print(f"Prompt: {prompt}")
        print(f"\nStructured Output ({len(results)} items):")
        for i, person in enumerate(results, 1):
            print(f"  {i}. {person.name} ({person.age}) - {person.occupation}")

        print()

        # Validate structure
        assert isinstance(results, list)
        assert len(results) >= 1
        assert all(isinstance(p, Person) for p in results)

        return True
    except Exception as e:
        print(f"ERROR: {e}\n")
        return False

def test_sentiment_analysis():
    """Test sentiment analysis structured output."""
    print("=" * 60)
    print("TEST 5: Sentiment Analysis Structured Output")
    print("=" * 60)

    try:
        client = OpenRouterClient()
        text = "I absolutely love this product! It's amazing, high quality, and works perfectly. Highly recommend!"
        prompt = f"Analyze the sentiment of this text: '{text}'"

        result = client.generate_structured(
            user_prompt=prompt,
            response_model=SentimentAnalysis,
            model="haiku"
        )

        print(f"Text: {text}")
        print(f"\nStructured Output:")
        print(f"  Sentiment: {result.sentiment.upper()}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Key Phrases: {', '.join(result.key_phrases)}\n")

        # Validate structure
        assert isinstance(result, SentimentAnalysis)
        assert result.sentiment.lower() in ["positive", "negative", "neutral"]
        assert 0 <= result.confidence <= 1
        assert len(result.key_phrases) > 0

        return True
    except Exception as e:
        print(f"ERROR: {e}\n")
        return False

def test_retry_on_validation():
    """Test that retries work when validation fails."""
    print("=" * 60)
    print("TEST 6: Retry on Validation Failure")
    print("=" * 60)

    try:
        client = OpenRouterClient()
        prompt = "Create a person. Name should be 'Bob', age 30, and occupation 'Consultant'."

        result = client.generate_structured(
            user_prompt=prompt,
            response_model=Person,
            model="sonnet",
            max_retries=3
        )

        print(f"Prompt: {prompt}")
        print(f"\nStructured Output (with retries allowed):")
        print(f"  Name: {result.name}")
        print(f"  Age: {result.age}")
        print(f"  Occupation: {result.occupation}\n")

        # Validate structure
        assert isinstance(result, Person)

        return True
    except Exception as e:
        print(f"ERROR: {e}\n")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("OPENROUTER CLIENT STRUCTURED OUTPUT TESTS")
    print("=" * 60 + "\n")

    results = []
    results.append(("Simple Structured (Person)", test_simple_structured()))
    results.append(("Complex Structured (Recipe)", test_complex_structured()))
    results.append(("Nested Structured (Movie Review)", test_nested_structured()))
    results.append(("List Structured (People)", test_list_structured()))
    results.append(("Sentiment Analysis", test_sentiment_analysis()))
    results.append(("Retry on Validation", test_retry_on_validation()))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")

    total_passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed\n")

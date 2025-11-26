"""Pytest configuration and shared fixtures."""

import os
import pytest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel


@pytest.fixture
def mock_api_key(monkeypatch):
    """Set a mock API key for testing."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-api-key-12345")
    return "test-api-key-12345"


@pytest.fixture
def mock_openai_client(mock_api_key):
    """Create a mocked OpenAI client."""
    with patch("orouter.client.OpenAI") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


@pytest.fixture
def sample_pydantic_model():
    """Create a simple Pydantic model for testing."""
    class SampleModel(BaseModel):
        name: str
        age: int
        email: str

    return SampleModel


@pytest.fixture
def mock_completion_response():
    """Create a mock completion response."""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    return mock_response

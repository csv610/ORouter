"""Unit tests for OpenRouterClient."""

import json
import pytest
from unittest.mock import MagicMock, patch, call
from pydantic import BaseModel, ValidationError

from orouter.client import OpenRouterClient
from orouter.config import ModelConfig


class TestOpenRouterClientInitialization:
    """Test OpenRouterClient initialization."""

    def test_initialization_without_api_key(self):
        """Test that initialization without API key raises ValueError."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key not found"):
                OpenRouterClient()

    def test_initialization_with_api_key(self, mock_api_key, mock_openai_client):
        """Test successful initialization with API key."""
        client = OpenRouterClient()
        assert client.client is not None
        assert client.default_config is not None

    def test_initialization_with_custom_config(self, mock_api_key, mock_openai_client):
        """Test initialization with custom ModelConfig."""
        custom_config = ModelConfig(temperature=1.5, model="llama")
        client = OpenRouterClient(config=custom_config)
        assert client.default_config.temperature == 1.5

    def test_initialization_creates_openai_client_with_correct_url(self, mock_api_key, mock_openai_client):
        """Test that OpenAI client is initialized with correct URL."""
        OpenRouterClient()
        mock_openai_client.assert_called_once()
        call_kwargs = mock_openai_client.call_args[1]
        assert call_kwargs["base_url"] == "https://openrouter.ai/api/v1"
        assert call_kwargs["api_key"] == "test-api-key-12345"

    def test_initialization_with_none_model_selects_random(self, mock_api_key, mock_openai_client):
        """Test that a random model is selected when model is None."""
        with patch("orouter.client.random.choice") as mock_choice:
            mock_choice.return_value = "meta-llama/llama-3.3-8b-instruct:free"
            config = ModelConfig(model=None)
            client = OpenRouterClient(config=config)
            mock_choice.assert_called()

    def test_initialization_with_alias_resolves_model(self, mock_api_key, mock_openai_client):
        """Test that model aliases are resolved during initialization."""
        config = ModelConfig(model="llama")
        client = OpenRouterClient(config=config)
        assert client.default_config.model == "meta-llama/llama-3.3-8b-instruct:free"

    def test_initialization_with_full_model_name(self, mock_api_key, mock_openai_client):
        """Test initialization with full model name."""
        full_model = "deepseek/deepseek-chat-v3.1:free"
        config = ModelConfig(model=full_model)
        client = OpenRouterClient(config=config)
        assert client.default_config.model == full_model


class TestResolveModel:
    """Test OpenRouterClient.resolve_model() method."""

    def test_resolve_model_with_valid_alias(self):
        """Test resolving a valid model alias."""
        result = OpenRouterClient.resolve_model("llama")
        assert result == "meta-llama/llama-3.3-8b-instruct:free"

    def test_resolve_model_case_insensitive(self):
        """Test that alias resolution is case insensitive."""
        result = OpenRouterClient.resolve_model("LLAMA")
        assert result == "meta-llama/llama-3.3-8b-instruct:free"

        result = OpenRouterClient.resolve_model("LlAmA")
        assert result == "meta-llama/llama-3.3-8b-instruct:free"

    def test_resolve_model_all_aliases(self):
        """Test that all known aliases resolve correctly."""
        aliases = {
            "deepseek": "deepseek/deepseek-chat-v3.1:free",
            "mistral": "mistralai/mistral-small-3.2-24b-instruct:free",
            "kimi": "moonshotai/kimi-dev-72b:free",
            "llama": "meta-llama/llama-3.3-8b-instruct:free",
            "nemotron": "nvidia/nemotron-nano-9b-v2:free",
            "gpt": "openai/gpt-oss-20b:free",
            "qwen14": "qwen/qwen3-14b:free",
            "qwen30": "qwen/qwen3-30b-a3b:free",
            "qwen235": "qwen/qwen3-235b-a22b:free",
            "hunyuan": "tencent/hunyuan-a13b-instruct:free",
            "grok": "x-ai/grok-4-fast:free",
            "glm": "z-ai/glm-4.5-air:free",
        }

        for alias, expected_model in aliases.items():
            result = OpenRouterClient.resolve_model(alias)
            assert result == expected_model

    def test_resolve_model_with_full_name(self):
        """Test that full model names are returned unchanged."""
        full_model = "deepseek/deepseek-chat-v3.1:free"
        result = OpenRouterClient.resolve_model(full_model)
        assert result == full_model

    def test_resolve_model_with_unknown_model(self):
        """Test that unknown models are returned as-is."""
        unknown = "unknown/model:free"
        result = OpenRouterClient.resolve_model(unknown)
        assert result == unknown


class TestGenerateText:
    """Test OpenRouterClient.generate_text() method."""

    def test_generate_text_success(self, mock_api_key, mock_openai_client, mock_completion_response):
        """Test successful text generation."""
        mock_openai_client.chat.completions.create.return_value = mock_completion_response
        client = OpenRouterClient()
        result = client.generate_text("Hello")

        assert result == "Test response"
        mock_openai_client.chat.completions.create.assert_called_once()

    def test_generate_text_includes_user_message(self, mock_api_key, mock_openai_client, mock_completion_response):
        """Test that user message is included in API call."""
        mock_openai_client.chat.completions.create.return_value = mock_completion_response
        client = OpenRouterClient()
        client.generate_text("Test prompt")

        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert any(msg["content"] == "Test prompt" and msg["role"] == "user" for msg in messages)

    def test_generate_text_includes_system_prompt(self, mock_api_key, mock_openai_client, mock_completion_response):
        """Test that system prompt is included when provided."""
        mock_openai_client.chat.completions.create.return_value = mock_completion_response
        config = ModelConfig(system_prompt="You are helpful")
        client = OpenRouterClient(config=config)
        client.generate_text("Test prompt")

        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        assert any(msg["content"] == "You are helpful" and msg["role"] == "system" for msg in messages)

    def test_generate_text_without_system_prompt(self, mock_api_key, mock_openai_client, mock_completion_response):
        """Test that system prompt is not included when not provided."""
        mock_openai_client.chat.completions.create.return_value = mock_completion_response
        config = ModelConfig(system_prompt=None)
        client = OpenRouterClient(config=config)
        client.generate_text("Test prompt")

        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        system_msgs = [msg for msg in messages if msg["role"] == "system"]
        assert len(system_msgs) == 0

    def test_generate_text_passes_config_params(self, mock_api_key, mock_openai_client, mock_completion_response):
        """Test that config parameters are passed to API."""
        mock_openai_client.chat.completions.create.return_value = mock_completion_response
        config = ModelConfig(temperature=1.5, top_p=0.8, max_tokens=256)
        client = OpenRouterClient(config=config)
        client.generate_text("Test")

        call_args = mock_openai_client.chat.completions.create.call_args
        call_kwargs = call_args[1]
        assert call_kwargs["temperature"] == 1.5
        assert call_kwargs["top_p"] == 0.8
        assert call_kwargs["max_tokens"] == 256

    def test_generate_text_uses_default_model(self, mock_api_key, mock_openai_client, mock_completion_response):
        """Test that default model is used."""
        mock_openai_client.chat.completions.create.return_value = mock_completion_response
        config = ModelConfig(model="llama")
        client = OpenRouterClient(config=config)
        client.generate_text("Test")

        call_args = mock_openai_client.chat.completions.create.call_args
        assert call_args[1]["model"] == "meta-llama/llama-3.3-8b-instruct:free"

    def test_generate_text_api_error_raises_runtime_error(self, mock_api_key, mock_openai_client):
        """Test that API errors raise RuntimeError."""
        mock_openai_client.chat.completions.create.side_effect = Exception("API Error")
        client = OpenRouterClient()

        with pytest.raises(RuntimeError, match="API request failed"):
            client.generate_text("Test")

    def test_generate_text_returns_string(self, mock_api_key, mock_openai_client, mock_completion_response):
        """Test that generate_text returns a string."""
        mock_openai_client.chat.completions.create.return_value = mock_completion_response
        client = OpenRouterClient()
        result = client.generate_text("Test")
        assert isinstance(result, str)


class TestExtractJson:
    """Test OpenRouterClient._extract_json() static method."""

    def test_extract_json_plain_json(self):
        """Test extracting plain JSON without markdown."""
        json_str = '{"key": "value", "number": 42}'
        result = OpenRouterClient._extract_json(json_str)
        assert result == json_str

    def test_extract_json_with_json_markdown_block(self):
        """Test extracting JSON from ```json markdown block."""
        text = '```json\n{"key": "value"}\n```'
        result = OpenRouterClient._extract_json(text)
        assert result == '{"key": "value"}'

    def test_extract_json_with_generic_markdown_block(self):
        """Test extracting JSON from ``` markdown block."""
        text = '```\n{"key": "value"}\n```'
        result = OpenRouterClient._extract_json(text)
        assert result == '{"key": "value"}'

    def test_extract_json_with_whitespace(self):
        """Test extracting JSON with extra whitespace."""
        text = '  ```json\n  {"key": "value"}  \n```  '
        result = OpenRouterClient._extract_json(text)
        assert "key" in result
        assert "value" in result

    def test_extract_json_nested_objects(self):
        """Test extracting nested JSON objects."""
        text = '```json\n{"outer": {"inner": "value"}}\n```'
        result = OpenRouterClient._extract_json(text)
        parsed = json.loads(result)
        assert parsed["outer"]["inner"] == "value"

    def test_extract_json_arrays(self):
        """Test extracting JSON with arrays."""
        text = '```json\n{"items": [1, 2, 3]}\n```'
        result = OpenRouterClient._extract_json(text)
        parsed = json.loads(result)
        assert parsed["items"] == [1, 2, 3]

    def test_extract_json_without_closing_marker(self):
        """Test extracting JSON when closing ``` is missing."""
        text = '```json\n{"key": "value"}'
        result = OpenRouterClient._extract_json(text)
        assert "key" in result

    def test_extract_json_with_markdown_code_fence_variations(self):
        """Test various markdown code fence formats."""
        # With language specified
        text1 = '```json\n{"test": true}\n```'
        result1 = OpenRouterClient._extract_json(text1)
        assert "test" in result1

        # Without language
        text2 = '```\n{"test": true}\n```'
        result2 = OpenRouterClient._extract_json(text2)
        assert "test" in result2

    def test_extract_json_strips_leading_trailing_whitespace(self):
        """Test that leading/trailing whitespace is stripped."""
        text = '   \n   {"key": "value"}   \n   '
        result = OpenRouterClient._extract_json(text)
        assert result == '{"key": "value"}'

    def test_extract_json_preserves_internal_formatting(self):
        """Test that internal JSON formatting is preserved."""
        json_str = '{"key": "value with spaces", "nested": {"inner": "data"}}'
        result = OpenRouterClient._extract_json(json_str)
        parsed = json.loads(result)
        assert parsed["key"] == "value with spaces"

    def test_extract_json_multiline_json(self):
        """Test extracting multiline JSON from markdown."""
        text = '''```json
{
  "key1": "value1",
  "key2": "value2",
  "nested": {
    "inner": "data"
  }
}
```'''
        result = OpenRouterClient._extract_json(text)
        parsed = json.loads(result)
        assert parsed["key1"] == "value1"
        assert parsed["nested"]["inner"] == "data"


class TestGenerateStructured:
    """Test OpenRouterClient.generate_structured() method."""

    def test_generate_structured_success(self, mock_api_key, mock_openai_client, sample_pydantic_model):
        """Test successful structured generation."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        valid_json = '{"name": "John", "age": 30, "email": "john@example.com"}'
        mock_response.choices[0].message.content = valid_json
        mock_openai_client.chat.completions.create.return_value = mock_response

        client = OpenRouterClient()
        result = client.generate_structured("Tell me about John", sample_pydantic_model)

        assert isinstance(result, sample_pydantic_model)
        assert result.name == "John"
        assert result.age == 30
        assert result.email == "john@example.com"

    def test_generate_structured_with_markdown_json(self, mock_api_key, mock_openai_client, sample_pydantic_model):
        """Test structured generation with markdown-wrapped JSON."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        json_with_markdown = '```json\n{"name": "Jane", "age": 25, "email": "jane@example.com"}\n```'
        mock_response.choices[0].message.content = json_with_markdown
        mock_openai_client.chat.completions.create.return_value = mock_response

        client = OpenRouterClient()
        result = client.generate_structured("Tell me about Jane", sample_pydantic_model)

        assert result.name == "Jane"
        assert result.age == 25

    def test_generate_structured_includes_schema_in_prompt(self, mock_api_key, mock_openai_client, sample_pydantic_model, mock_completion_response):
        """Test that schema is included in system prompt."""
        mock_openai_client.chat.completions.create.return_value = mock_completion_response
        client = OpenRouterClient()
        client.generate_text("Test")  # Initialize

        # Reset mock and test structured
        mock_openai_client.chat.completions.create.return_value = mock_completion_response
        try:
            client.generate_structured("Test", sample_pydantic_model)
        except (ValidationError, json.JSONDecodeError):
            pass  # Expected since mock returns "Test response"

        call_args = mock_openai_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        system_msg = next(msg for msg in messages if msg["role"] == "system")
        assert "schema" in system_msg["content"].lower()

    def test_generate_structured_validation_error_raises(self, mock_api_key, mock_openai_client, sample_pydantic_model):
        """Test that validation errors are raised."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"invalid": "json"}'  # Missing required fields
        mock_openai_client.chat.completions.create.return_value = mock_response

        client = OpenRouterClient()
        with pytest.raises(ValidationError):
            client.generate_structured("Test", sample_pydantic_model)

import pytest
from unittest.mock import Mock, MagicMock, patch
from pydantic import BaseModel

from openrouter_client import OpenRouterClient, ModelInput, ModelConfig


class SampleOutput(BaseModel):
    """Sample Pydantic model for testing structured output."""
    name: str
    age: int


class TestGenerateContent:
    """Test suite for generate_content function."""

    @pytest.fixture
    def client(self):
        """Create a mock OpenRouterClient for testing."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            return OpenRouterClient()

    def test_generate_content_text_mode(self, client):
        """Test generate_content delegates to generate_text when no response_model."""
        config = ModelConfig(model="haiku", temperature=0.7)
        input_data = ModelInput(user_prompt="Hello world")

        # Mock generate_text
        with patch.object(client, "generate_text", return_value="Hello!") as mock_text:
            result = client.generate_content(input_data, config)

            assert result == "Hello!"
            mock_text.assert_called_once_with(input_data)
            assert input_data.model == "haiku"

    def test_generate_content_structured_mode(self, client):
        """Test generate_content delegates to generate_structured with response_model."""
        config = ModelConfig(model="haiku", temperature=0.5, max_tokens=1000)
        input_data = ModelInput(
            user_prompt="Extract person info",
            response_model=SampleOutput
        )

        # Mock generate_structured
        mock_result = SampleOutput(name="John", age=30)
        with patch.object(client, "generate_structured", return_value=mock_result) as mock_struct:
            result = client.generate_content(input_data, config)

            assert result == mock_result
            mock_struct.assert_called_once_with(input_data)
            assert input_data.model == "haiku"

    def test_generate_content_missing_user_prompt(self, client):
        """Test generate_content raises ValueError when user_prompt is missing."""
        config = ModelConfig(model="haiku")
        input_data = ModelInput()  # No user_prompt

        with pytest.raises(ValueError, match="user_prompt must be provided"):
            client.generate_content(input_data, config)

    def test_generate_content_missing_model(self, client):
        """Test generate_content raises ValueError when model is missing in config."""
        config = ModelConfig()  # No model specified
        input_data = ModelInput(user_prompt="Hello")

        with pytest.raises(ValueError, match="model must be provided"):
            client.generate_content(input_data, config)

    def test_generate_content_sets_config_on_client(self, client):
        """Test generate_content updates client's default_config."""
        config = ModelConfig(
            model="haiku",
            temperature=0.3,
            max_tokens=500,
            top_p=0.9
        )
        input_data = ModelInput(user_prompt="Test")

        with patch.object(client, "generate_text", return_value="response"):
            client.generate_content(input_data, config)

            assert client.default_config == config
            assert client.default_config.temperature == 0.3
            assert client.default_config.max_tokens == 500
            assert client.default_config.top_p == 0.9

    def test_generate_content_model_resolution(self, client):
        """Test generate_content correctly sets model from config to input_data."""
        config = ModelConfig(model="mistral")
        input_data = ModelInput(user_prompt="Test prompt")

        with patch.object(client, "generate_text"):
            client.generate_content(input_data, config)

            assert input_data.model == "mistral"

    def test_generate_content_text_with_image(self, client):
        """Test generate_content in text mode with image source."""
        config = ModelConfig(model="haiku")
        input_data = ModelInput(
            user_prompt="Describe this image",
            image_source="https://example.com/image.jpg"
        )

        with patch.object(client, "generate_text", return_value="Image description") as mock_text:
            result = client.generate_content(input_data, config)

            assert result == "Image description"
            assert input_data.image_source == "https://example.com/image.jpg"
            mock_text.assert_called_once()

    def test_generate_content_structured_with_extra_body(self, client):
        """Test generate_content preserves extra_body in input_data."""
        config = ModelConfig(model="haiku")
        extra_body = {"custom_param": "value"}
        input_data = ModelInput(
            user_prompt="Extract data",
            response_model=SampleOutput,
            extra_body=extra_body
        )

        with patch.object(client, "generate_structured"):
            client.generate_content(input_data, config)

            assert input_data.extra_body == extra_body


class TestModelConfigWithModel:
    """Test ModelConfig's model field."""

    def test_model_config_with_model(self):
        """Test ModelConfig can be initialized with model."""
        config = ModelConfig(
            model="haiku",
            temperature=0.7,
            max_tokens=2000
        )

        assert config.model == "haiku"
        assert config.temperature == 0.7
        assert config.max_tokens == 2000

    def test_model_config_default_none(self):
        """Test ModelConfig model defaults to None."""
        config = ModelConfig()

        assert config.model is None

    def test_model_config_to_api_params_excludes_model(self):
        """Test to_api_params doesn't include model field."""
        config = ModelConfig(model="haiku", temperature=0.5)
        params = config.to_api_params()

        assert "model" not in params
        assert params["temperature"] == 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

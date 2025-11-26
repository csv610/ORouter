"""Unit tests for ModelConfig."""

import pytest
from orouter.config import ModelConfig


class TestModelConfigInitialization:
    """Test ModelConfig initialization with valid parameters."""

    def test_default_initialization(self):
        """Test creating ModelConfig with default values."""
        config = ModelConfig()
        assert config.temperature == 0.7
        assert config.top_p == 1.0
        assert config.frequency_penalty == 0.0
        assert config.presence_penalty == 0.0
        assert config.max_tokens is None
        assert config.system_prompt is None
        assert config.model is None
        assert config.extra_body == {}

    def test_initialization_with_custom_values(self):
        """Test creating ModelConfig with custom values."""
        config = ModelConfig(
            model="test-model",
            temperature=1.5,
            max_tokens=512,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=-1.0,
            system_prompt="You are a helpful assistant"
        )
        assert config.model == "test-model"
        assert config.temperature == 1.5
        assert config.max_tokens == 512
        assert config.top_p == 0.9
        assert config.frequency_penalty == 0.5
        assert config.presence_penalty == -1.0
        assert config.system_prompt == "You are a helpful assistant"

    def test_initialization_with_extra_body(self):
        """Test creating ModelConfig with extra_body parameter."""
        extra_body = {"custom_param": "value", "number": 42}
        config = ModelConfig(extra_body=extra_body)
        assert config.extra_body == extra_body

    def test_temperature_boundary_values(self):
        """Test temperature at boundary values."""
        # Min boundary
        config = ModelConfig(temperature=0)
        assert config.temperature == 0

        # Max boundary
        config = ModelConfig(temperature=2)
        assert config.temperature == 2

        # Mid-range
        config = ModelConfig(temperature=1.0)
        assert config.temperature == 1.0


class TestModelConfigTemperatureValidation:
    """Test ModelConfig temperature validation."""

    def test_temperature_below_zero(self):
        """Test that temperature below 0 raises ValueError."""
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            ModelConfig(temperature=-0.1)

    def test_temperature_above_two(self):
        """Test that temperature above 2 raises ValueError."""
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            ModelConfig(temperature=2.1)

    def test_temperature_negative_large(self):
        """Test that large negative temperature raises ValueError."""
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            ModelConfig(temperature=-10)

    def test_temperature_large_positive(self):
        """Test that large positive temperature raises ValueError."""
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            ModelConfig(temperature=10)


class TestModelConfigTopPValidation:
    """Test ModelConfig top_p validation."""

    def test_top_p_below_zero(self):
        """Test that top_p below 0 raises ValueError."""
        with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
            ModelConfig(top_p=-0.1)

    def test_top_p_above_one(self):
        """Test that top_p above 1 raises ValueError."""
        with pytest.raises(ValueError, match="top_p must be between 0 and 1"):
            ModelConfig(top_p=1.1)

    def test_top_p_boundary_values(self):
        """Test top_p at boundary values."""
        # Min boundary
        config = ModelConfig(top_p=0)
        assert config.top_p == 0

        # Max boundary
        config = ModelConfig(top_p=1)
        assert config.top_p == 1


class TestModelConfigFrequencyPenaltyValidation:
    """Test ModelConfig frequency_penalty validation."""

    def test_frequency_penalty_below_minus_two(self):
        """Test that frequency_penalty below -2 raises ValueError."""
        with pytest.raises(ValueError, match="frequency_penalty must be between -2 and 2"):
            ModelConfig(frequency_penalty=-2.1)

    def test_frequency_penalty_above_two(self):
        """Test that frequency_penalty above 2 raises ValueError."""
        with pytest.raises(ValueError, match="frequency_penalty must be between -2 and 2"):
            ModelConfig(frequency_penalty=2.1)

    def test_frequency_penalty_boundary_values(self):
        """Test frequency_penalty at boundary values."""
        # Min boundary
        config = ModelConfig(frequency_penalty=-2)
        assert config.frequency_penalty == -2

        # Max boundary
        config = ModelConfig(frequency_penalty=2)
        assert config.frequency_penalty == 2

        # Mid-range positive
        config = ModelConfig(frequency_penalty=1.5)
        assert config.frequency_penalty == 1.5

        # Mid-range negative
        config = ModelConfig(frequency_penalty=-0.5)
        assert config.frequency_penalty == -0.5


class TestModelConfigPresencePenaltyValidation:
    """Test ModelConfig presence_penalty validation."""

    def test_presence_penalty_below_minus_two(self):
        """Test that presence_penalty below -2 raises ValueError."""
        with pytest.raises(ValueError, match="presence_penalty must be between -2 and 2"):
            ModelConfig(presence_penalty=-2.1)

    def test_presence_penalty_above_two(self):
        """Test that presence_penalty above 2 raises ValueError."""
        with pytest.raises(ValueError, match="presence_penalty must be between -2 and 2"):
            ModelConfig(presence_penalty=2.1)

    def test_presence_penalty_boundary_values(self):
        """Test presence_penalty at boundary values."""
        # Min boundary
        config = ModelConfig(presence_penalty=-2)
        assert config.presence_penalty == -2

        # Max boundary
        config = ModelConfig(presence_penalty=2)
        assert config.presence_penalty == 2


class TestModelConfigMaxTokensValidation:
    """Test ModelConfig max_tokens validation."""

    def test_max_tokens_zero(self):
        """Test that max_tokens of 0 raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            ModelConfig(max_tokens=0)

    def test_max_tokens_negative(self):
        """Test that negative max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            ModelConfig(max_tokens=-100)

    def test_max_tokens_positive_values(self):
        """Test that positive max_tokens values are accepted."""
        config = ModelConfig(max_tokens=1)
        assert config.max_tokens == 1

        config = ModelConfig(max_tokens=4096)
        assert config.max_tokens == 4096

        config = ModelConfig(max_tokens=1000000)
        assert config.max_tokens == 1000000

    def test_max_tokens_none(self):
        """Test that max_tokens can be None."""
        config = ModelConfig(max_tokens=None)
        assert config.max_tokens is None


class TestModelConfigToApiParams:
    """Test ModelConfig.to_api_params() method."""

    def test_to_api_params_default_values(self):
        """Test to_api_params with default values."""
        config = ModelConfig()
        params = config.to_api_params()

        assert params["temperature"] == 0.7
        assert params["top_p"] == 1.0
        assert params["frequency_penalty"] == 0.0
        assert params["presence_penalty"] == 0.0
        assert params["extra_body"] == {}
        assert "max_tokens" not in params  # Should not include None values

    def test_to_api_params_with_max_tokens(self):
        """Test to_api_params includes max_tokens when set."""
        config = ModelConfig(max_tokens=512)
        params = config.to_api_params()

        assert params["max_tokens"] == 512

    def test_to_api_params_with_none_max_tokens(self):
        """Test to_api_params excludes max_tokens when None."""
        config = ModelConfig(max_tokens=None)
        params = config.to_api_params()

        assert "max_tokens" not in params

    def test_to_api_params_custom_values(self):
        """Test to_api_params with custom values."""
        config = ModelConfig(
            temperature=1.5,
            max_tokens=256,
            top_p=0.8,
            frequency_penalty=-0.5,
            presence_penalty=1.0
        )
        params = config.to_api_params()

        assert params["temperature"] == 1.5
        assert params["max_tokens"] == 256
        assert params["top_p"] == 0.8
        assert params["frequency_penalty"] == -0.5
        assert params["presence_penalty"] == 1.0

    def test_to_api_params_with_extra_body(self):
        """Test to_api_params includes extra_body."""
        extra_body = {"custom_key": "custom_value", "another_key": 123}
        config = ModelConfig(extra_body=extra_body)
        params = config.to_api_params()

        assert params["extra_body"] == extra_body

    def test_to_api_params_returns_dict(self):
        """Test that to_api_params returns a dictionary."""
        config = ModelConfig()
        params = config.to_api_params()

        assert isinstance(params, dict)

    def test_to_api_params_includes_all_required_keys(self):
        """Test that to_api_params includes all required parameter keys."""
        config = ModelConfig(max_tokens=100)
        params = config.to_api_params()

        required_keys = {
            "temperature", "top_p", "frequency_penalty",
            "presence_penalty", "extra_body", "max_tokens"
        }
        assert set(params.keys()) == required_keys

    def test_to_api_params_does_not_include_non_api_params(self):
        """Test that to_api_params doesn't include model or system_prompt."""
        config = ModelConfig(
            model="test-model",
            system_prompt="You are helpful"
        )
        params = config.to_api_params()

        # These should not be in API params (handled separately)
        assert "model" not in params
        assert "system_prompt" not in params


class TestModelConfigMultipleValidationErrors:
    """Test combinations of validation errors."""

    def test_multiple_invalid_parameters(self):
        """Test that the first invalid parameter raises an error."""
        # When multiple parameters are invalid, the first checked one raises
        with pytest.raises(ValueError, match="Temperature must be between 0 and 2"):
            ModelConfig(temperature=-1, top_p=2)

    def test_all_parameters_at_extremes(self):
        """Test with all parameters at their extreme valid values."""
        config = ModelConfig(
            temperature=2,
            max_tokens=1,
            top_p=0,
            frequency_penalty=-2,
            presence_penalty=2
        )
        assert config.temperature == 2
        assert config.max_tokens == 1
        assert config.top_p == 0
        assert config.frequency_penalty == -2
        assert config.presence_penalty == 2

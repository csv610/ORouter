"""Configuration classes for OpenRouter client."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    """
    Configuration for model inference parameters.
    """
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    system_prompt: Optional[str] = None
    extra_body: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("Temperature must be between 0 and 2")
        if self.top_p < 0 or self.top_p > 1:
            raise ValueError("top_p must be between 0 and 1")
        if self.frequency_penalty < -2 or self.frequency_penalty > 2:
            raise ValueError("frequency_penalty must be between -2 and 2")
        if self.presence_penalty < -2 or self.presence_penalty > 2:
            raise ValueError("presence_penalty must be between -2 and 2")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")

    def to_api_params(self) -> Dict[str, Any]:
        """
        Convert config to API parameters dictionary.

        Returns:
            Dictionary of API parameters with non-None values.
        """
        params = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "extra_body": self.extra_body,
        }

        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        return params

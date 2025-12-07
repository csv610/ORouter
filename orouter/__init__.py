"""OpenRouter Python Client - Free AI Model Aggregator."""

from .openrouter_client import OpenRouterClient, ModelInput
from .config import ModelConfig

__version__ = "0.1.0"
__all__ = [
    "OpenRouterClient",
    "ModelConfig",
    "ModelInput",
]

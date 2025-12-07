import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .image_utils import ImageUtils

T = TypeVar("T", bound=BaseModel)


@dataclass
class ModelInput:
    """Generic input for text generation or structured output.

    Optional dataclass for users who prefer structured parameter passing.
    All fields are optional and only relevant fields for the operation should be set.
    """
    # Text/Vision generation fields
    user_prompt: Optional[str] = None
    system_prompt: Optional[str] = None

    image_source: Optional[str] = None

    # Structured generation fields
    response_model: Optional[Type[T]] = None

    # Common fields
    model: Optional[str] = None
    extra_body: Optional[Dict] = None


@dataclass
class ModelConfig:
    """Configuration for model inference parameters."""
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
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
        """Convert config to API parameters dictionary."""
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

class OpenRouterClient:
    """Manage OpenRouter API interactions with text and vision models."""

    TEXT_MODELS = {
        "deepseek": "tngtech/deepseek-r1t2-chimera:free",
        "gemma3n" : "google/gemma-3n-e4b-it:free",
        "glm": "z-ai/glm-4.5-air:free",
        "gpt-oss": "openai/gpt-oss-20b:free",
        "grok": "x-ai/grok-4.1-fast:free",
        "llama": "meta-llama/llama-3.3-70b-instruct:free",
        "mistral": "mistralai/mistral-small-3.1-24b-instruct:free",
        "nemotron": "nvidia/nemotron-nano-12b-v2-vl:free",
        "qwen-235b": "qwen/qwen3-235b-a22b:free"
    }

    VISION_MODELS = {
        "gemma27b": "google/gemma-3-27b-it:free",
        "haiku": "anthropic/claude-haiku-4.5",
        "mistral": "mistralai/mistral-small-3.2-24b-instruct:free",
    }

    NONFREE_MODELS = {
        "sonnet": "anthropic/claude-sonnet-4.5",
        "sonar": "perplexity/sonar",
        "sonar-pro": "perplexity/sonar-pro",
        "sonar-research": "perplexity/sonar-deep-research",
        "sonar-search": "perplexity/sonar-pro-search",
        "sonar-reason": "perplexity/sonar-reasoning-pro"
    }

    ALL_MODELS = list(TEXT_MODELS.values()) + list(VISION_MODELS.values())

    def __init__(self, api_key: Optional[str] = None, config: Optional[ModelConfig] = None):
        """
        Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY env variable.
            config: Optional ModelConfig for default inference parameters.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set in OPENROUTER_API_KEY environment variable"
            )
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=self.api_key
        )

        self.default_config = config or ModelConfig()

    def _resolve_model(self, model: Optional[str]) -> Optional[str]:
        """Resolve model alias to full model ID.

        Args:
            model: Model alias or full model ID

        Returns:
            Full model ID or None if not provided
        """
        if not model:
            return None
        # Check if it's an alias
        if model in self.TEXT_MODELS:
            return self.TEXT_MODELS[model]
        if model in self.VISION_MODELS:
            return self.VISION_MODELS[model]
        # Return as-is if it's already a full model ID
        return model

    @staticmethod
    def payload_message(
        user_prompt: str,
        image_source: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> List[Dict]:
        """Create message payload for LLM API call.

        Args:
            prompt: Text prompt (required)
            image_source: Optional URL, base64-encoded image, or local file path
            system_prompt: Optional system message to set context

        Returns:
            List of message dictionaries formatted for API
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Create content array
        content = [{"type": "text", "text": user_prompt}]

        if image_source:
            image_url = image_source
            if not image_source.startswith(("http://", "https://", "data:")):
                image_url = ImageUtils.encode_to_base64(image_source)
            content.append({"type": "image_url", "image_url": {"url": image_url}})

        messages.append({"role": "user", "content": content})
        return messages

    def generate_text(
        self,
        input_data: ModelInput,
        **kwargs,
    ) -> str:
        """Send a chat completion request with flexible input types.

        Args:
            input_data: ModelInput instance with prompt and optional image_source
            **kwargs: Additional arguments to pass to the API

        Returns:
            The response content as a string

        Raises:
            ValueError: If prompt is not provided in input_data
        """
        if not input_data.user_prompt:
            raise ValueError("'prompt' must be provided in ModelInput.")

        messages = self.payload_message(input_data.user_prompt, 
                                        input_data.image_source, 
                                        input_data.system_prompt)

        completion = self.client.chat.completions.create(
            model=input_data.model,
            messages=messages,
            extra_body=input_data.extra_body or {},
            **kwargs
        )

        return completion.choices[0].message.content

    def generate_content(
        self,
        input_data: ModelInput,
        config: ModelConfig,
    ) -> str | BaseModel:
        """Generate content, choosing between text and structured output automatically.

        Decides which generation method to use based on ModelInput:
        - If response_model is provided: calls generate_structured
        - Otherwise: calls generate_text

        Args:
            input_data: ModelInput instance with prompt and optional response_model
            config: ModelConfig with model selection and inference parameters

        Returns:
            str if generating text, BaseModel instance if generating structured output

        Raises:
            ValueError: If user_prompt or model is not provided
        """
        if not input_data.user_prompt:
            raise ValueError("user_prompt must be provided in ModelInput.")

        if not config.model:
            raise ValueError("model must be provided in ModelConfig.")

        # Set model from config and resolve alias
        input_data.model = self._resolve_model(config.model)
        self.default_config = config

        if input_data.response_model:
            return self.generate_structured(input_data)
        else:
            return self.generate_text(input_data)

    def generate_structured(
        self,
        input_data: ModelInput,
    ) -> BaseModel:
        """Generate structured output using a Pydantic model schema.

        Args:
            input_data: ModelInput instance with user_prompt and response_model

        Returns:
            Instance of response_model populated with the validated API response

        Raises:
            ValueError: If response_model is not provided in input_data
            RuntimeError: If the response cannot be validated
        """
        if not input_data.response_model:
            raise ValueError("response_model must be provided in ModelInput.")

        if not input_data.user_prompt:
            raise ValueError("user_prompt must be provided in ModelInput.")

        schema = input_data.response_model.model_json_schema()
        schema_instruction = (
            f"You must respond with valid JSON that matches this exact schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            f"Respond ONLY with the JSON object, no additional text or markdown formatting."
        )

        config = self.default_config
        enhanced_system_prompt = schema_instruction.strip()
        messages = self.payload_message(input_data.user_prompt, 
                                        input_data.image_source, 
                                        enhanced_system_prompt)
        try:
            api_params = config.to_api_params()
            completion = self.client.chat.completions.create(
                model=input_data.model, messages=messages, **api_params
            )
            response_text = completion.choices[0].message.content
            json_text = self._extract_json(response_text)
            parsed_data = json.loads(json_text)
            return input_data.response_model.model_validate(parsed_data)
        except Exception as e:
            raise RuntimeError(f"Failed to generate structured output: {str(e)}")

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract JSON from text, handling markdown code blocks."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="OpenRouter CLI - Interact with various LLM models",
    )
    parser.add_argument(
        "-p", "--prompt",
        required=True,
        help="Prompt text"
    )
    parser.add_argument(
        "-m", "--model",
        default="haiku",
        help="Model to use (default: haiku)"
    )
    parser.add_argument(
        "-i", "--image",
        default = None,
        help="Image source: URL, base64 data, or local file path"
    )

    args = parser.parse_args()

    try:
        client = OpenRouterClient()
        input_data = ModelInput(
            prompt=args.prompt,
            image_source=args.image,
            model=args.model
        )
        response = client.generate_text(input_data)
        print(response)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

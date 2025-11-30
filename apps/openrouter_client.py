import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from image_utils import ImageUtils

T = TypeVar("T", bound=BaseModel)

@dataclass
class ModelInput:
    """Generic input for text generation or structured output.

    Optional dataclass for users who prefer structured parameter passing.
    All fields are optional and only relevant fields for the operation should be set.
    """
    # Text/Vision generation fields
    prompt: Optional[str] = None
    messages: Optional[List[Dict]] = None
    image_source: Optional[str] = None

    # Structured generation fields
    user_prompt: Optional[str] = None
    response_model: Optional[Type[T]] = None

    # Common fields
    model: Optional[str] = None
    max_retries: int = 3
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
        "deepseek": "deepseek/deepseek-chat-v3.1:free",
        "mistral": "mistralai/mistral-small-3.2-24b-instruct:free",
        "kimi": "moonshotai/kimi-dev-72b:free",
        "llama": "meta-llama/llama-3.3-8b-instruct:free",
        "nemotron": "nvidia/nemotron-nano-9b-v2:free",
        "gpt-oss": "openai/gpt-oss-20b:free",
        "qwen-14b": "qwen/qwen3-14b:free",
        "qwen-30b": "qwen/qwen3-30b-a3b:free",
        "qwen-235b": "qwen/qwen3-235b-a22b:free",
        "hunyuan": "tencent/hunyuan-a13b-instruct:free",
        "grok": "x-ai/grok-4-fast:free",
        "glm": "z-ai/glm-4.5-air:free"
    }

    VISION_MODELS = {
        "llama4": "meta-llama/llama-4-maverick:free",
        "gemma27b": "google/gemma-3-27b-it:free",
        "mistral": "mistralai/mistral-small-3.2-24b-instruct:free",
        "haiku": "anthropic/claude-haiku-4.5",
        "sonnet": "anthropic/claude-sonnet-4.5",
        "sonar": "perplexity/sonar",
        "sonar-pro": "perplexity/sonar-pro",
        "sonar-research": "perplexity/sonar-deep-research",
        "sonar-search": "perplexity/sonar-pro-search",
        "sonar-reason": "perplexity/sonar-reasoning-pro",
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

        self.current_model = self.VISION_MODELS["sonnet"]
        self.default_config = config or ModelConfig()

    def generate_text(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        image_source: Optional[str] = None,
        model: Optional[str] = None,
        extra_body: Optional[Dict] = None,
        **kwargs,
    ) -> str:
        """Send a chat completion request with flexible input types.

        Args:
            prompt: Simple text prompt (creates user message)
            messages: List of message dictionaries with 'role' and 'content'
            image_source: URL, base64-encoded image, or local file path (requires prompt)
            model: Optional model identifier or alias (overrides current_model)
            extra_body: Optional extra parameters for the request
            **kwargs: Additional arguments to pass to the API

        Returns:
            The response content as a string

        Raises:
            ValueError: If neither prompt nor messages provided
        """
        if image_source:
            image_url = image_source
            if not image_source.startswith(("http://", "https://", "data:")):
                image_url = ImageUtils.encode_to_base64(image_source)

            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }]
        elif prompt:
            messages = [{"role": "user", "content": prompt}]
        elif not messages:
            raise ValueError("Either prompt, messages, or image_source must be provided")

        use_model = model if model else self.current_model
        if use_model in self.TEXT_MODELS:
            use_model = self.TEXT_MODELS[use_model]
        elif use_model in self.VISION_MODELS:
            use_model = self.VISION_MODELS[use_model]

        completion = self.client.chat.completions.create(
            model=use_model, messages=messages, extra_body=extra_body or {}, **kwargs
        )
        return completion.choices[0].message.content

    def get_current_model(self) -> str:
        """Get the currently selected model."""
        return self.current_model

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model": self.current_model,
        }

    def generate_structured(
        self,
        user_prompt: str,
        response_model: Type[T],
        model: Optional[str] = None,
        max_retries: int = 3,
    ) -> T:
        """Generate structured output using a Pydantic model schema.

        Args:
            user_prompt: The user's message/prompt
            response_model: Pydantic model class defining the expected structure
            model: Optional model identifier (uses current_model if None)
            max_retries: Maximum number of retry attempts for validation failures

        Returns:
            Instance of response_model populated with the validated API response

        Raises:
            ValidationError: If the response cannot be validated after max_retries
        """
        use_model = model if model else self.current_model
        if use_model in self.TEXT_MODELS:
            use_model = self.TEXT_MODELS[use_model]
        elif use_model in self.VISION_MODELS:
            use_model = self.VISION_MODELS[use_model]

        schema = response_model.model_json_schema()
        schema_instruction = (
            f"You must respond with valid JSON that matches this exact schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            f"Respond ONLY with the JSON object, no additional text or markdown formatting."
        )

        config = self.default_config
        base_system_prompt = config.system_prompt or ""
        enhanced_system_prompt = f"{base_system_prompt}\n\n{schema_instruction}".strip()
        messages = [
            {"role": "system", "content": enhanced_system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error = None
        response_text = None

        for attempt in range(max_retries):
            try:
                api_params = config.to_api_params()
                completion = self.client.chat.completions.create(
                    model=use_model, messages=messages, **api_params
                )
                response_text = completion.choices[0].message.content
                json_text = self._extract_json(response_text)
                parsed_data = json.loads(json_text)
                return response_model.model_validate(parsed_data)
            except (json.JSONDecodeError, ValidationError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    error_msg = (
                        f"Your previous response was invalid. Error: {str(e)}\n"
                        f"Please provide a valid JSON response matching the schema exactly."
                    )
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": error_msg})
            except Exception as e:
                raise RuntimeError(f"API request failed: {str(e)}")

        raise ValidationError(
            f"Failed to generate valid structured output after {max_retries} attempts. "
            f"Last error: {str(last_error)}"
        )

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


# Example usage
if __name__ == "__main__":
    client = OpenRouterClient()

    # Simple text prompt
    response = client.generate_text("What is the meaning of life?")
    print(f"Response: {response[:100]}...")

    # Switch model using alias
    client.set_model("llama")
    print(f"\nSwitched to: {client.get_current_model()} ({client.get_model_type()})")

    # Vision with URL
    # response = client.generate_text(
    #     prompt="What do you see in this image?",
    #     image_source="https://example.com/image.jpg",
    #     model="sonnet"
    # )

    # Vision with local image file
    # response = client.generate_text(
    #     prompt="What's in this image?",
    #     image_source="path/to/local/image.jpg",
    #     model="llama4"
    # )
    # print(f"Response: {response}")

    # With full messages list
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "Hello!"}
    # ]
    # response = client.generate_text(messages=messages)

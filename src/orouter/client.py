"""OpenRouter API client for text and vision models."""

import json
import os
import random
import sys
from typing import List, Optional, Type, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .config import ModelConfig

T = TypeVar('T', bound=BaseModel)


class OpenRouterClient:
    """
    A client for interacting with the OpenRouter API with support for
    both text generation and structured output.
    """
    MODELS = [
        "deepseek/deepseek-chat-v3.1:free",
        "mistralai/mistral-small-3.2-24b-instruct:free",
        "moonshotai/kimi-dev-72b:free",
        "meta-llama/llama-3.3-8b-instruct:free",
        "nvidia/nemotron-nano-9b-v2:free",
        "openai/gpt-oss-20b:free",
        "qwen/qwen3-14b:free",
        "qwen/qwen3-30b-a3b:free",
        "qwen/qwen3-235b-a22b:free",
        "tencent/hunyuan-a13b-instruct:free",
        "x-ai/grok-4-fast:free",
        "z-ai/glm-4.5-air:free",
    ]

    MODEL_ALIASES = {
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

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the OpenRouter client.

        Args:
            config: Default ModelConfig to use for all requests.

        Raises:
            ValueError: If OPENROUTER_API_KEY is not set.
        """
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError(
                "API key not found. Set OPENROUTER_API_KEY environment variable "
                "or pass it to the constructor."
            )

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        self.default_config = config or ModelConfig()

        # Select and validate model
        if self.default_config.model is None:
            self.default_config.model = random.choice(self.MODELS)
        else:
            self.default_config.model = self.resolve_model(self.default_config.model)
            if self.default_config.model not in self.MODELS:
                print(
                    f"Warning: '{self.default_config.model}' not in predefined list. Attempting anyway...",
                    file=sys.stderr
                )

    @classmethod
    def resolve_model(cls, model_input: str) -> str:
        """
        Resolve a model alias or full name to the full model name.

        Args:
            model_input: Either an alias (e.g., 'deepseek') or full model name

        Returns:
            Full model name
        """
        return cls.MODEL_ALIASES.get(model_input.lower(), model_input)

    def generate_text(self, user_prompt: str) -> str:
        """
        Gets a completion from the OpenRouter API using the default config.

        Args:
            user_prompt: The user's message/prompt

        Returns:
            The generated text response

        Raises:
            RuntimeError: If the API request fails
        """
        config = self.default_config

        messages = []
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        try:
            api_params = config.to_api_params()
            completion = self.client.chat.completions.create(
                model=config.model,
                messages=messages,
                **api_params
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"API request failed: {str(e)}")

    def generate_structured(
        self,
        user_prompt: str,
        response_model: Type[T]
    ) -> T:
        """
        Generate structured output using a Pydantic model schema.

        This method instructs the LLM to respond with JSON matching the provided
        Pydantic model schema and validates the response.

        Args:
            user_prompt: The user's message/prompt
            response_model: Pydantic model class defining the expected structure

        Returns:
            Instance of response_model populated with the validated API response

        Raises:
            ValidationError: If the response cannot be validated
            RuntimeError: If the API request fails

        Example:
            >>> from pydantic import BaseModel
            >>> class Person(BaseModel):
            ...     name: str
            ...     age: int
            >>> client = OpenRouterClient()
            >>> person = client.generate_structured(
            ...     "Tell me about Albert Einstein",
            ...     Person
            ... )
            >>> print(person.name, person.age)
        """
        config = self.default_config

        # Generate JSON schema from Pydantic model
        schema = response_model.model_json_schema()

        # Create enhanced system prompt with schema instructions
        schema_instruction = (
            f"You must respond with valid JSON that matches this exact schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            f"Respond ONLY with the JSON object, no additional text or markdown formatting."
        )

        base_system_prompt = config.system_prompt or ""
        enhanced_system_prompt = f"{base_system_prompt}\n\n{schema_instruction}".strip()

        messages = [
            {"role": "system", "content": enhanced_system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            api_params = config.to_api_params()
            completion = self.client.chat.completions.create(
                model=config.model,
                messages=messages,
                **api_params
            )

            response_text = completion.choices[0].message.content

            # Try to extract JSON from response (handle markdown code blocks)
            json_text = self._extract_json(response_text)

            # Parse and validate with Pydantic
            parsed_data = json.loads(json_text)
            validated_response = response_model.model_validate(parsed_data)

            return validated_response

        except (json.JSONDecodeError, ValidationError) as e:
            raise ValidationError(f"Failed to generate valid structured output: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"API request failed: {str(e)}")

    @staticmethod
    def _extract_json(text: str) -> str:
        """
        Extract JSON from text, handling markdown code blocks and other formatting.

        Many LLMs wrap JSON responses in markdown code blocks like:
        ```json
        {"key": "value"}
        ```

        This method strips those wrappers to get the raw JSON.

        Args:
            text: Raw text that may contain JSON with markdown formatting

        Returns:
            Extracted JSON string without markdown wrappers
        """
        text = text.strip()

        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json or ```)
            lines = lines[1:]
            # Remove last line if it's ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        return text

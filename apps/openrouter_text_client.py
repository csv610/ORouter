import os
import sys
import json
import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Type, TypeVar
from openai import OpenAI
from pydantic import BaseModel, ValidationError

T = TypeVar('T', bound=BaseModel)

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
            api_key: Optional API key. If not provided, reads from OPENROUTER_API_KEY env variable.
            config: Default ModelConfig to use for all requests.
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
            self.default_config.model = random.choice(self.get_models())
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

    def get_models(self) -> List[str]:
        """
        Returns the list of available models.
        
        Returns:
            List of model identifiers
        """
        return self.MODELS

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
        response_model: Type[T],
        max_retries: int = 3
    ) -> T:
        """
        Generate structured output using a Pydantic model schema.
        
        This method instructs the LLM to respond with JSON matching the provided
        Pydantic model schema. It includes retry logic to handle validation failures,
        providing error feedback to the model for self-correction.
        
        Args:
            user_prompt: The user's message/prompt
            response_model: Pydantic model class defining the expected structure
            max_retries: Maximum number of retry attempts for validation failures (default: 3)
            
        Returns:
            Instance of response_model populated with the validated API response
            
        Raises:
            ValidationError: If the response cannot be validated after max_retries
            RuntimeError: If the API request fails
            
        Example:
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
        
        last_error = None
        response_text = None
        
        for attempt in range(max_retries):
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
                last_error = e
                if attempt < max_retries - 1:
                    # Add error feedback to messages for retry
                    error_msg = (
                        f"Your previous response was invalid. Error: {str(e)}\n"
                        f"Please provide a valid JSON response matching the schema exactly."
                    )
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": error_msg})
            except Exception as e:
                raise RuntimeError(f"API request failed: {str(e)}")
        
        # If we get here, all retries failed
        raise ValidationError(
            f"Failed to generate valid structured output after {max_retries} attempts. "
            f"Last error: {str(last_error)}"
        )
    
    def generate_structured_list(
        self,
        user_prompt: str,
        item_model: Type[T],
        max_retries: int = 3
    ) -> List[T]:
        """
        Generate a list of structured outputs.
        
        This is a convenience method for generating multiple items of the same type.
        It wraps the item_model in a list structure and validates each item.
        
        Args:
            user_prompt: The user's message/prompt
            item_model: Pydantic model class for each item in the list
            max_retries: Maximum number of retry attempts (default: 3)
            
        Returns:
            List of validated item_model instances
            
        Example:
            >>> class Task(BaseModel):
            ...     name: str
            ...     priority: str
            >>> client = OpenRouterClient()
            >>> tasks = client.generate_structured_list(
            ...     "Create 3 project tasks",
            ...     Task
            ... )
            >>> for task in tasks:
            ...     print(f"{task.name} - {task.priority}")
        """
        # Create a wrapper model for the list
        class ListWrapper(BaseModel):
            items: List[item_model]
        
        result = self.generate_structured(
            user_prompt=user_prompt,
            response_model=ListWrapper,
            max_retries=max_retries
        )
        
        return result.items
    
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

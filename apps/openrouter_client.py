from openai import OpenAI
import os
from typing import List, Dict, Optional, Literal

class OpenRouterChat:
    """
    A class to manage OpenRouter API interactions with multiple free models.
    Supports both text-only and vision-capable models.
    """
    
    # Available free text models
    TEXT_MODELS = [
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
        "z-ai/glm-4.5-air:free"
    ]
    
    # Available free vision models (support image inputs)
    VISION_MODELS = [
        # Add vision models here when available
        # Example: "openai/gpt-4-vision:free"
    ]
    
    # All available models
    ALL_MODELS = TEXT_MODELS + VISION_MODELS
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the OpenRouter client.
        
        Args:
            api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY env variable.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set in OPENROUTER_API_KEY environment variable")
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )
        self.current_model = self.TEXT_MODELS[0]  # Default to first text model
    
    def list_models(self, model_type: Literal["all", "text", "vision"] = "all") -> List[str]:
        """
        Get list of available free models.
        
        Args:
            model_type: Filter by model type - "all", "text", or "vision"
        
        Returns:
            List of model identifiers
        """
        if model_type == "text":
            return self.TEXT_MODELS.copy()
        elif model_type == "vision":
            return self.VISION_MODELS.copy()
        else:
            return self.ALL_MODELS.copy()
    
    def is_vision_model(self, model: str) -> bool:
        """
        Check if a model supports vision/image inputs.
        
        Args:
            model: Model identifier
        
        Returns:
            True if model supports vision, False otherwise
        """
        return model in self.VISION_MODELS
    
    def get_model_type(self, model: Optional[str] = None) -> str:
        """
        Get the type of the specified model.
        
        Args:
            model: Model identifier (uses current_model if None)
        
        Returns:
            "vision" or "text"
        """
        check_model = model if model else self.current_model
        return "vision" if self.is_vision_model(check_model) else "text"
    
    def set_model(self, model: str) -> None:
        """
        Set the current model to use.
        
        Args:
            model: Model identifier from ALL_MODELS
        
        Raises:
            ValueError: If model is not in available models list
        """
        if model not in self.ALL_MODELS:
            raise ValueError(f"Model '{model}' not found. Available models: {self.ALL_MODELS}")
        self.current_model = model
    
    def generate_text(self, 
             messages: List[Dict], 
             model: Optional[str] = None,
             extra_body: Optional[Dict] = None,
             **kwargs) -> str:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
                     For vision models, content can be a list with text and image_url
            model: Optional model to use (overrides current_model)
            extra_body: Optional extra parameters for the request
            **kwargs: Additional arguments to pass to the API
        
        Returns:
            The response content as a string
        """
        use_model = model if model else self.current_model
        
        if use_model not in self.ALL_MODELS:
            raise ValueError(f"Model '{use_model}' not found. Available models: {self.ALL_MODELS}")
        
        completion = self.client.chat.completions.create(
            model=use_model,
            messages=messages,
            **kwargs,
            extra_body=extra_body or {},
        )
        
        return completion.choices[0].message.content
    
    def quick_chat(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Quick helper method for single text prompts.

        Args:
            prompt: User prompt text
            model: Optional model to use (overrides current_model)

        Returns:
            The response content as a string
        """
        messages = [{"role": "user", "content": prompt}]
        return self.generate_text(messages, model)
    
    def vision_chat(self,
                    prompt: str,
                    image_url: str,
                    model: Optional[str] = None) -> str:
        """
        Quick helper method for vision prompts with images.

        Args:
            prompt: User prompt text
            image_url: URL or base64-encoded image
            model: Optional vision model to use

        Returns:
            The response content as a string

        Raises:
            ValueError: If the model doesn't support vision
        """
        use_model = model if model else self.current_model

        if not self.is_vision_model(use_model):
            raise ValueError(f"Model '{use_model}' does not support vision. Use a vision model: {self.VISION_MODELS}")

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }]
        return self.generate_text(messages, use_model)
    
    def get_current_model(self) -> str:
        """
        Get the currently selected model.
        
        Returns:
            Current model identifier
        """
        return self.current_model
    
    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model": self.current_model,
            "type": self.get_model_type(),
            "supports_vision": self.is_vision_model(self.current_model)
        }


# Example usage
if __name__ == "__main__":
    # Initialize the chat manager
    chat_manager = OpenRouterChat()
    
    # List models by type
    print("=== TEXT MODELS ===")
    for i, model in enumerate(chat_manager.list_models("text"), 1):
        print(f"{i}. {model}")
    
    print("\n=== VISION MODELS ===")
    vision_models = chat_manager.list_models("vision")
    if vision_models:
        for i, model in enumerate(vision_models, 1):
            print(f"{i}. {model}")
    else:
        print("No vision models available")
    
    # Get current model info
    print(f"\n=== CURRENT MODEL INFO ===")
    info = chat_manager.get_model_info()
    print(f"Model: {info['model']}")
    print(f"Type: {info['type']}")
    print(f"Supports Vision: {info['supports_vision']}")
    
    # Quick text chat
    print("\n=== TEXT CHAT EXAMPLE ===")
    response = chat_manager.quick_chat("What is the meaning of life?")
    print(f"Response: {response[:100]}...")
    
    # Switch model
    chat_manager.set_model("meta-llama/llama-3.3-8b-instruct:free")
    print(f"\nSwitched to: {chat_manager.get_current_model()} ({chat_manager.get_model_type()})")
    
    # Example of vision chat (when vision models are available)
    # response = chat_manager.vision_chat(
    #     "What do you see in this image?",
    #     "https://example.com/image.jpg",
    #     model="openai/gpt-4-vision:free"
    # )

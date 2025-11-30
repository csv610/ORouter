# OpenRouter Client

A Python client for interacting with the OpenRouter API, providing seamless access to multiple language models and vision models with a clean, intuitive interface.

## Features

- 🤖 Support for 12+ text models (DeepSeek, Mistral, Llama, Qwen, and more)
- 👁️ Vision capabilities with multiple models (Claude, Llama, Gemma, Mistral)
- 📝 Structured output generation using Pydantic models
- 🔧 Simple configuration for inference parameters
- 📸 Image handling (URLs, base64, local files)
- ⚡ Built on OpenAI SDK for compatibility

## Installation

Get your free API key at [OpenRouter](https://openrouter.ai/)

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

## Quick Start

### Basic Text Generation

```python
from openrouter_client import OpenRouterClient, ModelInput

client = OpenRouterClient()

# Simple text prompt
input_data = ModelInput(prompt="What is 2+2?")
response = client.generate_text(input_data)
print(response)  # Output: 2 + 2 = 4
```

### Using Model Aliases

```python
# Use model aliases for easier access
input_data = ModelInput(
    prompt="Explain quantum computing in one sentence.",
    model="haiku"  # Uses Claude Haiku 4.5
)
response = client.generate_text(input_data)
```

### Vision with Images

```python
# Vision with URL
input_data = ModelInput(
    prompt="What do you see in this image?",
    image_source="https://example.com/image.jpg",
    model="sonnet"
)
response = client.generate_text(input_data)

# Vision with local file
input_data = ModelInput(
    prompt="Describe this image",
    image_source="path/to/image.jpg",
    model="llama4"
)
response = client.generate_text(input_data)

# Vision with base64 data URI
input_data = ModelInput(
    prompt="What color is this?",
    image_source="data:image/png;base64,iVBORw0KGgo...",
    model="haiku"
)
response = client.generate_text(input_data)
```

### Custom Messages

```python
# Use pre-built message format
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is Python?"}
]

input_data = ModelInput(messages=messages)
response = client.generate_text(input_data)
```

## Structured Output

Generate structured data using Pydantic models:

```python
from pydantic import BaseModel
from typing import List

class Person(BaseModel):
    name: str
    age: int
    occupation: str

client = OpenRouterClient()
input_data = ModelInput(
    user_prompt="Create a person named Alice, age 28, software engineer",
    response_model=Person,
    model="sonnet"
)
result = client.generate_structured(input_data)
print(result.name)  # Alice
print(result.age)   # 28
```

### Complex Structured Output

```python
class Recipe(BaseModel):
    name: str
    servings: int
    ingredients: List[str]
    instructions: List[str]
    prep_time_minutes: int
    cook_time_minutes: int

input_data = ModelInput(
    user_prompt="Provide a recipe for chocolate chip cookies",
    response_model=Recipe,
    model="sonnet"
)
recipe = client.generate_structured(input_data)
```

### Generating Lists

```python
class PeopleList(BaseModel):
    items: List[Person]

input_data = ModelInput(
    user_prompt="Create 3 different people with names, ages, and occupations",
    response_model=PeopleList,
    model="sonnet"
)
results = client.generate_structured(input_data)
for person in results.items:
    print(f"{person.name} ({person.age}) - {person.occupation}")
```

## Configuration

### Model Configuration

```python
from openrouter_client import OpenRouterClient, ModelConfig, ModelInput

config = ModelConfig(
    temperature=0.7,          # Controls randomness (0.0-2.0)
    max_tokens=500,           # Maximum response length
    top_p=1.0,                # Nucleus sampling parameter
    frequency_penalty=0.0,    # Penalize frequent tokens (-2.0 to 2.0)
    presence_penalty=0.0,     # Penalize new tokens (-2.0 to 2.0)
    extra_body={}             # Additional API parameters
)

client = OpenRouterClient(config=config)
input_data = ModelInput(prompt="Tell me about AI")
response = client.generate_text(input_data)
```

## Available Models

### Text Models

| Alias | Full Model ID |
|-------|---------------|
| `deepseek` | `deepseek/deepseek-chat-v3.1:free` |
| `mistral` | `mistralai/mistral-small-3.2-24b-instruct:free` |
| `kimi` | `moonshotai/kimi-dev-72b:free` |
| `llama` | `meta-llama/llama-3.3-8b-instruct:free` |
| `nemotron` | `nvidia/nemotron-nano-9b-v2:free` |
| `gpt-oss` | `openai/gpt-oss-20b:free` |
| `qwen-14b` | `qwen/qwen3-14b:free` |
| `qwen-30b` | `qwen/qwen3-30b-a3b:free` |
| `qwen-235b` | `qwen/qwen3-235b-a22b:free` |
| `hunyuan` | `tencent/hunyuan-a13b-instruct:free` |
| `grok` | `x-ai/grok-4-fast:free` |
| `glm` | `z-ai/glm-4.5-air:free` |

### Vision Models

| Alias | Full Model ID |
|-------|---------------|
| `llama4` | `meta-llama/llama-4-maverick:free` |
| `gemma27b` | `google/gemma-3-27b-it:free` |
| `mistral` | `mistralai/mistral-small-3.2-24b-instruct:free` |
| `haiku` | `anthropic/claude-haiku-4.5` |
| `sonnet` | `anthropic/claude-sonnet-4.5` |
| `sonar` | `perplexity/sonar` |
| `sonar-pro` | `perplexity/sonar-pro` |
| `sonar-research` | `perplexity/sonar-deep-research` |
| `sonar-search` | `perplexity/sonar-pro-search` |
| `sonar-reason` | `perplexity/sonar-reasoning-pro` |

## API Reference

### OpenRouterClient

#### `__init__(api_key: Optional[str] = None, config: Optional[ModelConfig] = None)`

Initialize the OpenRouter client.

**Parameters:**
- `api_key`: OpenRouter API key (defaults to `OPENROUTER_API_KEY` env variable)
- `config`: Optional `ModelConfig` for default inference parameters

#### `generate_text(input_data: ModelInput, **kwargs) -> str`

Send a chat completion request with flexible input types.

**Parameters:**
- `input_data`: `ModelInput` instance with prompt, messages, or image_source
- `**kwargs`: Additional arguments to pass to the API

**Returns:** Response content as a string

**Raises:** `ValueError` if neither prompt nor messages provided in input_data

#### `generate_structured(input_data: ModelInput) -> BaseModel`

Generate structured output using a Pydantic model schema.

**Parameters:**
- `input_data`: `ModelInput` instance with user_prompt and response_model

**Returns:** Instance of response_model populated with validated API response

**Raises:**
- `ValueError` if response_model or user_prompt not provided in input_data
- `RuntimeError` if response cannot be validated

### ModelInput

Dataclass for unified input to both text and structured generation methods.

**Fields:**
- `prompt`: Optional[str] - Text prompt for text/vision generation
- `messages`: Optional[List[Dict]] - Pre-built message format
- `image_source`: Optional[str] - Image URL, base64, or local file path
- `user_prompt`: Optional[str] - Text prompt for structured generation
- `response_model`: Optional[Type[BaseModel]] - Pydantic model for structured output
- `model`: Optional[str] - Model alias or full model ID
- `extra_body`: Optional[Dict] - Additional API parameters

### ModelConfig

Dataclass for inference parameter configuration.

**Fields:**
- `temperature`: float = 0.7 - Controls randomness (0.0-2.0)
- `max_tokens`: Optional[int] - Maximum response length
- `top_p`: float = 1.0 - Nucleus sampling (0.0-1.0)
- `frequency_penalty`: float = 0.0 - Penalize frequent tokens (-2.0 to 2.0)
- `presence_penalty`: float = 0.0 - Penalize new tokens (-2.0 to 2.0)
- `extra_body`: Dict[str, Any] - Additional API parameters

## Testing

The client includes comprehensive test suites:

```bash
# Text generation tests
python test_text.py

# Vision tests
python test_vision.py

# Structured output tests
python test_structured.py
```

All tests pass with the current implementation (15/15 tests):
- 4/4 text generation tests
- 5/5 vision tests
- 6/6 structured output tests

## Examples

### Sentiment Analysis

```python
from pydantic import BaseModel

class SentimentAnalysis(BaseModel):
    sentiment: str  # positive, negative, neutral
    confidence: float  # 0.0 to 1.0
    key_phrases: List[str]

text = "I absolutely love this product! It's amazing and works perfectly."
input_data = ModelInput(
    user_prompt=f"Analyze sentiment: {text}",
    response_model=SentimentAnalysis,
    model="haiku"
)
result = client.generate_structured(input_data)
print(f"Sentiment: {result.sentiment} ({result.confidence:.2f})")
```

### Movie Review

```python
class MovieReview(BaseModel):
    title: str
    rating: float
    genres: List[str]
    summary: str
    strengths: List[str]
    weaknesses: List[str]
    would_recommend: bool

input_data = ModelInput(
    user_prompt="Review the movie 'The Matrix' from 1999",
    response_model=MovieReview,
    model="sonnet"
)
review = client.generate_structured(input_data)
print(f"{review.title}: {review.rating}/10.0")
print(f"Recommend: {review.would_recommend}")
```

## Error Handling

```python
try:
    input_data = ModelInput(prompt="Hello")
    response = client.generate_text(input_data)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"API error: {e}")
```

## License

MIT License - See LICENSE file for details

## Support

For issues and feature requests, visit the project repository or contact OpenRouter support at https://openrouter.ai/

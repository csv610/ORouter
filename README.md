# OpenRouter Python Client

A Python library for interacting with [OpenRouter](https://openrouter.ai/) - a free AI model aggregator that provides access to various large language models (LLMs) and vision models.

## Features

- **Text Generation**: Generate text with multiple LLM models via aliases or full model IDs
- **Vision Support**: Process images from URLs, data URIs, or local file paths
- **Structured Output**: Generate structured JSON/Pydantic models with automatic validation and retries
- **Model Aliases**: Easy-to-use aliases for text and vision models (e.g., "haiku", "sonnet", "mistral")
- **Multiple Models**: 12+ text models and 10+ vision models available through OpenRouter
- **Flexible Configuration**: Customize temperature, top-p, penalties, max tokens, and system prompts
- **Automatic Retries**: Built-in retry mechanism for structured output validation failures
- **Base64 Image Conversion**: Automatically converts local image files to base64 for API submission

## Installation

### Requirements
- Python 3.8+
- OpenRouter API Key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/csv610/ORouter.git
cd ORouter
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set your OpenRouter API key:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

Get your free API key at [OpenRouter](https://openrouter.ai/)

## Quick Start

### Using the Python Library

```python
from orouter import OpenRouterClient, ModelConfig, ModelInput

# Initialize the client
client = OpenRouterClient()

# Simple text generation
response = client.generate_text("What is Python?")
print(response)

# Use a specific model alias
response = client.generate_text(
    prompt="Explain quantum computing",
    model="sonnet"  # Uses Claude Sonnet 4.5
)

# Vision - from URL
response = client.generate_text(
    prompt="What's in this image?",
    image_source="https://example.com/image.jpg",
    model="haiku"
)

# Vision - from local file
response = client.generate_text(
    prompt="Describe this image",
    image_source="/path/to/local/image.png",
    model="sonnet"
)

# Structured output with Pydantic
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int
    occupation: str

person = client.generate_structured(
    user_prompt="Create a person named Alice, age 28, software engineer",
    response_model=Person,
    model="sonnet"
)
print(f"{person.name} is {person.age} and works as a {person.occupation}")

# List of structured outputs
from typing import List

class PeopleList(BaseModel):
    items: List[Person]

result = client.generate_structured(
    user_prompt="Create 3 people",
    response_model=PeopleList,
    model="sonnet"
)
people = result.items

# Custom configuration
config = ModelConfig(
    temperature=0.5,
    max_tokens=200,
    system_prompt="You are a helpful assistant"
)
client = OpenRouterClient(config=config)

# Alternative: using ModelInput dataclass
input_data = ModelInput(
    prompt="What is machine learning?",
    model="sonnet"
)
response = client.generate_text(input_data.prompt, model=input_data.model)
```

### Using the CLI

**Text Query CLI:**
```bash
python apps/text_query_cli.py "What is artificial intelligence?"
```

**Vision Query CLI:**
```bash
python apps/vision_query_cli.py "What's in this image?" --image "path/to/image.jpg"
```

**Medical Topic Assistant:**
```bash
python apps/medtopic_cli.py --topic "diabetes" --model "sonnet"
```

## Available Models

### Text Models

Use aliases for convenience or full model IDs:

| Alias | Full Model ID |
|-------|---------------|
| deepseek | `deepseek/deepseek-chat-v3.1:free` |
| mistral | `mistralai/mistral-small-3.2-24b-instruct:free` |
| kimi | `moonshotai/kimi-dev-72b:free` |
| llama | `meta-llama/llama-3.3-8b-instruct:free` |
| nemotron | `nvidia/nemotron-nano-9b-v2:free` |
| gpt-oss | `openai/gpt-oss-20b:free` |
| qwen-14b | `qwen/qwen3-14b:free` |
| qwen-30b | `qwen/qwen3-30b-a3b:free` |
| qwen-235b | `qwen/qwen3-235b-a22b:free` |
| hunyuan | `tencent/hunyuan-a13b-instruct:free` |
| grok | `x-ai/grok-4-fast:free` |
| glm | `z-ai/glm-4.5-air:free` |

### Vision Models

| Alias | Full Model ID |
|-------|---------------|
| llama4 | `meta-llama/llama-4-maverick:free` |
| gemma27b | `google/gemma-3-27b-it:free` |
| mistral | `mistralai/mistral-small-3.2-24b-instruct:free` |
| haiku | `anthropic/claude-haiku-4.5` |
| sonnet | `anthropic/claude-sonnet-4.5` |
| sonar | `perplexity/sonar` |
| sonar-pro | `perplexity/sonar-pro` |
| sonar-research | `perplexity/sonar-deep-research` |
| sonar-search | `perplexity/sonar-pro-search` |
| sonar-reason | `perplexity/sonar-reasoning-pro` |

## API Documentation

### OpenRouterClient Class

#### Methods

**`__init__(api_key: Optional[str] = None, config: Optional[ModelConfig] = None)`**
- Initialize the OpenRouter client
- If `api_key` is None, reads from `OPENROUTER_API_KEY` environment variable
- `config`: Optional ModelConfig for default inference parameters

**`generate_text(prompt: Optional[str] = None, messages: Optional[List[Dict]] = None, image_source: Optional[str] = None, model: Optional[str] = None, extra_body: Optional[Dict] = None, **kwargs) -> str`**
- Send a chat completion request with flexible input types
- `prompt`: Simple text prompt (creates user message)
- `messages`: List of message dictionaries with 'role' and 'content'
- `image_source`: URL, base64-encoded image, or local file path
- `model`: Optional model identifier or alias (overrides current_model)
- Returns response content as string

**`get_current_model() -> str`**
- Returns the currently selected model

**`get_model_info() -> Dict[str, Any]`**
- Returns information about the current model

**`generate_structured(user_prompt: str, response_model: Type[T], model: Optional[str] = None, max_retries: int = 3) -> T`**
- Generate structured output using a Pydantic model schema
- `user_prompt`: The user's message/prompt
- `response_model`: Pydantic model class defining the expected structure
- `model`: Optional model identifier (uses current_model if None)
- `max_retries`: Maximum number of retry attempts for validation failures
- Returns instance of response_model populated with validated API response


## Project Structure

```
ORouter/
├── orouter/
│   ├── __init__.py                 # Package exports
│   ├── openrouter_client.py        # Main client library (OpenRouterClient)
│   ├── config.py                   # ModelConfig dataclass
│   └── image_utils.py              # Image processing utilities
├── apps/
│   ├── text_query_cli.py           # CLI for text queries
│   ├── vision_query_cli.py         # CLI for vision queries
│   ├── medtopic_cli.py             # Medical topic assistant
│   ├── compare_text_models_cly.py  # Model comparison tool
│   └── CLAUDE.md                   # Claude-specific instructions
├── docs/
│   ├── CODE_OF_CONDUCT.md          # Community guidelines
│   └── CONTRIBUTING.md             # Contribution guidelines
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── LICENSE                         # License file
```

## Configuration

### ModelConfig

Customize default inference parameters when initializing the client:

```python
from orouter import OpenRouterClient, ModelConfig

config = ModelConfig(
    temperature=0.7,           # Controls randomness (0.0-2.0)
    top_p=0.9,                # Top-p sampling
    frequency_penalty=0.0,     # Frequency penalty
    presence_penalty=0.0,      # Presence penalty
    max_tokens=1000           # Maximum response length
)

client = OpenRouterClient(config=config)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This project is not affiliated with OpenRouter, OpenAI, or any of the model providers. Use the OpenRouter API responsibly and respect their terms of service.

## Support

For issues and questions:
- Check [OpenRouter Documentation](https://openrouter.ai/docs)
- Open an issue on GitHub
- Review existing issues and discussions

## Usage Examples

### Text Generation
```bash
python apps/text_query_cli.py "Explain machine learning in simple terms"
```

### Vision Analysis
```bash
python apps/vision_query_cli.py "Describe this image" --image "https://example.com/image.jpg"
```

### Medical Topic Assistant
```bash
python apps/medtopic_cli.py --topic "hypertension" --detail
```

## Changelog

### v1.0.0 (Current Release)
- Refactored OpenRouterClient with modern API
- Full text generation support with flexible inputs
- Complete vision model support (URLs, data URIs, local files)
- Structured output generation with Pydantic validation
- Automatic retry mechanism for validation failures
- Model alias system for easy model selection
- Comprehensive test coverage (15+ tests)
- Updated documentation and examples

### v0.1.0 (Initial Release)
- Initial implementation of OpenRouterChat class
- Basic text chat support
- Multiple model support
- CLI tools for querying models

## Roadmap

- [ ] Streaming response support
- [ ] Response caching
- [ ] Cost tracking and monitoring
- [ ] Batch processing support
- [ ] Conversation history management
- [ ] Multi-turn conversation helpers

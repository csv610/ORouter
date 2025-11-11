# OpenRouter Python Client

A Python library for interacting with [OpenRouter](https://openrouter.ai/) - a free AI model aggregator that provides access to various large language models (LLMs) and vision models.

## Features

- **Multiple Free Models**: Access to 12+ free LLM models including:
  - DeepSeek Chat v3.1
  - Mistral Small 3.2
  - Meta Llama 3.3
  - Qwen 3
  - Grok 4 Fast
  - And more...

- **Easy Integration**: Simple Python API for text and vision chat
- **Model Switching**: Easily switch between different models
- **Vision Support**: Ready for vision-capable models (coming soon)
- **CLI Tools**: Command-line interfaces for quick queries
- **Flexible Configuration**: Customize temperature, top-p, penalties, and more

## Installation

### Requirements
- Python 3.8+
- OpenRouter API Key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/openrouter-python.git
cd openrouter-python
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
from apps.openrouter_client import OpenRouterChat

# Initialize the client
chat = OpenRouterChat()

# Quick text chat
response = chat.quick_chat("What is Python?")
print(response)

# Switch models
chat.set_model("meta-llama/llama-3.3-8b-instruct:free")

# List available models
models = chat.list_models("text")
for model in models:
    print(model)

# Get model information
info = chat.get_model_info()
print(f"Current model: {info['model']}")
print(f"Type: {info['type']}")
```

### Using the CLI

List available models:
```bash
python apps/cli_text_client.py -l
```

Query a model:
```bash
python apps/cli_text_client.py -q "What is artificial intelligence?" -v
```

With specific model and settings:
```bash
python apps/cli_text_client.py \
  -q "Explain quantum computing" \
  -m "meta-llama/llama-3.3-8b-instruct:free" \
  -t 0.7 \
  --max-tokens 500
```

## Available Models

### Text Models (Free)
- `deepseek/deepseek-chat-v3.1:free`
- `mistralai/mistral-small-3.2-24b-instruct:free`
- `moonshotai/kimi-dev-72b:free`
- `meta-llama/llama-3.3-8b-instruct:free`
- `nvidia/nemotron-nano-9b-v2:free`
- `openai/gpt-oss-20b:free`
- `qwen/qwen3-14b:free`
- `qwen/qwen3-30b-a3b:free`
- `qwen/qwen3-235b-a22b:free`
- `tencent/hunyuan-a13b-instruct:free`
- `x-ai/grok-4-fast:free`
- `z-ai/glm-4.5-air:free`

### Vision Models
Vision model support is coming soon!

## API Documentation

### OpenRouterChat Class

#### Methods

**`__init__(api_key: Optional[str] = None)`**
- Initialize the OpenRouter client
- If `api_key` is None, reads from `OPENROUTER_API_KEY` environment variable

**`list_models(model_type: str = "all") -> List[str]`**
- Returns list of available models
- `model_type`: "all", "text", or "vision"

**`set_model(model: str) -> None`**
- Set the current model to use
- Raises `ValueError` if model not found

**`get_current_model() -> str`**
- Returns currently selected model identifier

**`generate_text(messages: List[Dict], model: Optional[str] = None, **kwargs) -> str`**
- Send a chat completion request
- `messages`: List of message dicts with 'role' and 'content'
- Returns response as string

**`quick_chat(prompt: str, model: Optional[str] = None) -> str`**
- Quick method for single text prompts
- Returns response as string

**`vision_chat(prompt: str, image_url: str, model: Optional[str] = None) -> str`**
- Support for vision-capable models with image inputs
- `image_url`: URL or base64-encoded image
- Raises `ValueError` if model doesn't support vision

**`get_model_info() -> Dict[str, any]`**
- Returns information about current model

**`is_vision_model(model: str) -> bool`**
- Check if a model supports vision/image inputs

## Project Structure

```
openrouter-python/
├── apps/
│   ├── openrouter_client.py        # Main client library
│   ├── cli_text_client.py          # CLI for text queries
│   ├── cli_compare_text_clients.py # Model comparison tool
│   ├── cli_medtopic.py             # Medical topic assistant
│   ├── text_chat.py                # Text chat interface
│   ├── vision_chat.py              # Vision chat interface
│   ├── openrouter_text_client.py   # Alternative text client
│   └── medical_reports/            # Sample medical data
├── openrouter/                     # Package module (extensible)
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
├── LICENSE                         # MIT License
└── README.md                       # This file
```

## Configuration

### CLI Arguments

The CLI tool supports the following arguments:
- `-q, --question`: Query text (default: "Hello, who are you?")
- `-m, --model`: Specific model to use
- `-s, --system`: System prompt
- `-t, --temp`: Temperature (0.0-2.0, default: 1.0)
- `--top-p`: Top-p sampling (default: 1.0)
- `--frequency-penalty`: Frequency penalty (default: 0.0)
- `--presence-penalty`: Presence penalty (default: 0.0)
- `--max-tokens`: Maximum response tokens
- `-l, --list-models`: List all available models
- `-v, --verbose`: Print verbose output

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

## Changelog

### v0.1.0 (Initial Release)
- Initial implementation of OpenRouterChat class
- Basic text chat support
- Multiple model support
- CLI tools for querying models
- Vision chat framework (ready for vision models)

## Roadmap

- [ ] Vision model support integration
- [ ] Streaming response support
- [ ] Response caching
- [ ] Cost tracking and monitoring
- [ ] Batch processing support
- [ ] Conversation history management

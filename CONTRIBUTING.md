# Contributing to OpenRouter Python Client

Thank you for your interest in contributing to the OpenRouter Python Client! We welcome contributions of all kinds, including bug reports, feature requests, documentation improvements, and code contributions.

## Code of Conduct

Please note that this project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- OpenRouter API key (get a free key at [OpenRouter](https://openrouter.ai/))

### Setting Up Your Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ORouter.git
   cd ORouter
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

5. Set up your environment variables:
   ```bash
   export OPENROUTER_API_KEY="your-api-key-here"
   ```

## Making Changes

### Code Style

We follow the PEP 8 style guide with some modifications:

- **Line Length**: Maximum 100 characters
- **Formatter**: Black
- **Import Sorting**: isort with Black profile
- **Type Hints**: Encouraged but not enforced

Before committing, run the formatting tools:

```bash
# Format code with Black
black .

# Sort imports with isort
isort .

# Check code style with flake8
flake8 .

# Type checking with mypy
mypy apps/
```

### Creating a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-streaming-support`
- `fix/correct-model-selection`
- `docs/update-api-documentation`
- `test/add-vision-tests`

### Commit Messages

Write clear, descriptive commit messages:

```
Short summary (50 characters or less)

More detailed explanation of the changes, if needed.
Explain the why, not just the what.

- Bullet points are okay for lists
- Keep lines under 72 characters
```

Example:
```
Add streaming support for text responses

Implement server-sent events (SSE) handling to support
streaming responses from the OpenRouter API. This allows
for real-time output as the model generates text.

- Add StreamingChat class
- Implement SSE parser
- Update CLI to support --stream flag
- Add streaming examples to documentation
```

### Testing

We use pytest for testing. Before submitting a pull request:

1. Write tests for your changes
2. Ensure all tests pass:
   ```bash
   pytest
   ```

3. Check code coverage:
   ```bash
   pytest --cov=apps --cov-report=html
   ```

Tests should be placed in a `tests/` directory that mirrors the `apps/` structure.

### Documentation

- Update the README.md for user-facing changes
- Add docstrings to new functions and classes
- Update the API Documentation section in README.md for new methods
- Document CLI arguments in the Configuration section

## Submitting Changes

### Pull Request Process

1. Ensure your code follows our style guidelines
2. Ensure all tests pass locally
3. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a Pull Request on GitHub with:
   - A clear, descriptive title
   - A detailed description of what your changes do
   - Reference to any related issues (e.g., "Fixes #123")
   - A summary of testing performed

### PR Template

```markdown
## Description
Brief description of the changes

## Related Issues
Fixes #(issue number)

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Added/updated tests
- [ ] All tests pass locally
- [ ] Code coverage maintained or improved

## Documentation
- [ ] Updated README.md
- [ ] Added/updated docstrings
- [ ] Updated API documentation if applicable

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
```

## Reporting Bugs

Before creating a bug report, check the issue list to ensure the bug hasn't already been reported.

When reporting a bug, include:

- **Title**: Clear, descriptive bug title
- **Description**: What is the bug?
- **Steps to Reproduce**: Exact steps to reproduce
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Environment**: Python version, OS, relevant package versions
- **Error Messages**: Full traceback if applicable
- **Screenshots**: If applicable

## Suggesting Features

We'd love to hear about features you'd like to see! When suggesting a feature:

1. Check the issue list to see if it's already been suggested
2. Provide a clear description of the feature
3. Explain the use case and why it would be useful
4. Provide examples of how it might be used
5. Suggest a possible implementation (optional but helpful)

## Questions?

Feel free to open an issue with the `question` label if you have any questions about the codebase or contributing process.

## License

By contributing to this project, you agree that your contributions will be licensed under its MIT License.

## Recognition

Contributors will be recognized in:
- The commit history
- Pull request discussions
- Future CONTRIBUTORS.md file

Thank you for contributing! 🎉

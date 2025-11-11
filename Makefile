.PHONY: help install install-dev test format lint type-check clean build push

help:
	@echo "OpenRouter Python Client - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install       - Install core dependencies"
	@echo "  make install-dev   - Install dependencies including dev tools"
	@echo ""
	@echo "Development:"
	@echo "  make format        - Format code with Black and isort"
	@echo "  make lint          - Run flake8 linting checks"
	@echo "  make type-check    - Run mypy type checking"
	@echo "  make test          - Run pytest tests"
	@echo "  make check         - Run all checks (format, lint, type-check, test)"
	@echo ""
	@echo "Build & Deploy:"
	@echo "  make build         - Build distribution packages"
	@echo "  make push          - Push to GitHub (commits must be staged)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         - Remove build artifacts, caches, and temp files"
	@echo ""

install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"

format:
	black .
	isort .

lint:
	flake8 .

type-check:
	mypy apps/

test:
	pytest

check: format lint type-check test
	@echo "✓ All checks passed!"

build:
	python -m build

push:
	git push origin main

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .eggs/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	@echo "✓ Cleaned up!"

.PHONY: venv lint format test

venv:
	uv venv
	uv pip install -e ".[dev]"

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest

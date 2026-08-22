UV ?= uv

.PHONY: test format lint typecheck check

test:
	$(UV) run pytest

format:
	$(UV) run ruff format --check .

lint:
	$(UV) run ruff check .

typecheck:
	$(UV) run mypy

check: format lint typecheck test

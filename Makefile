FUNCTIONS := data/input/functions_definition.json
INPUT     := data/input/function_calling_tests.json
OUTPUT    := data/output/function_calling_results.json

.PHONY: all install run debug lint lint-strict clean help

all: run

install:
	uv sync

run:
	uv run python -m src \
		--functions_definition $(FUNCTIONS) \
		--input $(INPUT) \
		--output $(OUTPUT)

debug:
	uv run python -m pdb -m src \
		--functions_definition $(FUNCTIONS) \
		--input $(INPUT) \
		--output $(OUTPUT)

lint:
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src
	uv run mypy src --strict

clean:
	find . -type d -name __pycache__  -exec rm -rf {} +
	find . -type d -name .mypy_cache  -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

help:
	@echo ""
	@echo "  make install      Install project dependencies"
	@echo "  make run          Run the full pipeline"
	@echo "  make debug        Run the pipeline under pdb (Python debugger)"
	@echo "  make lint         flake8 + mypy (required flags from subject)"
	@echo "  make lint-strict  flake8 + mypy --strict"
	@echo "  make clean        Remove __pycache__, .mypy_cache, and .pyc files"
	@echo ""
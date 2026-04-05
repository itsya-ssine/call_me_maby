"""File I/O utilities for reading and writing JSON data."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from src.models import FunctionCall, FunctionDefinition, Prompt


def load_json_file(path: str) -> Any:
    """Load and parse a JSON file, returning the parsed content.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content.

    Raises:
        SystemExit: On file not found, permission error, or invalid JSON.
    """
    file_path = Path(path)
    if not file_path.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in '{path}': {e}", file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        print(f"[ERROR] Permission denied reading '{path}'", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"[ERROR] Could not read '{path}': {e}", file=sys.stderr)
        sys.exit(1)


def load_function_definitions(path: str) -> List[FunctionDefinition]:
    """Load function definitions from a JSON file.

    Args:
        path: Path to function_definitions.json.

    Returns:
        List of FunctionDefinition objects.
    """
    raw: Any = load_json_file(path)
    if not isinstance(raw, list):
        print(f"[ERROR] Expected a JSON array in '{path}'", file=sys.stderr)
        sys.exit(1)
    definitions: List[FunctionDefinition] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            print(
                f"[ERROR] Item {i} in '{path}' is not an object.",
                file=sys.stderr
            )
            sys.exit(1)
        try:
            definitions.append(FunctionDefinition.model_validate(item))
        except Exception as e:
            print(
                "[ERROR] Could not parse function definition ",
                f"at index {i}: {e}",
                file=sys.stderr,
            )
            sys.exit(1)
    return definitions


def load_prompts(path: str) -> List[Prompt]:
    """Load prompts from a JSON file.

    Args:
        path: Path to function_calling_tests.json.

    Returns:
        List of Prompt objects.
    """
    raw: Any = load_json_file(path)
    if not isinstance(raw, list):
        print(f"[ERROR] Expected a JSON array in '{path}'", file=sys.stderr)
        sys.exit(1)
    prompts: List[Prompt] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            print(
                f"[ERROR] Item {i} in '{path}' is not an object.",
                file=sys.stderr
            )
            continue
        try:
            prompts.append(Prompt.model_validate(item))
        except Exception as e:
            print(
                f"[WARNING] Skipping prompt at index {i}: {e}",
                file=sys.stderr
            )
    return prompts


def save_results(results: List[FunctionCall], path: str) -> None:
    """Save function call results to a JSON file.

    Args:
        results: List of FunctionCall objects.
        path: Output file path.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized: List[Dict[str, Any]] = [r.model_dump() for r in results]
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Results written to '{path}'")
    except OSError as e:
        print(f"[ERROR] Could not write to '{path}': {e}", file=sys.stderr)
        sys.exit(1)

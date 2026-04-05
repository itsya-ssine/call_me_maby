"""Entry point for the call-me-maybe function calling tool.

Usage::

    uv run python -m src \\
        [--functions_definition <path>] \\
        [--input <path>] \\
        [--output <path>]
"""

import argparse
import sys
from typing import Any, List

from src.file_io import load_function_definitions, load_prompts, save_results
from src.function_selector import select_function
from src.models import AppConfig, FunctionCall


def _load_model(model_name: str) -> Any:
    """Attempt to load the Small_LLM_Model from llm_sdk.

    Args:
        model_name: HuggingFace-style model identifier.

    Returns:
        Model instance, or None on failure.
    """
    try:
        from llm_sdk import Small_LLM_Model
        print(f"[INFO] Loading model '{model_name}' …", flush=True)
        model: Any = Small_LLM_Model(model_name)
        print("[INFO] Model loaded.", flush=True)
        return model
    except ImportError:
        print(
            "[ERROR] llm_sdk package not found.  ",
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(
            f"[ERROR] Failed to load model '{model_name}': {e}",
            file=sys.stderr
        )
        return None


def _parse_args() -> AppConfig:
    """Parse command-line arguments.

    Returns:
        AppConfig populated from CLI flags / defaults.
    """
    parser = argparse.ArgumentParser(
        description="Function calling tool with constrained decoding."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to input prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calls.json",
        help="Path for the output JSON file.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        help="Model name to use (default: Qwen/Qwen3-0.6B).",
    )
    args = parser.parse_args()
    return AppConfig(
        functions_definition=args.functions_definition,
        input=args.input,
        output=args.output,
        model_name=args.model,
    )


def main() -> int:
    """Run the function-calling pipeline.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    config = _parse_args()

    # Load function definitions
    functions = load_function_definitions(config.functions_definition)
    if not functions:
        print("[ERROR] No function definitions loaded.", file=sys.stderr)
        return 1
    print(f"[INFO] Loaded {len(functions)} function definition(s).")

    # Load prompts
    prompts = load_prompts(config.input)
    if not prompts:
        print("[ERROR] No prompts loaded.", file=sys.stderr)
        return 1
    print(f"[INFO] Loaded {len(prompts)} prompt(s).")

    # Load model
    model = _load_model(config.model_name or "Qwen/Qwen3-0.6B")
    if model is None:
        return 1

    # Process each prompt
    results: List[FunctionCall] = []
    for i, prompt_obj in enumerate(prompts):
        print(
            f"[INFO] Processing prompt {i + 1}/{len(prompts)}: "
            f"'{prompt_obj.prompt}'",
            flush=True,
        )
        fc = select_function(model, prompt_obj.prompt, functions)
        if fc is not None:
            results.append(fc)
            print(f"         → {fc.name}({fc.parameters})")
        else:
            print(
                f"[WARNING] Could not resolve prompt: '{prompt_obj.prompt}'",
                file=sys.stderr,
            )

    # Save results
    save_results(results, config.output)
    print(f"[INFO] Done. {len(results)}/{len(prompts)} prompts resolved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

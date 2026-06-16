"""Entry point for the call-me-maybe function calling tool.

Usage::

    uv run python -m src \\
        [--functions_definition <path>] \\
        [--input <path>] \\
        [--output <path>]
"""

import argparse
import sys
import time
from typing import Any, List, Optional

from src.file_io import load_function_definitions, load_prompts, save_results
from src.function_selector import select_function_batch
from src.models import AppConfig, FunctionCall


# ============================================================================
# OPTIMIZATION: Lazy model loading with warm-up
# ============================================================================

def _load_model(model_name: str, warmup: bool = True) -> Any:
    """Attempt to load the Small_LLM_Model from llm_sdk.
    
    OPTIMIZED: Performs a warm-up inference to initialize GPU kernels
    and allocate memory before the actual workload.

    Args:
        model_name: HuggingFace-style model identifier.
        warmup: Whether to run a warm-up inference pass.

    Returns:
        Model instance, or None on failure.
    """
    try:
        from llm_sdk import Small_LLM_Model
        print(f"[INFO] Loading model '{model_name}' …", flush=True)
        start = time.time()
        
        model: Any = Small_LLM_Model(model_name)
        
        load_time = time.time() - start
        print(f"[INFO] Model loaded in {load_time:.1f}s.", flush=True)
        
        # Warm-up: run a dummy inference to initialize CUDA/MPS kernels
        if warmup:
            print("[INFO] Warming up model …", flush=True)
            warm_start = time.time()
            _warmup_model(model)
            print(f"[INFO] Warm-up complete in {time.time() - warm_start:.1f}s.", flush=True)
        
        return model
        
    except ImportError:
        print(
            "[ERROR] llm_sdk package not found. Install it first.",
            file=sys.stderr,
        )
        return None
    except Exception as e:
        print(
            f"[ERROR] Failed to load model '{model_name}': {e}",
            file=sys.stderr
        )
        return None


def _warmup_model(model: Any) -> None:
    """Run a short warm-up inference to initialize GPU kernels.
    
    This avoids cold-start latency on the first real query.
    """
    warmup_prompt = "Hello"
    try:
        encoded = model.encode(warmup_prompt)
        if hasattr(encoded, 'tolist'):
            ids = encoded.tolist()
        else:
            ids = list(encoded)
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        
        # Run one forward pass
        _ = model.get_logits_from_input_ids([int(x) for x in ids])
    except Exception:
        pass  # Warm-up failure is non-critical


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
    parser.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help="Use batch-optimized processing (shared vocabulary, cached prompts).",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        default=False,
        help="Skip model warm-up (faster startup, slower first query).",
    )
    args = parser.parse_args()
    return AppConfig(
        functions_definition=args.functions_definition,
        input=args.input,
        output=args.output,
        model_name=args.model,
    ), args.batch, args.no_warmup


def main() -> int:
    """Run the function-calling pipeline.

    OPTIMIZED: Uses batch processing with shared vocabulary and
    pre-computed function parameter maps to avoid redundant work.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    config, use_batch, skip_warmup = _parse_args()

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

    # Load model (with warm-up unless disabled)
    model = _load_model(config.model_name or "Qwen/Qwen3-0.6B", 
                        warmup=not skip_warmup)
    if model is None:
        return 1

    total_start = time.time()
    
    if use_batch:
        # OPTIMIZED: Batch mode — share vocabulary and pre-compute structures
        results = _process_batch(model, prompts, functions)
    else:
        # Standard mode (one-by-one, but still with optimizations)
        results = _process_sequential(model, prompts, functions)

    total_time = time.time() - total_start
    mins, secs = divmod(total_time, 60)
    print(f"[INFO] Total processing time: {int(mins)}m {secs:.1f}s")
    
    if results:
        avg_time = total_time / len(results)
        print(f"[INFO] Average per prompt: {avg_time:.2f}s")

    # Save results
    save_results(results, config.output)
    print(f"[INFO] Done. {len(results)}/{len(prompts)} prompts resolved.")
    return 0


# ============================================================================
# OPTIMIZATION: Batch processing with shared resources
# ============================================================================

def _process_batch(
    model: Any,
    prompts: List[Any],
    functions: List[Any],
) -> List[FunctionCall]:
    """Process all prompts using batch-optimized path.
    
    OPTIMIZATIONS:
    1. Vocabulary loaded ONCE and shared across all prompts
    2. Function parameter maps pre-computed ONCE
    3. Prompt templates pre-built for each function combination
    """
    from src.constrained_decoder import load_vocabulary
    
    # Load vocabulary once (expensive — ~151k entries)
    print("[INFO] Loading vocabulary (shared across all prompts) …", flush=True)
    vocab_start = time.time()
    vocab = load_vocabulary(model)
    print(f"[INFO] Vocabulary loaded in {time.time() - vocab_start:.1f}s.", flush=True)
    
    # Pre-compute function metadata (avoids rebuilding per prompt)
    fn_metadata = _precompute_function_metadata(functions)
    
    # Build all prompts upfront (allows progress tracking)
    print(f"[INFO] Building prompts for {len(prompts)} queries …", flush=True)
    prompt_data = []
    for prompt_obj in prompts:
        prompt_text = _build_prompt_optimized(prompt_obj.prompt, functions, fn_metadata)
        prompt_ids = _encode_prompt(model, prompt_text)
        if prompt_ids is not None:
            prompt_data.append((prompt_obj, prompt_text, prompt_ids))
        else:
            print(f"[WARNING] Skipping prompt: '{prompt_obj.prompt}'", file=sys.stderr)
    
    # Process each prompt with shared resources
    results: List[FunctionCall] = []
    total = len(prompt_data)
    
    for i, (prompt_obj, prompt_text, prompt_ids) in enumerate(prompt_data):
        print(
            f"[INFO] Processing {i + 1}/{total}: '{prompt_obj.prompt}'",
            flush=True,
        )
        
        fc = _select_function_optimized(
            model=model,
            prompt_ids=prompt_ids,
            prompt_text=prompt_text,
            vocab=vocab,
            fn_metadata=fn_metadata,
            functions=functions,
        )
        
        if fc is not None:
            results.append(fc)
            print(f"         → {fc.name}({fc.parameters})")
        else:
            print(
                f"[WARNING] Could not resolve: '{prompt_obj.prompt}'",
                file=sys.stderr,
            )
    
    return results


def _process_sequential(
    model: Any,
    prompts: List[Any],
    functions: List[Any],
) -> List[FunctionCall]:
    """Process prompts sequentially (original path, slightly optimized)."""
    from src.function_selector import select_function
    
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
    
    return results


# ============================================================================
# Helper functions
# ============================================================================

def _precompute_function_metadata(functions: List[Any]) -> dict[str, Any]:
    """Pre-compute function metadata for faster lookups.
    
    Returns dict with:
    - names: List of function names
    - params: dict mapping fn_name → {param_name: type}
    - descriptions: dict mapping fn_name → description
    - name_set: frozenset for O(1) membership testing
    """
    metadata = {
        'names': [],
        'params': {},
        'descriptions': {},
        'name_set': set(),
    }
    
    for fn in functions:
        metadata['names'].append(fn.name)
        metadata['name_set'].add(fn.name)
        metadata['params'][fn.name] = {
            pname: pdef.type for pname, pdef in fn.parameters.items()
        }
        metadata['descriptions'][fn.name] = fn.description
    
    return metadata


def _build_prompt_optimized(
    user_query: str,
    functions: List[Any],
    fn_metadata: dict[str, Any],
) -> str:
    """Build prompt using pre-computed metadata (avoids repeated string building)."""
    lines: List[str] = [
        "You are a function-calling assistant. "
        "Given a user request and a list of available functions, "
        "select the best function and extract the required arguments.\n",
        "Available functions:",
    ]
    
    # Use pre-computed parameter strings
    for fn in functions:
        params_desc = ", ".join(
            f"{pname}: {ptype}"
            for pname, ptype in fn_metadata['params'][fn.name].items()
        )
        lines.append(f"  - {fn.name}({params_desc}): {fn.description}")
    
    lines.extend([
        f"\nUser request: {user_query}",
        "\nRespond with ONLY a JSON object in this exact format:\n"
        '{"function_name": "<name>", "arguments": {<args>}}',
        "\n{",
    ])
    
    return "\n".join(lines)


def _encode_prompt(model: Any, prompt_text: str) -> Optional[List[int]]:
    """Encode a prompt text to token IDs.
    
    Returns flat list of ints, or None on failure.
    """
    try:
        encoded = model.encode(prompt_text)
        if hasattr(encoded, 'tolist'):
            ids_raw = encoded.tolist()
        else:
            ids_raw = list(encoded)
        if ids_raw and isinstance(ids_raw[0], list):
            ids_raw = ids_raw[0]
        return [int(x) for x in ids_raw]
    except Exception as e:
        print(f"[ERROR] Encoding failed: {e}", file=sys.stderr)
        return None


def _select_function_optimized(
    model: Any,
    prompt_ids: List[int],
    prompt_text: str,
    vocab: dict[int, str],
    fn_metadata: dict[str, Any],
    functions: List[Any],
) -> Optional[FunctionCall]:
    """Select function using pre-loaded resources.
    
    Uses shared vocabulary and pre-computed metadata to avoid
    redundant work across multiple prompts.
    """
    from src.constrained_decoder import (
        JSONSchemaStateMachine,
        generate_constrained,
    )
    from src.function_selector import _coerce_value
    from src.models import FunctionCall
    
    # Build state machine
    sm = JSONSchemaStateMachine(
        allowed_function_names=fn_metadata['names'],
        function_parameters=fn_metadata['params'],
    )
    sm.advance("{")  # Prompt ends with '{'
    
    # Run constrained generation
    generated, result = generate_constrained(
        model=model,
        prompt_ids=prompt_ids,
        vocab=vocab,  # Shared vocabulary!
        sm=sm,
        max_tokens=512,
    )
    
    if result is None:
        return None
    
    chosen_fn_name = result["function_name"]
    raw_args = result["arguments"]
    
    # Build function definition map (could be cached, but it's cheap)
    fn_def_map = {fn.name: fn for fn in functions}
    fn_def = fn_def_map.get(chosen_fn_name)
    
    coerced_args: dict[str, Any] = {}
    if fn_def is not None:
        for pname, pdef in fn_def.parameters.items():
            if pname in raw_args:
                coerced_args[pname] = _coerce_value(raw_args[pname], pdef.type)
            else:
                print(
                    f"[WARNING] Missing argument '{pname}' for "
                    f"'{chosen_fn_name}'",
                    file=sys.stderr,
                )
    else:
        coerced_args = raw_args
    
    try:
        return FunctionCall(
            prompt=prompt_text.split("User request: ")[-1].split("\n")[0] 
                   if "User request: " in prompt_text else prompt_text,
            name=chosen_fn_name,
            parameters=coerced_args,
        )
    except Exception as e:
        print(f"[ERROR] Could not build FunctionCall: {e}", file=sys.stderr)
        return None


if __name__ == "__main__":
    sys.exit(main())
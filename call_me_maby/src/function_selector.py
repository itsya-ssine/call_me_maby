"""Function selector: uses the LLM to pick the right function and extract args.

OPTIMIZED VERSION: Supports batch processing with shared vocabulary,
pre-computed metadata, and efficient prompt building.
"""

import sys
from typing import Any, Dict, List, Optional

from src.constrained_decoder import (
    JSONSchemaStateMachine,
    generate_constrained,
    load_vocabulary,
)
from src.models import FunctionCall, FunctionDefinition


# Cache for prompt templates (keyed by frozenset of function names)
_PROMPT_TEMPLATE_CACHE: Dict[int, str] = {}


def _build_prompt(
    user_query: str,
    functions: List[FunctionDefinition],
) -> str:
    """Build the prompt sent to the LLM.
    
    OPTIMIZED: Caches the function description part since it's 
    identical for all queries with the same function set.

    Args:
        user_query: The natural language request.
        functions: Available function definitions.

    Returns:
        Full prompt string.
    """
    # Generate cache key from function names (order-independent)
    fn_key = hash(frozenset(fn.name for fn in functions))
    
    if fn_key not in _PROMPT_TEMPLATE_CACHE:
        # Build template once
        lines: List[str] = [
            "You are a function-calling assistant. "
            "Given a user request and a list of available functions, "
            "select the best function and extract the required arguments.\n",
            "Available functions:",
        ]
        for fn in functions:
            params_desc = ", ".join(
                f"{pname}: {pdef.type}"
                for pname, pdef in fn.parameters.items()
            )
            lines.append(f"  - {fn.name}({params_desc}): {fn.description}")
        
        lines.append("{query_placeholder}")
        _PROMPT_TEMPLATE_CACHE[fn_key] = "\n".join(lines)
    
    # Insert user query into template
    template = _PROMPT_TEMPLATE_CACHE[fn_key]
    query_section = (
        f"\nUser request: {user_query}\n\n"
        "Respond with ONLY a JSON object in this exact format:\n"
        '{"function_name": "<name>", "arguments": {<args>}}\n'
        "{"
    )
    
    return template.replace("{query_placeholder}", query_section)


def _coerce_value(value: Any, param_type: str) -> Any:
    """Coerce a parsed argument value to the declared parameter type.
    
    OPTIMIZED: Uses early returns and avoids redundant conversions.

    Args:
        value: The raw parsed value.
        param_type: JSON Schema type string (``number``, ``string``, etc.).

    Returns:
        Coerced value.
    """
    # Fast path: value is already correct type
    if param_type == "string" and isinstance(value, str):
        return value
    if param_type == "boolean" and isinstance(value, bool):
        return value
    if param_type == "number" and isinstance(value, (int, float)):
        return float(value)
    if param_type == "integer" and isinstance(value, int):
        return value
    
    # Conversion path
    try:
        if param_type == "number":
            return float(value)
        if param_type == "integer":
            return int(float(value))
        if param_type == "boolean":
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            return bool(value)
        if param_type == "string":
            return str(value)
    except (ValueError, TypeError):
        pass
    return value


def select_function(
    model: Any,
    user_query: str,
    functions: List[FunctionDefinition],
) -> Optional[FunctionCall]:
    """Use the LLM with constrained decoding
    to select a function and extract args.
    
    OPTIMIZED: Uses cached prompt templates and efficient type coercion.

    Args:
        model: Small_LLM_Model instance.
        user_query: Natural language request from the user.
        functions: List of available function definitions.

    Returns:
        FunctionCall if successful, None on failure.
    """
    if not functions:
        print("[WARNING] No functions available.", file=sys.stderr)
        return None

    # Load vocabulary once per call (caller can cache for batch mode)
    vocab = load_vocabulary(model)

    # Build the prompt (uses cached template)
    prompt_text = _build_prompt(user_query, functions)

    # Encode the prompt
    try:
        encoded = model.encode(prompt_text)
        if hasattr(encoded, 'tolist'):
            ids_raw = encoded.tolist()
        else:
            ids_raw = list(encoded)
        if ids_raw and isinstance(ids_raw[0], list):
            ids_raw = ids_raw[0]
        prompt_ids: List[int] = [int(x) for x in ids_raw]
    except Exception as e:
        print(f"[ERROR] Encoding failed for prompt: {e}", file=sys.stderr)
        return None

    # Build parameter map
    fn_names = [fn.name for fn in functions]
    fn_params: Dict[str, Dict[str, str]] = {
        fn.name: {
            pname: pdef.type for pname, pdef in fn.parameters.items()
        }
        for fn in functions
    }

    # Build state machine
    sm = JSONSchemaStateMachine(
        allowed_function_names=fn_names,
        function_parameters=fn_params,
    )
    sm.advance("{")  # Prompt ends with '{'

    # Run constrained generation
    generated, result = generate_constrained(
        model=model,
        prompt_ids=prompt_ids,
        vocab=vocab,
        sm=sm,
        max_tokens=512,
    )

    if result is None:
        print(
            f"[WARNING] Constrained decoding failed for: '{user_query}'",
            file=sys.stderr,
        )
        return None

    chosen_fn_name: str = result["function_name"]
    raw_args: Dict[str, Any] = result["arguments"]

    # Coerce arguments
    fn_def_map: Dict[str, FunctionDefinition] = {
        fn.name: fn for fn in functions
    }
    fn_def = fn_def_map.get(chosen_fn_name)
    coerced_args: Dict[str, Any] = {}

    if fn_def is not None:
        for pname, pdef in fn_def.parameters.items():
            if pname in raw_args:
                coerced_args[pname] = _coerce_value(raw_args[pname], pdef.type)
            else:
                print(
                    f"[WARNING] Missing argument '{pname}' for function "
                    f"'{chosen_fn_name}' in query: '{user_query}'",
                    file=sys.stderr,
                )
    else:
        coerced_args = raw_args

    try:
        return FunctionCall(
            prompt=user_query,
            name=chosen_fn_name,
            parameters=coerced_args,
        )
    except Exception as e:
        print(f"[ERROR] Could not build FunctionCall: {e}", file=sys.stderr)
        return None


# ============================================================================
# Batch-optimized entry point (for use from main.py with --batch flag)
# ============================================================================

def select_function_batch(
    model: Any,
    user_query: str,
    functions: List[FunctionDefinition],
    vocab: Optional[Dict[int, str]] = None,
    fn_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[FunctionCall]:
    """Batch-optimized version that accepts pre-loaded resources.
    
    Use this when processing multiple prompts with the same function set.
    It avoids reloading the vocabulary and recomputing metadata.

    Args:
        model: Small_LLM_Model instance.
        user_query: Natural language request from the user.
        functions: List of available function definitions.
        vocab: Pre-loaded vocabulary (optional, loaded if not provided).
        fn_metadata: Pre-computed function metadata (optional).

    Returns:
        FunctionCall if successful, None on failure.
    """
    if not functions:
        print("[WARNING] No functions available.", file=sys.stderr)
        return None

    # Use provided vocabulary or load it
    if vocab is None:
        vocab = load_vocabulary(model)
    
    # Use provided metadata or compute it
    if fn_metadata is None:
        fn_names = [fn.name for fn in functions]
        fn_params = {
            fn.name: {
                pname: pdef.type for pname, pdef in fn.parameters.items()
            }
            for fn in functions
        }
    else:
        fn_names = fn_metadata['names']
        fn_params = fn_metadata['params']

    # Build prompt
    prompt_text = _build_prompt(user_query, functions)

    # Encode
    try:
        encoded = model.encode(prompt_text)
        if hasattr(encoded, 'tolist'):
            ids_raw = encoded.tolist()
        else:
            ids_raw = list(encoded)
        if ids_raw and isinstance(ids_raw[0], list):
            ids_raw = ids_raw[0]
        prompt_ids: List[int] = [int(x) for x in ids_raw]
    except Exception as e:
        print(f"[ERROR] Encoding failed for prompt: {e}", file=sys.stderr)
        return None

    # Build state machine
    sm = JSONSchemaStateMachine(
        allowed_function_names=fn_names,
        function_parameters=fn_params,
    )
    sm.advance("{")

    # Run constrained generation
    generated, result = generate_constrained(
        model=model,
        prompt_ids=prompt_ids,
        vocab=vocab,
        sm=sm,
        max_tokens=512,
    )

    if result is None:
        return None

    chosen_fn_name = result["function_name"]
    raw_args = result["arguments"]

    # Coerce arguments
    fn_def_map = {fn.name: fn for fn in functions}
    fn_def = fn_def_map.get(chosen_fn_name)
    coerced_args: Dict[str, Any] = {}

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
            prompt=user_query,
            name=chosen_fn_name,
            parameters=coerced_args,
        )
    except Exception as e:
        print(f"[ERROR] Could not build FunctionCall: {e}", file=sys.stderr)
        return None
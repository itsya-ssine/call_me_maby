"""Function selector: uses the LLM to pick the right function and extract args.

The selection process:
1. Build a structured prompt listing all available functions.
2. Run constrained decoding to guarantee a valid JSON result:
   ``{"function_name": "<name>", "arguments": {<key: value, ...>}}``.
3. Return a FunctionCall object.

The LLM *chooses* the function — we do not heuristically match keywords.
Constrained decoding ensures the output is always parseable and schema-valid.
"""

import sys
from typing import Any, Dict, List, Optional

from src.constrained_decoder import (
    JSONSchemaStateMachine,
    generate_constrained,
    load_vocabulary,
)
from src.models import FunctionCall, FunctionDefinition


def _build_prompt(
    user_query: str,
    functions: List[FunctionDefinition],
) -> str:
    """Build the prompt sent to the LLM.

    The prompt includes:
    - A system instruction.
    - All function names with their descriptions and parameter types.
    - The user query.
    - A partial JSON prefix to prime generation.

    Args:
        user_query: The natural language request.
        functions: Available function definitions.

    Returns:
        Full prompt string.
    """
    lines: List[str] = []
    lines.append(
        "You are a function-calling assistant. "
        "Given a user request and a list of available functions, "
        "select the best function and extract the required arguments.\n"
    )
    lines.append("Available functions:")
    for fn in functions:
        params_desc = ", ".join(
            f"{pname}: {pdef.type}"
            for pname, pdef in fn.parameters.items()
        )
        lines.append(f"  - {fn.name}({params_desc}): {fn.description}")

    lines.append(f"\nUser request: {user_query}")
    lines.append(
        "\nRespond with ONLY a JSON object in this exact format:\n"
        '{"function_name": "<name>", "arguments": {<args>}}'
    )
    # Start the JSON to prime generation
    lines.append("\n{")
    return "\n".join(lines)


def _coerce_value(value: Any, param_type: str) -> Any:
    """Coerce a parsed argument value to the declared parameter type.

    Args:
        value: The raw parsed value.
        param_type: JSON Schema type string (``number``, ``string``, etc.).

    Returns:
        Coerced value.
    """
    try:
        if param_type == "number":
            return float(value)
        if param_type == "integer":
            return int(float(value))
        if param_type == "boolean":
            if isinstance(value, bool):
                return value
            return str(value).lower() in ("true", "1", "yes")
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

    # Load vocabulary once
    vocab = load_vocabulary(model)

    # Build the prompt
    prompt_text = _build_prompt(user_query, functions)

    # Encode the prompt.
    # Small_LLM_Model.encode() returns a 2-D tensor of shape (1, seq_len).
    # We flatten it to a plain List[int] for use as input_ids.
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

    # Build parameter map: fn_name -> {param_name -> type_string}
    fn_names = [fn.name for fn in functions]
    fn_params: Dict[str, Dict[str, str]] = {
        fn.name: {
            pname: pdef.type for pname, pdef in fn.parameters.items()
        }
        for fn in functions
    }

    # Build state machine — it expects to see the full object from '{'
    # The prompt already ends with '\n{', so the SM starts at ST_AWAIT_KEY1.
    sm = JSONSchemaStateMachine(
        allowed_function_names=fn_names,
        function_parameters=fn_params,
    )
    # Manually advance past the opening brace since the prompt ends with '{'
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
        print(
            f"[WARNING] Constrained decoding failed for: '{user_query}'",
            file=sys.stderr,
        )
        return None

    chosen_fn_name: str = result["function_name"]
    raw_args: Dict[str, Any] = result["arguments"]

    # Coerce each argument to its declared type
    fn_def_map: Dict[str, FunctionDefinition] = {
        fn.name: fn
        for fn in functions
    }
    fn_def = fn_def_map.get(chosen_fn_name)
    coerced_args: Dict[str, Any] = {}

    if fn_def is not None:
        for pname, pdef in fn_def.parameters.items():
            if pname in raw_args:
                coerced_args[pname] = _coerce_value(raw_args[pname], pdef.type)
            else:
                # Missing argument — leave empty (will show as absent)
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

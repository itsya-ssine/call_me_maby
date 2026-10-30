"""
Function selection via constrained trie search.

Given a user prompt and a list of candidate FunctionDefinitions, selects
the function whose name the model most prefers, by comparing token
sequences directly rather than generating free text and string-matching
afterwards.
"""

from typing import List, Tuple, Dict, Set
import numpy as np
from .models import FunctionDefinition, PromptBuilder, Prompt
from llm_sdk import Small_LLM_Model


def select_function(
    prompt: str,
    functions: List[FunctionDefinition],
    model: Small_LLM_Model
) -> FunctionDefinition:
    """
    Select the function whose name the model most prefers for a given prompt.

    Builds a selection prompt, then performs a token-level trie search over
    the candidate function names: at each generation step, only tokens that
    continue a still-viable candidate name (or complete one) are allowed,
    and the model's top choice among them prunes the candidate set. A
    completed candidate is only declared the winner once its terminator
    token actually wins against the other active candidates' next tokens,
    so a short name only wins over a longer, overlapping name if the model
    genuinely prefers stopping there.

    Args:
        prompt: The user's natural language request.
        functions: List of candidate FunctionDefinition objects to choose
            from. Must be non-empty.
        model: Model used to compute next-token logits for the search.

    Returns:
        The FunctionDefinition whose name best matches the model's
        preference for the given prompt.

    Raises:
        ValueError: If `functions` is empty, if no valid tokens are found
            at some search step, if no candidate matches the tokens
            generated so far, or if the model fails to provide logits.
    """
    if not functions:
        raise ValueError(
            "Cannot select from an empty list of functions. "
            "Please provide at least one FunctionDefinition."
        )

    try:
        builder = PromptBuilder(functions=functions)
        base_prompt = builder.build_selection(Prompt(prompt=prompt))
    except Exception as e:
        raise ValueError(
            f"Failed to build selection prompt for input '{prompt}': {str(e)}"
        ) from e

    try:
        base_ids = model.encode(base_prompt)[0].tolist()
    except Exception as e:
        raise ValueError(
            f"Failed to encode base prompt '{base_prompt}': {str(e)}"
        ) from e

    TERMINATOR = "\n"

    candidates: List[Tuple[FunctionDefinition, List[int]]] = []
    for fn in functions:
        try:
            full_encoding = model.encode(base_prompt + fn.name)[0].tolist()
            name_tokens = full_encoding[len(base_ids):]
            candidates.append((fn, name_tokens))
        except Exception as e:
            raise ValueError(
                f"Failed to encode function name '{fn.name}' for prompt "
                f"'{prompt}': {str(e)}"
            ) from e

    input_ids = list(base_ids)
    step = 0

    while len(candidates) > 1:
        try:
            logits = model.get_logits_from_input_ids(input_ids)
        except Exception as e:
            raise ValueError(
                f"Failed to get logits from model for prompt '{prompt}' "
                f"at step {step}: {str(e)}"
            ) from e

        valid_token_ids: Set[int] = set()
        # map token_id -> list of fncs that would be "complete"
        # if this token is a terminator
        completed_by_token: Dict[int, FunctionDefinition] = {}

        for fn, tokens in candidates:
            if step < len(tokens):
                valid_token_ids.add(tokens[step])
            else:
                # This candidate's name has been fully generated already.
                # Figure out what token would follow it (the terminator),
                # and treat THAT as its valid "vote" at this step.
                try:
                    term_full = model.encode(
                        base_prompt + fn.name + TERMINATOR
                    )[0].tolist()
                    term_token = term_full[len(base_ids) + len(tokens)]
                    valid_token_ids.add(term_token)
                    completed_by_token[term_token] = fn
                except Exception as e:
                    raise ValueError(
                        "Failed to encode terminator for function "
                        f"'{fn.name}' at step {step}: {str(e)}"
                    ) from e

        if not valid_token_ids:
            raise ValueError(
                f"No valid tokens found at step {step} for prompt: "
                f"'{prompt}'. "
                f"This may indicate that the model's tokenizer is "
                f"incompatible with the function names or prompt format."
            )

        try:
            masked_logits = np.full_like(logits, float('-inf'))
            valid_indices = list(valid_token_ids)
            masked_logits[valid_indices] = np.array(logits)[valid_indices]
            best_token = int(np.argmax(masked_logits))
        except Exception as e:
            raise ValueError(
                f"Failed to process logits at step {step} for prompt "
                f"'{prompt}': {str(e)}"
            ) from e

        # If the winning token is a terminator for a completed candidate,
        # that candidate wins outright.
        if best_token in completed_by_token:
            return completed_by_token[best_token]

        # Otherwise, keep only candidates that still match at this step.
        candidates = [
            (fn, tokens)
            for fn, tokens in candidates
            if step < len(tokens) and tokens[step] == best_token
        ]

        if not candidates:
            raise ValueError(
                f"No matching function found for prompt: '{prompt}'. "
                f"Search failed at step {step}. This may indicate that the "
                f"model's top token choice was invalid for all remaining "
                f"candidates."
            )

        input_ids.append(best_token)
        step += 1

    if not candidates:
        raise ValueError(
            f"No valid candidates remaining for prompt: '{prompt}'. "
            f"This should not happen and indicates an internal error."
        )

    return candidates[0][0]

"""Constrained decoding engine for guaranteed valid JSON output.

This module implements token-by-token constrained generation that forces
the LLM to produce valid JSON matching a given schema. At every decoding
step, only tokens that could continue a valid JSON document are allowed;
all others are masked to -inf before the argmax / sampling step.
"""

import copy
import json
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

# We lazily import llm_sdk so the module can be imported without it installed.
try:
    from llm_sdk import Small_LLM_Model
except ImportError:
    Small_LLM_Model = None


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

def load_vocabulary(model: Any) -> Dict[int, str]:
    """Load the token-id -> string mapping from the model's vocabulary file.

    Tries ``get_path_to_tokenizer_file()`` first (``tokenizer.json`` contains
    the full BPE merge vocabulary with proper byte-level representations).
    Falls back to ``get_path_to_vocab_file()`` (``vocab.json``) if unavailable.

    Args:
        model: An instance of Small_LLM_Model.

    Returns:
        Dictionary mapping token IDs to their string representations.
    """
    # Prefer tokenizer.json — it has the complete token -> id mapping including
    # special tokens and byte-level representations used by Qwen's tokenizer.
    tokenizer_path: Optional[str] = None
    try:
        tokenizer_path = model.get_path_to_tokenizer_file()
    except Exception:
        pass

    if tokenizer_path:
        vocab = _load_vocab_from_tokenizer_json(tokenizer_path)
        if vocab:
            return vocab

    # Fallback: plain vocab.json  {"token_string": token_id, ...}
    try:
        vocab_path: str = model.get_path_to_vocab_file()
    except Exception as e:
        print(f"[ERROR] Could not get vocabulary file path: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            raw: Any = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[ERROR] Could not load vocabulary: {e}", file=sys.stderr)
        sys.exit(1)

    vocab_result: Dict[int, str] = {}
    if isinstance(raw, dict):
        for token_str, token_id in raw.items():
            if isinstance(token_id, int):
                vocab_result[token_id] = token_str
    elif isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                token_str, token_id = entry
                if isinstance(token_id, int):
                    vocab_result[token_id] = token_str
    return vocab_result


def _load_vocab_from_tokenizer_json(path: str) -> Dict[int, str]:
    """Parse a HuggingFace ``tokenizer.json`` into a token-id -> string dict.

    The ``tokenizer.json`` format stores the vocabulary under
    ``model.vocab`` as ``{"token_string": id, ...}``.

    Args:
        path: Path to tokenizer.json.

    Returns:
        Token-id to string mapping, or empty dict on failure.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data: Any = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    vocab: Dict[int, str] = {}

    # Primary location: model.vocab
    model_section = data.get("model", {})
    raw_vocab = model_section.get("vocab", {})
    if isinstance(raw_vocab, dict):
        for token_str, token_id in raw_vocab.items():
            if isinstance(token_id, int):
                vocab[token_id] = token_str

    # Also pull added_tokens (special tokens) which may not appear in model.vocab
    for entry in data.get("added_tokens", []):
        tid = entry.get("id")
        content = entry.get("content", "")
        if isinstance(tid, int) and content:
            vocab.setdefault(tid, content)

    return vocab


# ---------------------------------------------------------------------------
# JSON Schema state machine
# ---------------------------------------------------------------------------

class JSONSchemaStateMachine:
    """Tracks parsing state for a JSON object matching a known schema.

    The state machine enforces:
    - The top-level structure is ``{"function_name": "...", "arguments": {...}}``
    - ``function_name`` must be one of the allowed names.
    - ``arguments`` is an object whose keys and value types come from the
      selected function's parameter definitions.

    The machine is advanced character-by-character (via ``advance``).

    Attributes:
        STATES: Named integer constants for each parser state.
    """

    # Parser states
    ST_START = 0                  # Expect opening '{'
    ST_AWAIT_KEY1 = 1             # Inside top obj, expect first key
    ST_IN_KEY1 = 2                # Reading top-level key chars
    ST_AFTER_KEY1 = 3             # Expect ':' after first key
    ST_AWAIT_VAL1 = 4             # Expect value for first key
    ST_IN_FNAME = 5               # Reading function_name string value
    ST_AFTER_FNAME = 6            # Expect ',' or end after function_name
    ST_AWAIT_KEY2 = 7             # Expect "arguments" key
    ST_IN_KEY2 = 8                # Reading second key chars
    ST_AFTER_KEY2 = 9             # Expect ':' after second key
    ST_AWAIT_ARGS_OBJ = 10        # Expect '{' for arguments object
    ST_AWAIT_ARG_KEY = 11         # Inside args, expect key or '}'
    ST_IN_ARG_KEY = 12            # Reading argument key chars
    ST_AFTER_ARG_KEY = 13         # Expect ':' after arg key
    ST_AWAIT_ARG_VAL = 14         # Expect argument value
    ST_IN_ARG_STR = 15            # Reading string argument value
    ST_IN_ARG_NUM = 16            # Reading numeric argument value
    ST_IN_ARG_BOOL_NULL = 17      # Reading boolean/null literal
    ST_AFTER_ARG_VAL = 18         # Expect ',' or '}' after arg value
    ST_AFTER_ARGS_OBJ = 19        # Expect ',' or '}' after args obj
    ST_DONE = 20                  # Fully formed JSON — stop generation
    ST_ERROR = 99                 # Unrecoverable parse error

    def __init__(
        self,
        allowed_function_names: List[str],
        function_parameters: Dict[str, Dict[str, str]],
    ) -> None:
        """Initialise the state machine.

        Args:
            allowed_function_names: Names the function_name field may take.
            function_parameters: Maps each function name to its parameter
                definitions, e.g. ``{"fn_add": {"a": "number", "b": "number"}}``.
        """
        self.allowed_function_names = allowed_function_names
        self.function_parameters = function_parameters

        self.state = self.ST_START
        self.buffer = ""           # chars accumulated in the current field
        self.chosen_function: Optional[str] = None
        self.current_arg_key: Optional[str] = None
        self.collected_args: Dict[str, Any] = {}
        self._arg_str_escape = False  # track backslash in string values
        self._bool_null_target = ""   # expected literal (true/false/null)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_result(self) -> Optional[Dict[str, Any]]:
        """Return the parsed result if complete, else None.

        Returns:
            Dict with keys ``function_name`` and ``arguments``, or None.
        """
        if self.state == self.ST_DONE and self.chosen_function is not None:
            return {
                "function_name": self.chosen_function,
                "arguments": self.collected_args,
            }
        return None

    def is_done(self) -> bool:
        """Return True when the JSON object is fully parsed."""
        return self.state == self.ST_DONE

    def is_error(self) -> bool:
        """Return True when the parser is in an unrecoverable error state."""
        return self.state == self.ST_ERROR

    def get_allowed_next_chars(self) -> Set[str]:
        """Return the set of characters that are valid continuations.

        Returns:
            Set of single-character strings that may appear next.
        """
        s = self.state
        ws = set(" \t\n\r")

        if s == self.ST_START:
            return ws | {"{"}

        if s == self.ST_AWAIT_KEY1:
            return ws | {'"'}

        if s == self.ST_IN_KEY1:
            # We only accept the two valid top-level keys
            completions = self._completions_for_buffer(
                self.buffer, ["function_name", "arguments"]
            )
            if not completions:
                return set()
            next_chars: Set[str] = set()
            for c in completions:
                next_chars.add(c[len(self.buffer)] if len(c) > len(self.buffer) else '"')
            # Always allow closing quote when buffer exactly matches a key
            if self.buffer in ["function_name", "arguments"]:
                next_chars.add('"')
            return next_chars

        if s == self.ST_AFTER_KEY1:
            return ws | {":"}

        if s == self.ST_AWAIT_VAL1:
            return ws | {'"'}

        if s == self.ST_IN_FNAME:
            completions = self._completions_for_buffer(
                self.buffer, self.allowed_function_names
            )
            next_chars_fn: Set[str] = set()
            for c in completions:
                if len(c) > len(self.buffer):
                    next_chars_fn.add(c[len(self.buffer)])
            if self.buffer in self.allowed_function_names:
                next_chars_fn.add('"')
            return next_chars_fn

        if s == self.ST_AFTER_FNAME:
            return ws | {","}

        if s == self.ST_AWAIT_KEY2:
            return ws | {'"'}

        if s == self.ST_IN_KEY2:
            completions = self._completions_for_buffer(
                self.buffer, ["arguments"]
            )
            next_chars_k2: Set[str] = set()
            for c in completions:
                if len(c) > len(self.buffer):
                    next_chars_k2.add(c[len(self.buffer)])
            if self.buffer == "arguments":
                next_chars_k2.add('"')
            return next_chars_k2

        if s == self.ST_AFTER_KEY2:
            return ws | {":"}

        if s == self.ST_AWAIT_ARGS_OBJ:
            return ws | {"{"}

        if s == self.ST_AWAIT_ARG_KEY:
            if self.chosen_function is None:
                return {"}"}
            remaining = self._remaining_arg_keys()
            if not remaining:
                return ws | {"}"}
            return ws | {'"'}

        if s == self.ST_IN_ARG_KEY:
            if self.chosen_function is None:
                return {'"'}
            param_names = list(
                self.function_parameters.get(self.chosen_function, {}).keys()
            )
            remaining = [k for k in param_names if k not in self.collected_args]
            completions = self._completions_for_buffer(self.buffer, remaining)
            next_chars_ak: Set[str] = set()
            for c in completions:
                if len(c) > len(self.buffer):
                    next_chars_ak.add(c[len(self.buffer)])
            if self.buffer in remaining:
                next_chars_ak.add('"')
            return next_chars_ak

        if s == self.ST_AFTER_ARG_KEY:
            return ws | {":"}

        if s == self.ST_AWAIT_ARG_VAL:
            param_type = self._current_param_type()
            if param_type == "string":
                return ws | {'"'}
            if param_type in ("number", "integer"):
                return ws | set("0123456789-")
            if param_type == "boolean":
                return ws | {"t", "f"}
            if param_type == "null":
                return ws | {"n"}
            # unknown — allow all reasonable starters
            return ws | {'"'} | set("0123456789-") | {"t", "f", "n", "{", "["}

        if s == self.ST_IN_ARG_STR:
            if self._arg_str_escape:
                return set('"\\nrtbf/')
            # allow any printable except unescaped control chars; closing quote allowed
            allowed_str: Set[str] = set()
            for code in range(32, 127):
                allowed_str.add(chr(code))
            allowed_str.add('"')
            return allowed_str

        if s == self.ST_IN_ARG_NUM:
            return set("0123456789.eE+-") | {",", "}", " ", "\n", "\t"}

        if s == self.ST_IN_ARG_BOOL_NULL:
            remaining_lit = self._bool_null_target[len(self.buffer):]
            if not remaining_lit:
                return {",", "}", " ", "\n", "\t"}
            return {remaining_lit[0]}

        if s == self.ST_AFTER_ARG_VAL:
            remaining = self._remaining_arg_keys()
            if remaining:
                return ws | {","}
            return ws | {",", "}"}

        if s == self.ST_AFTER_ARGS_OBJ:
            return ws | {"}"}

        if s == self.ST_DONE:
            return set()

        return set()

    def advance(self, char: str) -> None:
        """Advance the state machine by one character.

        Args:
            char: The next character produced by the decoder.
        """
        s = self.state
        ws = " \t\n\r"

        if s == self.ST_START:
            if char in ws:
                pass
            elif char == "{":
                self.state = self.ST_AWAIT_KEY1
            else:
                self.state = self.ST_ERROR

        elif s == self.ST_AWAIT_KEY1:
            if char in ws:
                pass
            elif char == '"':
                self.buffer = ""
                self.state = self.ST_IN_KEY1
            else:
                self.state = self.ST_ERROR

        elif s == self.ST_IN_KEY1:
            if char == '"':
                key = self.buffer
                self.buffer = ""
                if key == "function_name":
                    self.state = self.ST_AFTER_KEY1
                    self._pending_key = "function_name"
                elif key == "arguments":
                    self.state = self.ST_AFTER_KEY2
                    self._pending_key = "arguments"
                else:
                    self.state = self.ST_ERROR
            else:
                self.buffer += char

        elif s == self.ST_AFTER_KEY1:
            if char in ws:
                pass
            elif char == ":":
                self.state = self.ST_AWAIT_VAL1
            else:
                self.state = self.ST_ERROR

        elif s == self.ST_AWAIT_VAL1:
            if char in ws:
                pass
            elif char == '"':
                self.buffer = ""
                self.state = self.ST_IN_FNAME
            else:
                self.state = self.ST_ERROR

        elif s == self.ST_IN_FNAME:
            if char == '"':
                fn_name = self.buffer
                self.buffer = ""
                if fn_name in self.allowed_function_names:
                    self.chosen_function = fn_name
                    self.state = self.ST_AFTER_FNAME
                else:
                    self.state = self.ST_ERROR
            else:
                self.buffer += char

        elif s == self.ST_AFTER_FNAME:
            if char in ws:
                pass
            elif char == ",":
                self.state = self.ST_AWAIT_KEY2
            else:
                self.state = self.ST_ERROR

        elif s == self.ST_AWAIT_KEY2:
            if char in ws:
                pass
            elif char == '"':
                self.buffer = ""
                self.state = self.ST_IN_KEY2
            else:
                self.state = self.ST_ERROR

        elif s == self.ST_IN_KEY2:
            if char == '"':
                if self.buffer == "arguments":
                    self.buffer = ""
                    self.state = self.ST_AFTER_KEY2
                else:
                    self.state = self.ST_ERROR
            else:
                self.buffer += char

        elif s == self.ST_AFTER_KEY2:
            if char in ws:
                pass
            elif char == ":":
                self.state = self.ST_AWAIT_ARGS_OBJ
            else:
                self.state = self.ST_ERROR

        elif s == self.ST_AWAIT_ARGS_OBJ:
            if char in ws:
                pass
            elif char == "{":
                self.state = self.ST_AWAIT_ARG_KEY
            else:
                self.state = self.ST_ERROR

        elif s == self.ST_AWAIT_ARG_KEY:
            if char in ws:
                pass
            elif char == '"':
                self.buffer = ""
                self.state = self.ST_IN_ARG_KEY
            elif char == "}":
                self.state = self.ST_AFTER_ARGS_OBJ
            else:
                self.state = self.ST_ERROR

        elif s == self.ST_IN_ARG_KEY:
            if char == '"':
                self.current_arg_key = self.buffer
                self.buffer = ""
                self.state = self.ST_AFTER_ARG_KEY
            else:
                self.buffer += char

        elif s == self.ST_AFTER_ARG_KEY:
            if char in ws:
                pass
            elif char == ":":
                self.state = self.ST_AWAIT_ARG_VAL
            else:
                self.state = self.ST_ERROR

        elif s == self.ST_AWAIT_ARG_VAL:
            if char in ws:
                pass
            else:
                param_type = self._current_param_type()
                if param_type == "string":
                    if char == '"':
                        self.buffer = ""
                        self._arg_str_escape = False
                        self.state = self.ST_IN_ARG_STR
                    else:
                        self.state = self.ST_ERROR
                elif param_type in ("number", "integer"):
                    if char in "0123456789-":
                        self.buffer = char
                        self.state = self.ST_IN_ARG_NUM
                    else:
                        self.state = self.ST_ERROR
                elif param_type == "boolean":
                    if char == "t":
                        self.buffer = "t"
                        self._bool_null_target = "true"
                        self.state = self.ST_IN_ARG_BOOL_NULL
                    elif char == "f":
                        self.buffer = "f"
                        self._bool_null_target = "false"
                        self.state = self.ST_IN_ARG_BOOL_NULL
                    else:
                        self.state = self.ST_ERROR
                elif param_type == "null":
                    if char == "n":
                        self.buffer = "n"
                        self._bool_null_target = "null"
                        self.state = self.ST_IN_ARG_BOOL_NULL
                    else:
                        self.state = self.ST_ERROR
                else:
                    # Fallback: treat as string
                    if char == '"':
                        self.buffer = ""
                        self._arg_str_escape = False
                        self.state = self.ST_IN_ARG_STR
                    elif char in "0123456789-":
                        self.buffer = char
                        self.state = self.ST_IN_ARG_NUM
                    else:
                        self.state = self.ST_ERROR

        elif s == self.ST_IN_ARG_STR:
            if self._arg_str_escape:
                self._arg_str_escape = False
                if char == '"':
                    self.buffer += '"'
                elif char == "n":
                    self.buffer += "\n"
                elif char == "t":
                    self.buffer += "\t"
                elif char == "r":
                    self.buffer += "\r"
                elif char == "\\":
                    self.buffer += "\\"
                else:
                    self.buffer += char
            elif char == "\\":
                self._arg_str_escape = True
            elif char == '"':
                # End of string value
                self._store_arg_value(self.buffer)
                self.buffer = ""
                self.state = self.ST_AFTER_ARG_VAL
            else:
                self.buffer += char

        elif s == self.ST_IN_ARG_NUM:
            if char in "0123456789.eE+-":
                self.buffer += char
            else:
                # Terminator reached — store value
                try:
                    num_val: Any
                    if "." in self.buffer or "e" in self.buffer or "E" in self.buffer:
                        num_val = float(self.buffer)
                    else:
                        num_val = int(self.buffer)
                    # If schema says number, always use float
                    if self._current_param_type() == "number":
                        num_val = float(num_val)
                    self._store_arg_value(num_val)
                except ValueError:
                    self.state = self.ST_ERROR
                    return
                self.buffer = ""
                if char == ",":
                    self.state = self.ST_AWAIT_ARG_KEY
                elif char == "}":
                    self.state = self.ST_AFTER_ARGS_OBJ
                else:
                    self.state = self.ST_AFTER_ARG_VAL

        elif s == self.ST_IN_ARG_BOOL_NULL:
            self.buffer += char
            if self.buffer == self._bool_null_target:
                val: Any
                if self._bool_null_target == "true":
                    val = True
                elif self._bool_null_target == "false":
                    val = False
                else:
                    val = None
                self._store_arg_value(val)
                self.buffer = ""
                self.state = self.ST_AFTER_ARG_VAL
            elif not self._bool_null_target.startswith(self.buffer):
                self.state = self.ST_ERROR

        elif s == self.ST_AFTER_ARG_VAL:
            if char in ws:
                pass
            elif char == ",":
                self.state = self.ST_AWAIT_ARG_KEY
            elif char == "}":
                self.state = self.ST_AFTER_ARGS_OBJ
            else:
                self.state = self.ST_ERROR

        elif s == self.ST_AFTER_ARGS_OBJ:
            if char in ws:
                pass
            elif char == "}":
                self.state = self.ST_DONE
            else:
                self.state = self.ST_ERROR

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _completions_for_buffer(self, buf: str, candidates: List[str]) -> List[str]:
        """Return candidates whose prefix matches buf."""
        return [c for c in candidates if c.startswith(buf)]

    def _remaining_arg_keys(self) -> List[str]:
        """Return param names not yet collected for the chosen function."""
        if self.chosen_function is None:
            return []
        params = self.function_parameters.get(self.chosen_function, {})
        return [k for k in params if k not in self.collected_args]

    def _current_param_type(self) -> str:
        """Return the type of the current argument being parsed."""
        if self.chosen_function is None or self.current_arg_key is None:
            return "string"
        params = self.function_parameters.get(self.chosen_function, {})
        return params.get(self.current_arg_key, "string")

    def _store_arg_value(self, value: Any) -> None:
        """Store a parsed argument value and reset current_arg_key."""
        if self.current_arg_key is not None:
            self.collected_args[self.current_arg_key] = value
            self.current_arg_key = None


# ---------------------------------------------------------------------------
# Constrained generation
# ---------------------------------------------------------------------------

def _get_valid_token_ids(
    sm: JSONSchemaStateMachine,
    vocab: Dict[int, str],
) -> List[int]:
    """Return token IDs whose string starts with an allowed next character.

    A token is valid if *every* character in its string is reachable
    step-by-step from the current state machine state.  For efficiency we
    do a single-character look-ahead: a token is a candidate if its first
    character is in the allowed set, then we tentatively advance the SM on
    the full token string and accept only if the SM doesn't error out.

    Args:
        sm: Current state machine instance.
        vocab: Token-id to string mapping.

    Returns:
        List of valid token IDs.
    """
    allowed_first = sm.get_allowed_next_chars()
    valid: List[int] = []

    for token_id, token_str in vocab.items():
        if not token_str:
            continue
        # Normalise leading-space marker that some tokenisers use
        display = token_str.replace("Ġ", " ").replace("▁", " ")
        if not display:
            continue
        first_char = display[0]
        if first_char not in allowed_first:
            continue
        # Deep check: advance a clone of the SM over all chars of the token
        sm_clone = copy.deepcopy(sm)
        ok = True
        for ch in display:
            sm_clone.advance(ch)
            if sm_clone.is_error():
                ok = False
                break
        if ok:
            valid.append(token_id)

    return valid


def generate_constrained(
    model: Any,
    prompt_ids: List[int],
    vocab: Dict[int, str],
    sm: JSONSchemaStateMachine,
    max_tokens: int = 256,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Run constrained token-by-token generation.

    At each step:
    1. Feed current input_ids to the model to get logits.
    2. Mask all tokens not in the valid set to -inf.
    3. Argmax to pick the best valid token.
    4. Advance the state machine with the token's string.
    5. Repeat until SM is done or max_tokens reached.

    Args:
        model: Small_LLM_Model instance.
        prompt_ids: Encoded prompt token IDs.
        vocab: Token-id to string mapping.
        sm: Initialised JSONSchemaStateMachine.
        max_tokens: Maximum tokens to generate.

    Returns:
        Tuple of (generated_text, parsed_result_or_None).
    """
    current_ids: List[int] = list(prompt_ids)
    generated_text = ""

    for _ in range(max_tokens):
        if sm.is_done() or sm.is_error():
            break

        # get_logits_from_input_ids expects a plain list[int] and returns list[float]
        try:
            logits_list: List[float] = model.get_logits_from_input_ids(current_ids)
        except Exception as e:
            print(f"[ERROR] LLM inference failed: {e}", file=sys.stderr)
            break

        vocab_size = len(logits_list)

        # Determine valid token IDs for the current SM state
        valid_ids = _get_valid_token_ids(sm, vocab)

        if not valid_ids:
            # No valid continuation — stop
            break

        # Mask invalid tokens: build a list with -inf everywhere, then copy valid scores
        NEG_INF = float("-inf")
        masked: List[float] = [NEG_INF] * vocab_size
        for tid in valid_ids:
            if 0 <= tid < vocab_size:
                masked[tid] = logits_list[tid]

        # Greedy argmax over the masked logits
        best_id = 0
        best_val = NEG_INF
        for idx, val in enumerate(masked):
            if val > best_val:
                best_val = val
                best_id = idx

        if best_val == NEG_INF:
            # All candidates were out of vocab range — give up
            break

        next_token_id = best_id

        # Decode the chosen token and normalise leading-space markers
        token_str = vocab.get(next_token_id, "")
        display = token_str.replace("\u0120", " ").replace("\u2581", " ")
        # Also handle the literal Ġ / ▁ characters stored as-is in vocab.json
        display = display.replace("Ġ", " ").replace("▁", " ")

        # Advance the state machine character by character
        for ch in display:
            sm.advance(ch)
            if sm.is_error():
                break

        generated_text += display
        current_ids.append(next_token_id)

        if sm.is_done():
            break

    return generated_text, sm.get_result()

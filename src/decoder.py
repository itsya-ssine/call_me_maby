"""Constrained JSON decoding for LLM function-call parameter extraction.

This module drives a language model token by token, masking its logits
at each step so that only tokens which keep the output valid JSON (and
valid for the current parameter's type) can ever be chosen. This avoids
relying on the model to "know" JSON syntax and instead guarantees it by
construction.
"""

import json
from typing import Any, Dict, List, Set, cast

import numpy as np
from pydantic import BaseModel

from llm_sdk import Small_LLM_Model

from .models import FunctionDefinition, Prompt, PromptBuilder, Vocab


class Decoder(BaseModel):
    """Drive the model token by token to produce a valid JSON object.

    This class implements constrained decoding to generate structured
    JSON output from a language model. It uses token-level masking to
    enforce valid syntax, type-specific decoders for different parameter
    types, and repetition detection to prevent infinite loops.

    Attributes:
        model: The language model used for token generation.
        vocabulary: Bidirectional token-to-ID mapping for the model's
            tokenizer.

    Note:
        The model_config allows non-standard types (like
        Small_LLM_Model) to be used as Pydantic fields.
    """

    model: Small_LLM_Model
    vocabulary: Vocab

    #: Parameter types this decoder knows how to generate.
    _SUPPORTED_PARAM_TYPES = frozenset(
        {"integer", "number", "boolean", "null", "string"}
    )

    #: Safety limit on tokens generated for a single numeric parameter.
    _NUMERIC_TOKEN_LIMIT = 20

    #: Safety limit on tokens generated for a single string parameter.
    _STRING_TOKEN_LIMIT = 100

    model_config = {"arbitrary_types_allowed": True}

    def _mask_logits(
        self, logits: List[Any], valid_ids: Set[int]
    ) -> List[float]:
        """Set every logit to -inf except the ones in valid_ids.

        This forces the model to only pick from the allowed tokens,
        effectively constraining the generation to a specific
        vocabulary subset.

        Args:
            logits: Raw logit values from the model for the next token
                position.
            valid_ids: Set of token IDs that the model is allowed to
                generate.

        Returns:
            Masked logits where invalid tokens have -inf probability.

        Raises:
            ValueError: If logits is empty, valid_ids is empty, or
                valid_ids contains an ID outside the range of logits.
        """
        if not logits:
            raise ValueError(
                "Cannot mask logits: the model returned an empty "
                "logits vector, so no next token can be selected."
            )
        if not valid_ids:
            raise ValueError(
                "Cannot mask logits: no valid token ids were supplied, "
                "so no token could ever be selected at this step."
            )

        try:
            arr = np.array(logits, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Cannot mask logits: the model's logits could not be "
                f"converted to a numeric array ({exc})."
            ) from exc

        vocab_size = arr.shape[0]
        out_of_range = {
            token_id
            for token_id in valid_ids
            if token_id < 0 or token_id >= vocab_size
        }
        if out_of_range:
            raise ValueError(
                "Cannot mask logits: token id(s) "
                f"{sorted(out_of_range)} fall outside the valid range "
                f"[0, {vocab_size}) of the model's logits vector."
            )

        valid_id_list = list(valid_ids)
        mask = np.full(vocab_size, float("-inf"), dtype=np.float64)
        mask[valid_id_list] = arr[valid_id_list]
        masked_logits: List[float] = mask.tolist()
        return masked_logits

    def _predict_next_token(
        self, input_ids: List[int], allowed_ids: Set[int]
    ) -> int:
        """Ask the model for next-token logits and pick the best one.

        Centralizes the "get logits, mask, argmax" pattern shared by
        every type-specific decoder so that model failures are reported
        consistently.

        Args:
            input_ids: The token ids generated so far.
            allowed_ids: The set of token ids permitted at this step.

        Returns:
            The token id with the highest logit among allowed_ids.

        Raises:
            RuntimeError: If the model fails to produce logits.
            ValueError: If the logits it returns cannot be masked (see
                _mask_logits).
        """
        try:
            raw_logits = self.model.get_logits_from_input_ids(input_ids)
        except Exception as exc:
            raise RuntimeError(
                "Model failed to produce logits for a sequence of "
                f"{len(input_ids)} token(s): {exc}"
            ) from exc

        masked = self._mask_logits(raw_logits, allowed_ids)
        return masked.index(max(masked))

    def _encode(self, text: str) -> List[int]:
        """Encode text into a flat list of token ids.

        Args:
            text: The text to encode.

        Returns:
            The token ids for the first (and only) sequence in the
            batch returned by the model.

        Raises:
            RuntimeError: If the model fails to encode the text, or
                returns a result that does not match the expected
                batched-tensor shape.
        """
        try:
            encoded = self.model.encode(text)
        except Exception as exc:
            raise RuntimeError(
                f"Model failed to encode text {text}: {exc}"
            ) from exc

        try:
            return cast(List[int], encoded[0].tolist())
        except (IndexError, AttributeError, TypeError) as exc:
            raise RuntimeError(
                "Model.encode() returned an unexpected format for "
                f"text {text!r}; expected a batched tensor exposing "
                f"[0].tolist(), got {type(encoded).__name__}."
            ) from exc

    def _decode(self, token_ids: List[int]) -> str:
        """Decode token ids into text.

        Args:
            token_ids: The token ids to decode.

        Returns:
            The decoded text.

        Raises:
            RuntimeError: If the model fails to decode the token ids.
        """
        try:
            return cast(str, self.model.decode(token_ids))
        except Exception as exc:
            raise RuntimeError(
                f"Model failed to decode {len(token_ids)} token id(s): "
                f"{exc}"
            ) from exc

    def _has_repeating_pattern(
        self, tokens: List[int], max_pattern_len: int = 10
    ) -> bool:
        """Detect if the token sequence ends with a repeating pattern.

        Checks for patterns of length 3-10 where the last N tokens
        exactly match the N tokens before them. This helps identify
        when the model has entered an infinite loop.

        Args:
            tokens: List of generated token IDs to check.
            max_pattern_len: Maximum pattern length to check (default:
                10).

        Returns:
            True if a repeating pattern is detected, False otherwise.

        Note:
            Starts at pattern length 3 to avoid false positives on
            natural repetitions like double digits ('233') or double
            letters ('book').
        """
        for pattern_len in range(3, max_pattern_len + 1):
            if len(tokens) >= pattern_len * 2:
                tail = tokens[-pattern_len:]
                prev = tokens[-pattern_len * 2:-pattern_len]
                if tail == prev:
                    return True
        return False

    def _numeric_allowed_ids(self, allowed_chars: str) -> Set[int]:
        """Collect vocabulary token ids made up only of allowed_chars.

        Args:
            allowed_chars: Characters permitted in a matching token
                (e.g. digits, '-', '.').

        Returns:
            The set of matching token ids.

        Raises:
            ValueError: If no vocabulary tokens match, meaning the
                model would have no way to write a number at all.
        """
        numeric_ids = {
            token_id
            for token_str, token_id in self.vocabulary.token_to_id.items()
            if token_str and all(char in allowed_chars for char in token_str)
        }
        if not numeric_ids:
            raise ValueError(
                "Cannot decode a numeric parameter: the model's "
                "vocabulary contains no tokens made up only of "
                f"characters in {allowed_chars!r}."
            )
        return numeric_ids

    def _decode_integer_param(
        self, input_ids: List[int], sep_tokens: List[int]
    ) -> None:
        """Generate an integer value token by token.

        Constrains generation to only digit and sign tokens ('-',
        '0'-'9'), plus the separator token to signal completion.

        Args:
            input_ids: Mutable list of token IDs being built (modified
                in-place).
            sep_tokens: Token IDs for the separator that follows this
                value (e.g., ',' or '}').

        Raises:
            ValueError: If sep_tokens is empty, or if the vocabulary
                has no numeric tokens to generate with.
            RuntimeError: If the model fails during generation.

        Note:
            Has a safety limit of 20 tokens before forcing the
            separator to prevent infinite generation.
        """
        if not sep_tokens:
            raise ValueError(
                "Cannot decode an integer parameter: sep_tokens is "
                "empty, so there is no separator to close the value."
            )

        numeric_ids = self._numeric_allowed_ids("0123456789-")
        allowed_ids = numeric_ids | {sep_tokens[0]}

        for _ in range(self._NUMERIC_TOKEN_LIMIT):
            token_id = self._predict_next_token(input_ids, allowed_ids)
            input_ids.append(token_id)

            if token_id == sep_tokens[0]:
                input_ids.extend(sep_tokens[1:])
                return

        # Safety: force separator if the token limit is reached.
        input_ids.extend(sep_tokens)

    def _decode_number_param(
        self, input_ids: List[int], sep_tokens: List[int]
    ) -> None:
        """Generate a floating-point number token by token.

        Similar to integer decoding but allows decimal points. If the
        model never generates a '.', appends '.0' to ensure valid JSON
        number format.

        Args:
            input_ids: Mutable list of token IDs being built (modified
                in-place).
            sep_tokens: Token IDs for the separator that follows this
                value.

        Raises:
            ValueError: If sep_tokens is empty, or if the vocabulary
                has no numeric tokens to generate with.
            RuntimeError: If the model fails during generation.

        Note:
            Has a safety limit of 20 tokens. Ensures the output is
            always a valid JSON number by appending '.0' if no decimal
            point appears.
        """
        if not sep_tokens:
            raise ValueError(
                "Cannot decode a number parameter: sep_tokens is "
                "empty, so there is no separator to close the value."
            )

        numeric_ids = self._numeric_allowed_ids("0123456789.-")
        allowed_ids = numeric_ids | {sep_tokens[0]}
        seen_dot = False

        for _ in range(self._NUMERIC_TOKEN_LIMIT):
            token_id = self._predict_next_token(input_ids, allowed_ids)
            input_ids.append(token_id)

            token_text = self.vocabulary.id_to_token.get(token_id, "")
            if "." in token_text:
                seen_dot = True

            if token_id == sep_tokens[0]:
                if not seen_dot:
                    # Replace separator with ".0" then re-append it.
                    input_ids.pop()
                    input_ids.extend(self._encode(".0"))
                    input_ids.extend(sep_tokens)
                else:
                    input_ids.extend(sep_tokens[1:])
                return

        # Safety: append '.0' if no decimal point was seen.
        if not seen_dot:
            input_ids.extend(self._encode(".0"))
        input_ids.extend(sep_tokens)

    def _decode_boolean_param(
        self, input_ids: List[int], sep_tokens: List[int]
    ) -> None:
        """Generate 'true' or 'false' by constraining the first token.

        Forces the model to choose between the first token of 'true'
        or 'false', then automatically completes the rest of the
        chosen word.

        Args:
            input_ids: Mutable list of token IDs being built (modified
                in-place).
            sep_tokens: Token IDs for the separator that follows this
                value.

        Raises:
            ValueError: If sep_tokens, or the encoding of 'true'/
                'false', comes back empty.
            RuntimeError: If the model fails during generation.
        """
        if not sep_tokens:
            raise ValueError(
                "Cannot decode a boolean parameter: sep_tokens is "
                "empty, so there is no separator to close the value."
            )

        true_tokens = self._encode("true")
        false_tokens = self._encode("false")
        if not true_tokens or not false_tokens:
            raise ValueError(
                "Cannot decode a boolean parameter: encoding 'true' "
                "or 'false' produced no tokens."
            )

        token_id = self._predict_next_token(
            input_ids, {true_tokens[0], false_tokens[0]}
        )
        input_ids.append(token_id)

        chosen_tokens = (
            true_tokens if token_id == true_tokens[0] else false_tokens
        )
        input_ids.extend(chosen_tokens[1:])
        input_ids.extend(sep_tokens)

    def _decode_string_value(self, input_ids: List[int]) -> str:
        """Generate a quoted string value token by token.

        Handles string generation with proper quoting and stopping
        conditions. Monitors for repeating patterns to prevent infinite
        loops.

        Args:
            input_ids: Mutable list of token IDs being built (modified
                in-place).

        Returns:
            The decoded string value (without surrounding quotes).

        Raises:
            ValueError: If the model's vocabulary has no standalone
                '"' token, or no tokens usable as string content.
            RuntimeError: If the model fails during generation.

        Note:
            Stops when the model generates a closing quote token,
            detects a repeating pattern, or reaches the 100-token
            safety limit.
        """
        try:
            quote_id = self.vocabulary.token_to_id['"']
        except KeyError as exc:
            raise ValueError(
                'Model vocabulary has no standalone \'"\' token; '
                "cannot constrain string generation for this "
                "tokenizer."
            ) from exc
        input_ids.append(quote_id)  # Force opening quote.

        # Allow tokens without bare quotes, plus tokens starting with
        # '"' (closing quotes).
        allowed_ids = {
            token_id
            for token_str, token_id in self.vocabulary.token_to_id.items()
            if token_str and '"' not in token_str
        } | {
            token_id
            for token_str, token_id in self.vocabulary.token_to_id.items()
            if token_str and token_str.startswith('"')
        }
        if not allowed_ids:
            raise ValueError(
                "Cannot decode a string parameter: the model's "
                "vocabulary has no tokens usable as string content or "
                "as a closing quote."
            )

        generated_tokens: List[int] = []

        while len(generated_tokens) < self._STRING_TOKEN_LIMIT:
            token_id = self._predict_next_token(input_ids, allowed_ids)
            token_str = self._decode([token_id])

            if token_str.startswith('"'):
                input_ids.append(quote_id)
                break

            input_ids.append(token_id)
            generated_tokens.append(token_id)

            if self._has_repeating_pattern(generated_tokens):
                input_ids.append(quote_id)
                break
        else:
            # 100-token safety limit reached.
            input_ids.append(quote_id)

        raw_string = self._decode(generated_tokens)
        try:
            # json.loads handles double-escaped backslashes from JSON
            # encoding.
            return cast(str, json.loads(f'"{raw_string}"'))
        except json.JSONDecodeError:
            # Return the raw string if it isn't valid JSON on its own;
            # this is a best-effort fallback, not a fatal error, since
            # the caller still gets usable (if unescaped) text.
            return raw_string

    def _generate_shared(
        self, prompt: str, function: FunctionDefinition
    ) -> Dict[str, Any]:
        r"""Generate the full JSON object for all function parameters.

        This is the core generation method that:

        1. Builds a few-shot prompt with parameter extraction examples.
        2. Encodes the prompt with an opening brace '{' so the
           tokenizer sees '\\n{' consistently with the training
           examples.
        3. For each parameter, appends the JSON key and delegates to
           the appropriate type-specific decoder.
        4. Decodes the complete token sequence and extracts the JSON
           object.

        Args:
            prompt: The user's input prompt to extract parameters from.
            function: The FunctionDefinition specifying what parameters
                to extract.

        Returns:
            A dictionary containing the extracted parameter names and
            values.

        Raises:
            ValueError: If function.parameters is empty, if a
                parameter has an unsupported type, or if the generated
                text cannot be parsed as JSON.
            RuntimeError: If the model fails at any generation step.
        """
        if not function.parameters:
            raise ValueError(
                f"Cannot generate parameters for function "
                f"{function.name!r}: it declares no parameters."
            )

        unsupported = {
            param_spec.type
            for param_spec in function.parameters.values()
            if param_spec.type not in self._SUPPORTED_PARAM_TYPES
        }
        if unsupported:
            raise ValueError(
                f"Cannot generate parameters for function "
                f"{function.name!r}: unsupported parameter type(s) "
                f"{sorted(unsupported)}. Supported types are "
                f"{sorted(self._SUPPORTED_PARAM_TYPES)}."
            )

        builder = PromptBuilder(functions=[function])
        try:
            built_prompt = builder.build_parameters(
                Prompt(prompt=prompt), function
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to build the parameter-extraction prompt for "
                f"function {function.name!r}: {exc}"
            ) from exc

        input_ids = self._encode(built_prompt + "{\n")

        last_param_name = list(function.parameters.keys())[-1]

        for param_name, param_spec in function.parameters.items():
            key_tokens = self._encode(f'"{param_name}": ')
            input_ids.extend(key_tokens)

            is_last_param = param_name == last_param_name
            sep_tokens = (
                self._encode("\n}")
                if is_last_param
                else self._encode(",\n")
            )

            if param_spec.type == "integer":
                self._decode_integer_param(input_ids, sep_tokens)
            elif param_spec.type == "number":
                self._decode_number_param(input_ids, sep_tokens)
            elif param_spec.type == "boolean":
                self._decode_boolean_param(input_ids, sep_tokens)
            elif param_spec.type == "null":
                input_ids.extend(self._encode("null"))
                input_ids.extend(sep_tokens)
            else:
                self._decode_string_value(input_ids)
                input_ids.extend(sep_tokens)

        full_text = self._decode(input_ids)
        generated_text = full_text[len(built_prompt):]

        if not generated_text.startswith("{"):
            generated_text = "{" + generated_text

        try:
            result = json.JSONDecoder().raw_decode(generated_text)[0]
        except json.JSONDecodeError as exc:
            preview = generated_text[:200]
            raise ValueError(
                f"Failed to parse generated output as JSON for "
                f"function {function.name!r}: {exc}. Generated text "
                f"started with: {preview!r}"
            ) from exc

        if not isinstance(result, dict):
            raise ValueError(
                f"Generated output for function {function.name!r} was "
                f"valid JSON but not a JSON object (got "
                f"{type(result).__name__})."
            )

        return cast(Dict[str, Any], result)

    def generate(
        self, prompt: str, function: FunctionDefinition
    ) -> Dict[str, Any]:
        """Extract all parameter values from a prompt for a function.

        This is the main entry point for parameter extraction. It
        takes a user's natural language prompt and a function
        definition, then returns the extracted parameter values as a
        dictionary.

        Args:
            prompt: The user's natural language request.
            function: The FunctionDefinition specifying expected
                parameters.

        Returns:
            A dictionary mapping parameter names to their extracted
            values.

        Raises:
            ValueError: If function.parameters is empty, a parameter
                has an unsupported type, or the model's output cannot
                be parsed as a JSON object.
            RuntimeError: If the underlying model fails at any
                generation step.

        Example:
            >>> decoder = Decoder(model=model, vocabulary=vocab)
            >>> params = decoder.generate("add 5 and 3", add_function)
            >>> print(params)  # {'x': 5, 'y': 3}
        """
        return self._generate_shared(prompt, function)

"""Data models and helpers for LLM-driven function calling.

This module defines the schema for callable function definitions and
user prompts, provides a JSON parser that validates input files against
those schemas, builds the text prompts sent to the LLM, and exposes a
tokenizer vocabulary loader.
"""

import json
import keyword
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, cast

from pydantic import BaseModel, ValidationError, model_validator

from llm_sdk import Small_LLM_Model


class ValidType(str, Enum):
    """Allowed primitive types for function parameters and returns.

    These represent the basic data types that can be used in function
    signatures within the system. Being a ``str`` subclass, pydantic
    validates a field typed as ``ValidType`` natively: any input value
    that doesn't match one of the members below is rejected with a
    clear, automatically-generated validation error, and no manual
    validator is needed on the models that use it.
    """

    NUMBER = "number"
    STRING = "string"
    BOOLEAN = "boolean"
    INTEGER = "integer"
    NULL = "null"

    def __str__(self) -> str:
        """Return the plain type value (e.g. 'number') rather than
        the default 'ValidType.NUMBER' Enum representation.

        This keeps f-string interpolation (e.g. when building LLM
        prompts or prototype strings) producing exactly the same
        plain-text output as when ``type`` was a bare ``str`` field.

        Returns:
            The underlying string value of this member.
        """
        return str(self.value)


class ParameterSpec(BaseModel):
    """Specification for a single parameter of a function.

    Attributes:
        type: The parameter type. Declaring this field as
            ``ValidType`` (rather than ``str`` plus a manual
            validator) lets pydantic itself enforce membership in the
            allowed set at parse time.
    """

    type: ValidType


class ReturnSpec(BaseModel):
    """Specification for the return type of a function.

    Attributes:
        type: The return type. Declaring this field as ``ValidType``
            (rather than ``str`` plus a manual validator) lets
            pydantic itself enforce membership in the allowed set at
            parse time.
    """

    type: ValidType


class FunctionDefinition(BaseModel):
    """Complete definition of a callable function.

    Contains all information needed to understand and validate a
    function, including its name, description, parameters, and return
    type.

    Attributes:
        name: The function name. Must be a non-empty, valid Python
            identifier that is not a reserved keyword.
        description: Human-readable description of the function's
            purpose. Must be non-empty.
        parameters: Dictionary mapping parameter names to their
            specifications. Each key must be a valid identifier.
        returns: Specification of the function's return type.

    Raises:
        ValueError: If the function name is empty, whitespace-only,
            or not a valid identifier; if the description is empty or
            whitespace-only; or if any parameter name is not a valid
            identifier.
    """

    name: str
    description: str
    parameters: Dict[str, ParameterSpec]
    returns: ReturnSpec

    @model_validator(mode="after")
    def check_name(self) -> "FunctionDefinition":
        """Validate that the function name is a legal identifier.

        Returns:
            The validated FunctionDefinition instance.

        Raises:
            ValueError: If the name is empty, whitespace-only, not a
                valid identifier, or a reserved keyword.
        """
        if not self.name or not self.name.strip():
            raise ValueError(
                "Function name cannot be empty or whitespace-only."
            )
        if not self.name.isidentifier() or keyword.iskeyword(self.name):
            raise ValueError(
                f"'{self.name}' is not a valid function name."
            )
        return self

    @model_validator(mode="after")
    def check_description(self) -> "FunctionDefinition":
        """Validate that the function description is non-empty.

        Returns:
            The validated FunctionDefinition instance.

        Raises:
            ValueError: If the description is empty or whitespace-only.
        """
        if not self.description or not self.description.strip():
            raise ValueError("Function description cannot be empty.")
        return self

    @model_validator(mode="after")
    def check_parameter_names(self) -> "FunctionDefinition":
        """Validate that every parameter name is a legal identifier.

        Returns:
            The validated FunctionDefinition instance.

        Raises:
            ValueError: If any parameter name is not a valid
                identifier or is a reserved keyword.
        """
        for param_name in self.parameters:
            is_keyword = keyword.iskeyword(param_name)
            if not param_name.isidentifier() or is_keyword:
                raise ValueError(
                    f"Invalid parameter name '{param_name}' in "
                    f"function '{self.name}': must be a valid "
                    "identifier."
                )
        return self

    def __str__(self) -> str:
        """Return a human-readable function prototype string.

        Returns:
            A string representation showing the function signature,
            e.g., 'prototype: def add(x: number, y: number) -> number'
        """
        params = ", ".join(
            f"{name}: {spec.type}" for name, spec in self.parameters.items()
        )
        return f"prototype: def {self.name}({params}) -> {self.returns.type}"


class Prompt(BaseModel):
    """A raw user request or prompt.

    Attributes:
        prompt: The text content of the user's prompt. Must be
            non-empty and not whitespace-only.

    Raises:
        ValueError: If the prompt text is empty or whitespace-only.
    """

    prompt: str

    @model_validator(mode="after")
    def check_prompt(self) -> "Prompt":
        """Validate that the prompt text is non-empty.

        Returns:
            The validated Prompt instance.

        Raises:
            ValueError: If the prompt is empty or whitespace-only.
        """
        if not self.prompt or not self.prompt.strip():
            raise ValueError(
                "Prompt text cannot be empty or whitespace-only."
            )
        return self

    def __str__(self) -> str:
        """Return the prompt string.

        Returns:
            The raw prompt text.
        """
        return self.prompt


class InputParser(BaseModel):
    """Reads a JSON file and returns validated model objects.

    This parser handles loading and validating JSON data against
    Pydantic models, providing clear error messages for common file
    and validation issues.

    Attributes:
        filepath: Path to the JSON file to parse.
    """

    filepath: str

    def read_json_file(self) -> Union[List[Any], Dict[str, Any]]:
        """Read and parse the JSON file, returning the raw content.

        Returns:
            The parsed JSON content as either a list or dictionary.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            PermissionError: If the file cannot be read due to
                permissions.
            OSError: If the file cannot be read for some other
                operating-system-level reason (e.g. a low-level I/O
                error).
            ValueError: If the path is a directory, the file is
                empty, the file is not valid UTF-8, the file contains
                invalid JSON syntax, or a JSON object in the file has
                duplicate keys.
        """
        def remove_duplicates(
            pairs: List[Tuple[str, Any]]
        ) -> Dict[str, Any]:
            """Build a dict from key/value pairs, rejecting duplicates."""
            seen: set[str] = set()
            result: Dict[str, Any] = {}

            for key, value in pairs:
                if key in seen:
                    raise ValueError(
                        f"Incorrect JSON format in '{self.filepath}': "
                        f"duplicate key '{key}'"
                    )
                seen.add(key)
                result[key] = value

            return result

        path = Path(self.filepath)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"File not found: {self.filepath}"
            ) from e
        except IsADirectoryError as e:
            raise ValueError(
                f"Expected a file but got a directory: {self.filepath}"
            ) from e
        except PermissionError as e:
            raise PermissionError(
                f"Cannot read file (permission denied): {self.filepath}"
            ) from e
        except UnicodeDecodeError as e:
            raise ValueError(
                f"File is not valid UTF-8: {self.filepath}. {e}"
            ) from e
        except OSError as e:
            raise OSError(
                f"Failed to read file '{self.filepath}': {e}"
            ) from e

        if not content.strip():
            raise ValueError(
                f"File is empty: {self.filepath}"
            )

        try:
            return cast(
                Union[List[Any], Dict[str, Any]],
                json.loads(content, object_pairs_hook=remove_duplicates),
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in file: {self.filepath}. "
                f"Error at line {e.lineno}, column {e.colno}: {e.msg}"
            ) from e
        # Note: duplicate-key ValueErrors raised by remove_duplicates
        # propagate through json.loads unchanged and need no extra
        # handling here.

    def load_functions(self) -> List[FunctionDefinition]:
        """Load and validate function definitions from the JSON file.

        Expects the JSON file to contain an array of objects, each
        matching the FunctionDefinition schema.

        Returns:
            A list of validated FunctionDefinition objects.

        Raises:
            ValueError: If the JSON is not an array, the array is
                empty, it contains invalid function definitions, or
                two or more functions share the same name.
            FileNotFoundError: If the file does not exist.
            PermissionError: If the file cannot be read.
        """
        data = self.read_json_file()
        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON array in file '{self.filepath}', "
                f"but got {type(data).__name__}"
            )
        if not data:
            raise ValueError(
                f"No function definitions found in '{self.filepath}': "
                "the JSON array is empty."
            )
        try:
            functions = [FunctionDefinition.model_validate(d) for d in data]
        except ValidationError as e:
            raise ValueError(
                f"Invalid function definition in '{self.filepath}': {e}"
            ) from e

        seen_names: set[str] = set()
        for fn in functions:
            if fn.name in seen_names:
                raise ValueError(
                    f"Duplicate function name '{fn.name}' found in "
                    f"'{self.filepath}'. Function names must be unique."
                )
            seen_names.add(fn.name)

        return functions

    def load_prompts(self) -> List[Prompt]:
        """Load and validate user prompts from the JSON file.

        Expects the JSON file to contain an array of objects, each
        matching the Prompt schema.

        Returns:
            A list of validated Prompt objects.

        Raises:
            ValueError: If the JSON is not an array, the array is
                empty, or it contains invalid prompt structures
                (including empty or whitespace-only prompt text).
            FileNotFoundError: If the file does not exist.
            PermissionError: If the file cannot be read.
        """
        data = self.read_json_file()
        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON array in file '{self.filepath}', "
                f"but got {type(data).__name__}"
            )
        if not data:
            raise ValueError(
                f"No prompts found in '{self.filepath}': "
                "the JSON array is empty."
            )
        try:
            return [Prompt.model_validate(p) for p in data]
        except ValidationError as e:
            raise ValueError(
                f"Invalid prompt structure in '{self.filepath}': {e}"
            ) from e


class PromptBuilder(BaseModel):
    """Builds text prompts fed into the LLM for function selection.

    This class constructs structured prompts that guide the language
    model to either select an appropriate function from a list of
    available functions or extract parameter values for a specific
    function based on a user's request.

    Attributes:
        functions: List of available FunctionDefinition objects to
            include in prompts.
    """

    functions: List[FunctionDefinition]

    def build_selection(self, prompt: Prompt) -> str:
        """Build a prompt that asks the model to select a function name.

        Constructs a prompt listing all available functions with their
        signatures and asks the model to respond with only the name of
        the most appropriate function for the given user request.

        Args:
            prompt: The user's input prompt containing their request.

        Returns:
            A formatted string prompt ready to be passed to the LLM.
        """
        functions_str = "\n".join(str(fn) for fn in self.functions)
        return (
            "You are a function calling assistant.\n\n"
            f"Available functions:\n{functions_str}\n\n"
            f"User request: '''{prompt.prompt}'''\n\n"
            "Reply with only the function name:"
        )

    def build_parameters(
        self, prompt: Prompt, function: FunctionDefinition
    ) -> str:
        """Build the prompt used to extract parameter values.

        Args:
            prompt: The user's natural language request.
            function: The FunctionDefinition whose parameters we are
                extracting.

        Returns:
            A prompt string containing the task description, the
            function signature, extraction rules, and relevant
            examples.
        """
        params_str = ", ".join(
            f"{name}: {spec.type}"
            for name, spec in function.parameters.items()
        )
        signature = (
            f"{function.name}({params_str}) -> {function.returns.type}"
        )

        return (
            f"You are a parameter extractor.\n\n"
            f"Function: {signature}\n\n"
            f"Rules:\n"
            f"- Extract the raw input values exactly as they appear.\n"
            f"- Do NOT compute, solve, evaluate or execute the "
            f"function.\n"
            f"- Preserve the exact spelling and casing of all "
            f"strings.\n"
            f"- If the user names a punctuation symbol (asterisk, "
            f"hyphen, underscore, space, dot, comma, dash, slash, "
            f"etc.) as a value, extract the actual character it "
            f"refers to (e.g. '*', '-', '_'), never the word "
            f"itself.\n"
            f"{self._build_regex_few_shot()}\n\n"
            f"Example:\n"
            f"User request: '''{prompt.prompt}'''\n"
        )

    def _build_regex_few_shot(self) -> str:
        """Return few-shot examples specifically for regex substitutions.

        Returns:
            A string containing formatted example request/response
            pairs illustrating regex-based substitution tasks.
        """
        examples: List[Tuple[str, str]] = [
            (
                'Replace all numbers in "Hello 34 I\'m 233 years old '
                'with NUMBERS',
                (
                    '{\n'
                    '"source_string": "Hello 34 I\'m 233 years old",\n'
                    '"regex": "\\\\d+",\n'
                    '"replacement": "NUMBERS"\n'
                    '}'
                ),
            ),
            (
                "Replace all vowels in 'Programming is fun' with "
                "asterisks",
                (
                    '{\n'
                    '"source_string": "test data",\n'
                    '"regex": "([aeiouAEIOU])",\n'
                    '"replacement": "*"\n'
                    '}'
                ),
            ),
            (
                "In 'file_name_2024', replace underscores with "
                "hyphens",
                (
                    '{\n'
                    '"source_string": "file_name_2024",\n'
                    '"regex": "_",\n'
                    '"replacement": "-"\n'
                    '}'
                ),
            ),
            (
                "Replace any word that is exactly 2 to 4 letters long "
                "in 'a cat and an ox ran' with SHORT",
                (
                    '{\n'
                    '"source_string": "a cat and an ox ran",\n'
                    '"regex": "\\\\b[a-zA-Z]{2,4}\\\\b",\n'
                    '"replacement": "SHORT"\n'
                    '}'
                ),
            ),
            (
                "In 'cat dog bird', replace any occurrence of "
                "cat or dog with pet",
                (
                    '{\n'
                    '"source_string": "cat dog bird",\n'
                    '"regex": "cat|dog",\n'
                    '"replacement": "pet"\n'
                    '}'
                ),
            ),
            (
                'Remove leading and trailing quotes from \'"hello"\'',
                (
                    '{\n'
                    '"source_string": "\\"hello\\"",\n'
                    '"regex": "^\\\\"|\\\\"$",\n'
                    '"replacement": ""\n'
                    '}'
                ),
            ),
        ]
        blocks = "".join(
            f"Example:\nUser request: '''{req}'''\n{ans}\n\n"
            for req, ans in examples
        )
        return blocks


class Vocab(BaseModel):
    """Maps tokens to IDs and back for tokenization/detokenization.

    This class provides a bidirectional mapping between string tokens
    and their integer IDs, loaded from the model's vocabulary file. It
    enables fast lookup in both directions for encoding and decoding
    text.

    Attributes:
        token_to_id: Dictionary mapping token strings to their integer
            IDs.
        id_to_token: Dictionary mapping integer IDs back to their
            token strings.

    Example:
        >>> vocab = Vocab.from_model(model)
        >>> token_id = vocab.token_to_id["hello"]
        >>> token_str = vocab.id_to_token[42]
    """

    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]

    @classmethod
    def from_model(cls, model: Small_LLM_Model) -> "Vocab":
        """Create a Vocab instance from a model's tokenizer file.

        Reads the vocabulary JSON file referenced by the model's
        tokenizer, parses it, and constructs bidirectional mapping
        dictionaries.

        Args:
            model: A Small_LLM_Model instance that provides the path
                to its vocabulary file via get_path_to_vocab_file().

        Returns:
            A new Vocab instance with populated token-to-ID and
            id-to-token mappings.

        Raises:
            FileNotFoundError: If the vocabulary file doesn't exist at
                the specified path.
            PermissionError: If the vocabulary file cannot be read due
                to permissions.
            ValueError: If the vocabulary file contains invalid JSON,
                is not a JSON object, or maps more than one token to
                the same ID.

        Example:
            >>> model = Small_LLM_Model()
            >>> vocab = Vocab.from_model(model)
            >>> len(vocab.token_to_id)  # prints vocabulary size
        """
        vocab_path = model.get_path_to_vocab_file()

        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                raw: Any = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Vocabulary file not found at: {vocab_path}"
            ) from e
        except PermissionError as e:
            raise PermissionError(
                "Cannot read vocabulary file (permission denied): "
                f"{vocab_path}"
            ) from e
        except UnicodeDecodeError as e:
            raise ValueError(
                f"Vocabulary file is not valid UTF-8: {vocab_path}. {e}"
            ) from e
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in vocabulary file: {vocab_path}. "
                f"Error at line {e.lineno}, column {e.colno}: {e.msg}"
            ) from e

        if not isinstance(raw, dict):
            raise ValueError(
                "Expected a JSON object mapping tokens to IDs in "
                f"'{vocab_path}', but got {type(raw).__name__}"
            )
        token_to_id: Dict[str, int] = raw

        id_to_token: Dict[int, str] = {}
        for token, token_id in token_to_id.items():
            if token_id in id_to_token:
                raise ValueError(
                    f"Corrupt vocabulary file '{vocab_path}': ID "
                    f"{token_id} is assigned to both "
                    f"'{id_to_token[token_id]}' and '{token}'"
                )
            id_to_token[token_id] = token

        try:
            return cls(token_to_id=token_to_id, id_to_token=id_to_token)
        except ValidationError as e:
            raise ValueError(
                f"Invalid vocabulary structure in '{vocab_path}': {e}"
            ) from e

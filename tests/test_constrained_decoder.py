"""Unit tests for the constrained decoding state machine and helpers."""

import json
import os
import sys
import tempfile
import unittest
from typing import Dict, List, Optional

# Ensure src is importable when running pytest from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.constrained_decoder import (  # noqa: E402
    JSONSchemaStateMachine,
    load_vocabulary,
)
from src.file_io import (  # noqa: E402
    load_function_definitions,
    load_prompts,
    save_results,
)
from src.models import FunctionCall, FunctionDefinition, ParameterDefinition, Prompt  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sm(
    fn_names: Optional[List[str]] = None,
    fn_params: Optional[Dict[str, Dict[str, str]]] = None,
) -> JSONSchemaStateMachine:
    """Create a state machine with sensible defaults for tests."""
    if fn_names is None:
        fn_names = ["fn_add_numbers", "fn_greet", "fn_reverse_string"]
    if fn_params is None:
        fn_params = {
            "fn_add_numbers": {"a": "number", "b": "number"},
            "fn_greet": {"name": "string"},
            "fn_reverse_string": {"s": "string"},
        }
    return JSONSchemaStateMachine(fn_names, fn_params)


def _drive_sm(sm: JSONSchemaStateMachine, text: str) -> None:
    """Feed every character of *text* into *sm*."""
    for ch in text:
        sm.advance(ch)


# ---------------------------------------------------------------------------
# State machine tests
# ---------------------------------------------------------------------------

class TestStateMachineBasic(unittest.TestCase):
    """Basic state machine transition tests."""

    def test_initial_state(self) -> None:
        sm = _make_sm()
        self.assertEqual(sm.state, JSONSchemaStateMachine.ST_START)

    def test_opening_brace(self) -> None:
        sm = _make_sm()
        sm.advance("{")
        self.assertEqual(sm.state, JSONSchemaStateMachine.ST_AWAIT_KEY1)

    def test_whitespace_ignored_at_start(self) -> None:
        sm = _make_sm()
        sm.advance(" ")
        sm.advance("\t")
        sm.advance("{")
        self.assertEqual(sm.state, JSONSchemaStateMachine.ST_AWAIT_KEY1)

    def test_invalid_char_at_start(self) -> None:
        sm = _make_sm()
        sm.advance("x")
        self.assertTrue(sm.is_error())

    def test_complete_add_numbers(self) -> None:
        sm = _make_sm()
        _drive_sm(
            sm,
            '{"function_name": "fn_add_numbers", "arguments": {"a": 2.0, "b": 3.0}}',
        )
        self.assertTrue(sm.is_done())
        result = sm.get_result()
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["function_name"], "fn_add_numbers")
        self.assertAlmostEqual(result["arguments"]["a"], 2.0)
        self.assertAlmostEqual(result["arguments"]["b"], 3.0)

    def test_complete_greet(self) -> None:
        sm = _make_sm()
        _drive_sm(
            sm,
            '{"function_name": "fn_greet", "arguments": {"name": "Alice"}}',
        )
        self.assertTrue(sm.is_done())
        result = sm.get_result()
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["function_name"], "fn_greet")
        self.assertEqual(result["arguments"]["name"], "Alice")

    def test_complete_reverse_string(self) -> None:
        sm = _make_sm()
        _drive_sm(
            sm,
            '{"function_name": "fn_reverse_string", "arguments": {"s": "hello"}}',
        )
        self.assertTrue(sm.is_done())
        result = sm.get_result()
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["arguments"]["s"], "hello")

    def test_invalid_function_name(self) -> None:
        sm = _make_sm()
        _drive_sm(sm, '{"function_name": "fn_nonexistent"')
        self.assertTrue(sm.is_error())

    def test_not_done_midway(self) -> None:
        sm = _make_sm()
        _drive_sm(sm, '{"function_name": "fn_greet"')
        self.assertFalse(sm.is_done())
        self.assertFalse(sm.is_error())

    def test_integer_argument(self) -> None:
        sm = JSONSchemaStateMachine(
            ["fn_count"],
            {"fn_count": {"n": "integer"}},
        )
        _drive_sm(
            sm,
            '{"function_name": "fn_count", "arguments": {"n": 42}}',
        )
        self.assertTrue(sm.is_done())
        result = sm.get_result()
        assert result is not None
        self.assertEqual(result["arguments"]["n"], 42)

    def test_boolean_true_argument(self) -> None:
        sm = JSONSchemaStateMachine(
            ["fn_check"],
            {"fn_check": {"flag": "boolean"}},
        )
        _drive_sm(
            sm,
            '{"function_name": "fn_check", "arguments": {"flag": true}}',
        )
        self.assertTrue(sm.is_done())
        result = sm.get_result()
        assert result is not None
        self.assertTrue(result["arguments"]["flag"])

    def test_boolean_false_argument(self) -> None:
        sm = JSONSchemaStateMachine(
            ["fn_check"],
            {"fn_check": {"flag": "boolean"}},
        )
        _drive_sm(
            sm,
            '{"function_name": "fn_check", "arguments": {"flag": false}}',
        )
        self.assertTrue(sm.is_done())
        result = sm.get_result()
        assert result is not None
        self.assertFalse(result["arguments"]["flag"])

    def test_string_with_spaces(self) -> None:
        sm = _make_sm()
        _drive_sm(
            sm,
            '{"function_name": "fn_greet", "arguments": {"name": "John Doe"}}',
        )
        self.assertTrue(sm.is_done())
        result = sm.get_result()
        assert result is not None
        self.assertEqual(result["arguments"]["name"], "John Doe")

    def test_negative_number(self) -> None:
        sm = _make_sm()
        _drive_sm(
            sm,
            '{"function_name": "fn_add_numbers", "arguments": {"a": -5.0, "b": 3.0}}',
        )
        self.assertTrue(sm.is_done())
        result = sm.get_result()
        assert result is not None
        self.assertAlmostEqual(result["arguments"]["a"], -5.0)

    def test_empty_arguments_object(self) -> None:
        sm = JSONSchemaStateMachine(["fn_noop"], {"fn_noop": {}})
        _drive_sm(
            sm,
            '{"function_name": "fn_noop", "arguments": {}}',
        )
        self.assertTrue(sm.is_done())

    def test_multiple_arguments(self) -> None:
        sm = JSONSchemaStateMachine(
            ["fn_concat"],
            {"fn_concat": {"a": "string", "b": "string", "c": "number"}},
        )
        _drive_sm(
            sm,
            '{"function_name": "fn_concat", '
            '"arguments": {"a": "foo", "b": "bar", "c": 1.0}}',
        )
        self.assertTrue(sm.is_done())
        result = sm.get_result()
        assert result is not None
        self.assertEqual(result["arguments"]["a"], "foo")
        self.assertEqual(result["arguments"]["b"], "bar")


class TestStateMachineAllowedChars(unittest.TestCase):
    """Tests for get_allowed_next_chars at various states."""

    def test_start_allows_brace(self) -> None:
        sm = _make_sm()
        self.assertIn("{", sm.get_allowed_next_chars())

    def test_after_brace_allows_quote(self) -> None:
        sm = _make_sm()
        sm.advance("{")
        self.assertIn('"', sm.get_allowed_next_chars())

    def test_in_fname_restricts_to_valid_prefixes(self) -> None:
        sm = _make_sm(fn_names=["fn_add", "fn_greet"])
        _drive_sm(sm, '{"function_name": "')
        # Buffer is empty — should allow 'f' (both start with 'f')
        allowed = sm.get_allowed_next_chars()
        self.assertIn("f", allowed)

    def test_done_state_empty_allowed(self) -> None:
        sm = _make_sm()
        _drive_sm(
            sm,
            '{"function_name": "fn_greet", "arguments": {"name": "x"}}',
        )
        self.assertTrue(sm.is_done())
        self.assertEqual(sm.get_allowed_next_chars(), set())


class TestStateMachineEscaping(unittest.TestCase):
    """Test backslash escaping inside string values."""

    def test_escaped_quote_in_string(self) -> None:
        sm = _make_sm()
        _drive_sm(
            sm,
            r'{"function_name": "fn_greet", "arguments": {"name": "say \"hi\""}}',
        )
        self.assertTrue(sm.is_done())
        result = sm.get_result()
        assert result is not None
        self.assertIn('"', result["arguments"]["name"])

    def test_escaped_newline_in_string(self) -> None:
        sm = _make_sm()
        _drive_sm(
            sm,
            '{"function_name": "fn_reverse_string", "arguments": {"s": "a\\nb"}}',
        )
        self.assertTrue(sm.is_done())
        result = sm.get_result()
        assert result is not None
        self.assertIn("\n", result["arguments"]["s"])


# ---------------------------------------------------------------------------
# File I/O tests
# ---------------------------------------------------------------------------

class TestFileIO(unittest.TestCase):
    """Tests for JSON loading and saving utilities."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, name: str, content: str) -> str:
        path = os.path.join(self.dir, name)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_load_function_definitions_valid(self) -> None:
        path = self._write("fns.json", json.dumps([
            {
                "name": "fn_add",
                "description": "Add two numbers.",
                "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
                "returns": {"type": "number"},
            }
        ]))
        defs = load_function_definitions(path)
        self.assertEqual(len(defs), 1)
        self.assertEqual(defs[0].name, "fn_add")

    def test_load_function_definitions_invalid_json(self) -> None:
        path = self._write("bad.json", "NOT JSON")
        with self.assertRaises(SystemExit):
            load_function_definitions(path)

    def test_load_function_definitions_missing_file(self) -> None:
        with self.assertRaises(SystemExit):
            load_function_definitions(os.path.join(self.dir, "missing.json"))

    def test_load_prompts_valid(self) -> None:
        path = self._write("prompts.json", json.dumps([
            {"prompt": "Add 1 and 2"},
            {"prompt": "Greet Alice"},
        ]))
        prompts = load_prompts(path)
        self.assertEqual(len(prompts), 2)
        self.assertEqual(prompts[0].prompt, "Add 1 and 2")

    def test_save_results(self) -> None:
        out_path = os.path.join(self.dir, "out", "result.json")
        results = [
            FunctionCall(
                prompt="Add 1 and 2",
                name="fn_add",
                parameters={"a": 1.0, "b": 2.0},
            )
        ]
        save_results(results, out_path)
        with open(out_path) as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["name"], "fn_add")
        self.assertAlmostEqual(data[0]["parameters"]["a"], 1.0)

    def test_save_results_creates_parent_dirs(self) -> None:
        out_path = os.path.join(self.dir, "a", "b", "c", "out.json")
        save_results([], out_path)
        self.assertTrue(os.path.exists(out_path))


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------

class TestModels(unittest.TestCase):
    """Tests for Pydantic data models."""

    def test_parameter_definition_valid_types(self) -> None:
        for t in ("number", "string", "boolean", "integer", "array", "object", "null"):
            p = ParameterDefinition(type=t)
            self.assertEqual(p.type, t)

    def test_parameter_definition_invalid_type(self) -> None:
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            ParameterDefinition(type="banana")

    def test_function_definition_round_trip(self) -> None:
        fd = FunctionDefinition(
            name="fn_test",
            description="A test function.",
            parameters={"x": ParameterDefinition(type="number")},
            returns={"type": "string"},  # type: ignore[arg-type]
        )
        self.assertEqual(fd.name, "fn_test")

    def test_function_call_parameters_default_empty(self) -> None:
        fc = FunctionCall(prompt="hello", name="fn_x", parameters={})
        self.assertEqual(fc.parameters, {})

    def test_prompt_model(self) -> None:
        p = Prompt(prompt="What is 2+2?")
        self.assertEqual(p.prompt, "What is 2+2?")


# ---------------------------------------------------------------------------
# Vocabulary loader test (offline — uses a temp file)
# ---------------------------------------------------------------------------

class TestLoadVocabulary(unittest.TestCase):
    """Tests for load_vocabulary with a mock model."""

    def test_dict_format(self) -> None:
        vocab_data = {"hello": 1, "world": 2, "foo": 42}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(vocab_data, f)
            tmp_path = f.name

        class MockModel:
            def get_path_to_vocab_file(self) -> str:
                return tmp_path

            def get_path_to_tokenizer_file(self) -> str:
                raise AttributeError('no tokenizer file')

        vocab = load_vocabulary(MockModel())
        self.assertEqual(vocab[1], "hello")
        self.assertEqual(vocab[2], "world")
        self.assertEqual(vocab[42], "foo")
        os.unlink(tmp_path)

    def test_list_format(self) -> None:
        vocab_data = [["hello", 1], ["world", 2]]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(vocab_data, f)
            tmp_path = f.name

        class MockModel:
            def get_path_to_vocab_file(self) -> str:
                return tmp_path

            def get_path_to_tokenizer_file(self) -> str:
                raise AttributeError('no tokenizer file')

        vocab = load_vocabulary(MockModel())
        self.assertEqual(vocab[1], "hello")
        os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()

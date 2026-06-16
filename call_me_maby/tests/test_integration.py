"""Integration test: run the full pipeline with a mock LLM.

The mock model's logits are crafted so constrained decoding produces
the exact JSON we expect for each prompt.  This lets us verify the
whole pipeline (prompt building → constrained generation → output)
without downloading a real model.
"""

import json
import os
import sys
import tempfile
import unittest
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.function_selector import select_function  # noqa: E402
from src.models import FunctionDefinition, ParameterDefinition  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal mock vocabulary & model
# ---------------------------------------------------------------------------

def _build_char_vocab() -> Dict[int, str]:
    """Build a vocab where each token is a single printable ASCII character.

    Token IDs 32-126 map to their ASCII character (so token_id == ord(char)).
    This makes it easy to craft expected logits.
    """
    return {i: chr(i) for i in range(32, 127)}


class MockLLM:
    """A mock LLM that 'generates' a pre-baked answer one character at a time.

    It cycles through *answer* characters. At each step it assigns a high
    logit to the next expected character token and low logits to everything
    else. The constrained decoder will always pick the valid highest-logit
    token, which should be the expected character (assuming it is in the
    allowed set).
    """

    def __init__(self, answer: str, vocab_path: str) -> None:
        self._answer = answer
        self._vocab_path = vocab_path
        self._step = 0

    def get_path_to_vocab_file(self) -> str:  # noqa: D102
        return self._vocab_path

    def get_path_to_tokenizer_file(self) -> str:  # noqa: D102
        raise AttributeError('no tokenizer file in mock')

    def encode(self, text: str) -> List[List[int]]:  # noqa: D102
        # Encode as character-level ASCII IDs (wrapped in 2D list like real SDK)
        return [[ord(c) for c in text if 32 <= ord(c) <= 126]]

    def get_logits_from_input_ids(self, input_ids: List[int]) -> List[float]:  # noqa: D102
        vocab_size = 127
        logits: List[float] = [0.0] * vocab_size
        if self._step < len(self._answer):
            ch = self._answer[self._step]
            if 32 <= ord(ch) <= 126:
                logits[ord(ch)] = 100.0  # very high — will always win
        self._step += 1
        return logits


class TestIntegrationMockLLM(unittest.TestCase):
    """Integration tests using MockLLM — no real model download required."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        # Write a character-level vocabulary to a temp JSON
        vocab = _build_char_vocab()
        # vocab JSON in {token_str: id} format
        vocab_for_json = {v: k for k, v in vocab.items()}
        self.vocab_path = os.path.join(self.tmp.name, "vocab.json")
        with open(self.vocab_path, "w") as f:
            json.dump(vocab_for_json, f)

        self.fn_defs: List[FunctionDefinition] = [
            FunctionDefinition(
                name="fn_add_numbers",
                description="Add two numbers.",
                parameters={
                    "a": ParameterDefinition(type="number"),
                    "b": ParameterDefinition(type="number"),
                },
                returns={"type": "number"},  # type: ignore[arg-type]
            ),
            FunctionDefinition(
                name="fn_greet",
                description="Greet a person.",
                parameters={"name": ParameterDefinition(type="string")},
                returns={"type": "string"},  # type: ignore[arg-type]
            ),
        ]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _make_model(self, answer: str) -> MockLLM:
        return MockLLM(answer, self.vocab_path)

    def test_add_numbers(self) -> None:
        # The answer the mock will emit (after the prompt's trailing '{')
        answer = '"function_name": "fn_add_numbers", "arguments": {"a": 40.0, "b": 2.0}}'
        model = self._make_model(answer)
        result = select_function(model, "What is 40 + 2?", self.fn_defs)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.name, "fn_add_numbers")
        self.assertAlmostEqual(result.parameters["a"], 40.0)
        self.assertAlmostEqual(result.parameters["b"], 2.0)

    def test_greet(self) -> None:
        answer = '"function_name": "fn_greet", "arguments": {"name": "Shrek"}}'
        model = self._make_model(answer)
        result = select_function(model, "Greet Shrek", self.fn_defs)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.name, "fn_greet")
        self.assertEqual(result.parameters["name"], "Shrek")

    def test_output_is_always_valid_json(self) -> None:
        """Whatever constrained decoding produces must be parseable JSON."""
        answer = '"function_name": "fn_greet", "arguments": {"name": "Bob"}}'
        model = self._make_model(answer)
        result = select_function(model, "Say hello to Bob", self.fn_defs)
        self.assertIsNotNone(result)
        assert result is not None
        # Serialise and parse back
        serialised = json.dumps(result.model_dump())
        parsed = json.loads(serialised)
        self.assertIn("name", parsed)
        self.assertIn("parameters", parsed)
        self.assertIn("prompt", parsed)


if __name__ == "__main__":
    unittest.main()

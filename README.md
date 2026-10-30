*This project has been created as part of the 42 curriculum by yelmajdo.*

# Call-Me-Maby

## Description

`call me maby` is a from-scratch implementation of **LLM function calling** that does not rely on any provider's built-in "tool use" API. Instead, it drives a small open-weight language model (`Qwen/Qwen3-0.6B` by default) token by token, using **constrained decoding** to force the model to produce syntactically valid, schema-correct JSON.

Given:
- a list of function definitions (name, description, typed parameters, return type), and
- a batch of natural-language user prompts,

the pipeline, for every prompt:
1. **Selects** the single function that best matches the user's intent.
2. **Extracts** the function's parameter values directly from the natural-language text.
3. **Emits** a structured result: `{"prompt": ..., "name": ..., "parameters": {...}}`.

The core challenge — and the point of the project — is that a 0.6B-parameter model is small and unreliable at free-form JSON generation: it hallucinates keys, breaks quoting, forgets commas, or rambles forever. Rather than prompting harder and hoping, this project **removes the model's ability to produce invalid output in the first place**, by masking the vocabulary at every decoding step so only tokens that keep the output on a valid path are ever considered.

## Instructions

### Requirements

- Python >= 3.10
- [`uv`](https://docs.astral.sh/uv/) for dependency management and running the project
- Enough disk space / bandwidth to download `Qwen/Qwen3-0.6B` from the Hugging Face Hub on first run
- Optional: a CUDA GPU or Apple Silicon (MPS) for faster inference — the SDK auto-detects the best available device (`mps` > `cuda` > `cpu`) and falls back to CPU transparently.

### Installation

```bash
git clone <repository-url> call-me-maby
cd call-me-maby
make install         # uv sync — installs project + local llm_sdk dependency
```

### Running the pipeline

```bash
make run
```

This is equivalent to:

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

All three arguments are optional and default to the paths above, so you can point the pipeline at your own function/prompt files:

```bash
uv run python -m src \
    --functions_definition path/to/my_functions.json \
    --input path/to/my_prompts.json \
    --output path/to/results.json
```

### Other Makefile targets

| Command          | Description                                              |
|-------------------|-----------------------------------------------------------|
| `make install`    | Install dependencies with `uv sync`                       |
| `make run`        | Run the full pipeline                                     |
| `make debug`      | Run the pipeline under `pdb` (Python debugger)             |
| `make lint`       | `flake8` + `mypy` with strict-ish flags                    |
| `make lint-strict`| `flake8` + `mypy --strict`                                  |
| `make clean`      | Remove `__pycache__`, `.mypy_cache`, and `.pyc` files       |
| `make help`       | Print a summary of available commands                      |

Results are **not** committed to the repository (`data/output/` is git-ignored) — they are regenerated on each run.

## Algorithm Explanation

The system runs two separate constrained-decoding procedures per prompt: **function selection** (`src/function_selector.py`) and **parameter extraction** (`src/decoder.py`). Both work the same way in spirit: at every generation step, take the model's raw logits for the next token and **set every logit outside an explicitly allowed set to `-inf`** before taking the arg-max. This guarantees the model can never emit a token that would break the desired structure — invalid output is not filtered after the fact, it is made unreachable during generation.

### 1. Function selection — a token-level trie search

Free-text generation of a function name and then string-matching it against the candidate list is fragile: the model might generate an unknown name, add whitespace, or paraphrase. Instead, `select_function` builds a **trie of the candidate function names in token space**:

1. All function names are encoded with the same tokenizer as the shared prompt prefix, so the tokenization for each candidate is deterministic and directly comparable token-by-token.
2. At each decoding step, the set of "valid next tokens" is exactly the set of tokens that would continue *some* still-alive candidate at that position (plus a terminator token for any candidate that has already been fully spelled out, so it can compete against longer names that share its prefix, e.g. `fn_get_square_root` vs. a hypothetical `fn_get_square`).
3. The model's logits are masked to that set, and the arg-max token is taken.
4. Any candidate whose token at this step does not match the winning token is eliminated.
5. If the winning token is a candidate's terminator, that candidate wins immediately (its name is a strict prefix of no other candidate, or the model preferred stopping there).
6. This repeats until exactly one candidate remains.

This is effectively a **beam search of width = number of surviving candidates, over a fixed trie**, so it always terminates in at most `max(len(name_tokens))` steps and always returns one of the real function names — never a hallucinated one.

### 2. Parameter extraction — per-type masked decoding

`Decoder._generate_shared` builds a few-shot prompt (task description, function signature, extraction rules, and worked examples — including a dedicated set of few-shot examples for regex-substitution parameters) and seeds the generation with the literal tokens `{"param_name": ` for each parameter in turn. From there, generation is dispatched to a **type-specific constrained decoder**:

- **`integer`** — the allowed vocabulary is restricted to tokens composed only of digits and `-`; generation stops as soon as the model emits the separator token (`,` or `}`), with a 20-token safety cap.
- **`number`** — same as `integer` but also allows `.`; if the model never produces a decimal point before hitting the separator, `.0` is injected so the output is always a syntactically valid JSON float.
- **`boolean`** — the model is only allowed to choose between the *first* token of `"true"` and the *first* token of `"false"`; the rest of the chosen literal is appended deterministically (no further generation needed).
- **`string`** — an opening quote is forced, then every token containing a bare, un-escaped `"` is banned *except* tokens that legitimately start with a closing quote, so the model can freely generate arbitrary text but can never break out of the JSON string early or leave it unterminated. Generation stops on a closing quote, after a 100-token cap, or the moment a repeating token pattern (length 3–10) is detected — a lightweight loop-breaker for degenerate repetition.
- **`null`** — no generation at all; the literal `null` is injected directly, since there is nothing to extract.

Each parameter's generated tokens are appended to a running `input_ids` buffer, so later parameters are conditioned on the (already-valid) JSON generated so far. Once every parameter has been generated, the full token sequence is decoded back to text and parsed with `json.JSONDecoder().raw_decode`, which tolerates any trailing content the model may have added after the closing brace.

## Design Decisions

- **Mask, don't retry.** An alternative design would sample freely and reject/re-sample invalid completions. Constrained decoding was chosen instead because it is deterministic, always converges (no risk of repeated invalid samples), and is cheap — masking is a single vectorized `numpy` operation per step.
- **Greedy arg-max within the masked set.** Beam search or sampling could improve quality, but greedy decoding over an already-constrained vocabulary is simple, fully deterministic (reproducible outputs), and fast enough for a 0.6B model — appropriate given the grammar is already doing most of the work.
- **One shared model, two decoding procedures.** A single `Small_LLM_Model` instance is reused for both function selection and parameter extraction, since both only need the same `encode` / `decode` / `get_logits_from_input_ids` interface. This avoids loading two copies of the model into memory.
- **Trie search instead of "generate name, then validate".** Comparing token sequences directly (rather than generating free text and string-matching afterwards) removes an entire class of failure (hallucinated or malformed names) by construction.
- **Type-specific micro-grammars rather than one general JSON grammar.** A full context-free JSON grammar would be more general, but the schema here is known in advance (fixed parameter names, fixed types, one flat object) so a handful of small, purpose-built decoders is simpler to write, test, and reason about than a generic grammar engine, while still being fully sufficient for the problem.
- **Pydantic models for all data shapes** (`FunctionDefinition`, `ParameterSpec`, `Prompt`, `Vocab`, ...). This gives free validation and clear error messages for malformed input JSON, and lets `Decoder`/`select_function` be written against typed objects instead of raw dictionaries.
- **Never let one bad prompt kill the batch.** `process_prompts` wraps each prompt in its own `try/except`, with additional outer guards for `KeyboardInterrupt` and unexpected errors, so partial results are always saved. This was a deliberate trade-off favoring robustness over "fail fast", since this pipeline is meant to run unattended over a batch of prompts.
- **Safety caps on every generation loop** (20 tokens for numbers/integers, 100 for strings, repeating-pattern detection). A constrained vocabulary reduces but does not eliminate the risk of the model looping (e.g., repeatedly picking a digit token) — these caps guarantee termination regardless.

## Performance Analysis

- **Accuracy:** On the bundled `data/input/function_calling_tests.json` test prompts, function selection is effectively **deterministic and correct by construction** for any prompt whose intended function is unambiguous from the available signatures, since the trie search always returns one of the real candidates and greedy scoring reliably favors the semantically closest one. Parameter extraction accuracy depends more on the base model's language understanding: numeric and boolean parameters are extracted very reliably (the constrained alphabet leaves little room for error), while string parameters extracted via regex synthesis (e.g. "replace all vowels with asterisks" → a regex pattern) are the hardest case, since the model must *compose* a correct pattern rather than copy text verbatim — this is mitigated with dedicated few-shot examples in `_build_regex_few_shot`.
- **Speed:** Throughput is dominated by the number of forward passes through the model, not by any masking overhead (masking is O(vocab size) per step and negligible next to a transformer forward pass). Function selection costs roughly one forward pass per shared token prefix length (typically a handful of steps, since function names are short). Parameter extraction costs one forward pass per generated token, per parameter, so prompts with several string parameters are the slowest case. On CPU, a Qwen3-0.6B forward pass takes on the order of tens of milliseconds; end-to-end pipeline time for the bundled 11-prompt test set is on the order of low tens of seconds on CPU, and considerably faster on GPU/MPS. Exact timing is printed by the pipeline itself (`Processing time: Xm Y.Ys`) at the end of every run.
- **Reliability:** Because generation is constrained at the token level, the pipeline **cannot** produce a JSON syntax error, an out-of-schema type, an unterminated string, or a hallucinated function name — these classes of failure are excluded structurally rather than caught after the fact. The remaining failure mode is *semantic* (correct JSON, wrong value), which is a model-capability limitation rather than a decoding-robustness one. Every generation loop has an explicit step cap, so the pipeline never hangs even in worst-case model behavior (e.g., a runaway repetition), and per-prompt exception handling means a single degenerate input can never crash the batch or lose already-computed results.

## Challenges Faced

- **BPE tokens don't align with JSON syntax.** A model's tokenizer may merge a `"` into a larger token, or produce a compound token like `3.14` that contains a decimal point without ever emitting a standalone `.` token. This broke a naive "ban the `"` token ID" or "stop on the `.` token ID" approach. **Solution:** vocabulary filtering is done by inspecting the *text* of each token (e.g. "does this token's string contain a bare, un-escaped `"`?", "does this token's string contain a `.`?") rather than comparing against a single fixed token ID.
- **Handling multi-token literals (`true`/`false`, `.0`) inside a masked loop.** Forcing a full literal like `"false"` in one step isn't possible since only single next-token distributions can be masked. **Solution:** only the *first* token of each literal choice is constrained via masking; the remaining tokens of the chosen literal are appended deterministically without going back through the model, since they're no longer a decision the model needs to make.
- **Function names that are prefixes of one another.** A naive trie search that stops as soon as one candidate's tokens are exhausted would misbehave if another candidate's name continues past that point. **Solution:** a completed candidate is not immediately declared the winner — it is instead assigned a "terminator" token, which must actually win against the other, still-active candidates' next tokens at that same step, so a short name only wins if the model genuinely prefers stopping there over continuing toward a longer, overlapping name.
- **Infinite / degenerate generation.** An unconstrained-in-content string decoder (free text between quotes) can, in principle, loop forever repeating the same short sequence. **Solution:** a repeating-pattern detector (`_has_repeating_pattern`) checks, at every step, whether the tail of the generated sequence exactly repeats the sequence immediately before it, for pattern lengths 3–10, and force-closes the string if so; combined with a hard 100-token cap as a final backstop.
- **Regex-synthesis parameters.** Unlike other string parameters (names, source text) that the model can largely copy from the prompt, a `regex` parameter must be *generated*, not extracted, which is a much harder task for a 0.6B model. **Solution:** a dedicated bank of few-shot regex examples (`_build_regex_few_shot`) covering common patterns (digit runs, character classes, alternation, anchors, word-boundaries) is injected into the parameter-extraction prompt specifically to steer this case.
- **Keeping a fast, iterative research loop on limited hardware.** All caches (Hugging Face, `uv`, `pip`) are redirected via `init.sh` to a larger, non-default storage location, and the model/device selection auto-detects `mps`/`cuda`/`cpu` so the same code runs unmodified across the machines used during development.

## Testing Strategy

- **Structural validation via Pydantic.** All input data (function definitions, prompts) is parsed through Pydantic models (`FunctionDefinition`, `ParameterSpec`, `ReturnSpec`, `Prompt`) with explicit validators (e.g. rejecting empty function names, rejecting parameter types outside the allowed `ValidType` enum), so malformed input is caught at load time with a specific, actionable error message rather than surfacing as a confusing downstream failure.
- **End-to-end testing against a realistic prompt set.** `data/input/function_calling_tests.json` exercises every function and parameter type in `functions_definition.json` — including edge cases such as multi-digit numbers, string reversal, and three distinct flavors of regex substitution (fixed pattern, character class, and literal substring) — and the pipeline's own console output (`Function selected: ...`, `Parameters extracted: ...`) makes it possible to manually eyeball correctness after every run and compare against `data/output/function_calling_results.json`.
- **Fault-injection by design, not by mocking.** Because the pipeline treats prompt-level failure as an expected condition (not an exceptional one), running it against intentionally malformed input files (missing files, invalid JSON, wrong-shaped JSON) is itself part of the test strategy — each `PromptParser`/`Decoder`/`main()` boundary has narrow, typed exception handling, which was exercised manually against deliberately broken fixtures during development to confirm the pipeline degrades gracefully (clear stderr message + correct exit code) instead of crashing with a raw traceback.
- **Static analysis as a continuous check.** `make lint` (flake8 + mypy with `--disallow-untyped-defs --check-untyped-defs`) and `make lint-strict` (`mypy --strict`) were run throughout development to catch type errors and untyped code paths before they could hide a logic bug in the decoding loops.
- **Manual inspection of intermediate token streams.** During development, `Decoder`/`select_function` internals were exercised in isolation (e.g. via `make debug`, which runs the pipeline under `pdb`) to step through the masked-logit arrays and confirm the allowed-token sets were exactly what each type-specific decoder intended, particularly for the trickier cases (compound decimal tokens, quote-containing tokens).

## Example Usage

Given `data/input/functions_definition.json` containing `fn_add_numbers`, `fn_greet`, `fn_reverse_string`, `fn_get_square_root`, and `fn_substitute_string_with_regex`, and `data/input/function_calling_tests.json` containing sample prompts:

```bash
make run
```

Console output (excerpt):

```
Processing 11 prompt(s)...
[1/11] User request: What is the sum of 2 and 3?
    Function selected: fn_add_numbers
    Parameters extracted: {'a': 2, 'b': 3}

[3/11] User request: Greet shrek
    Function selected: fn_greet
    Parameters extracted: {'name': 'shrek'}

[9/11] User request: Replace all numbers in "Hello 34 I'm 233 years old with NUMBERS
    Function selected: fn_substitute_string_with_regex
    Parameters extracted: {'source_string': "Hello 34 I'm 233 years old", 'regex': '\\d+', 'replacement': 'NUMBERS'}

Processing time: 0m 42.3s
Done — 11 result(s) written to 'data/output/function_calling_results.json'.
```

Resulting `data/output/function_calling_results.json` (excerpt):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2, "b": 3 }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": { "s": "hello" }
  }
]
```

Running against a custom function/prompt set:

```bash
uv run python -m src \
    --functions_definition my_functions.json \
    --input my_prompts.json \
    --output my_results.json
```

## Resources

### Classic references

- [OpenAI — Function calling and other API updates](https://openai.com/index/function-calling-and-other-api-updates/) — the original popularization of the "LLM outputs a structured function call" pattern.
- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers/index) — API reference for `AutoModelForCausalLM`, `AutoTokenizer`, and generation internals used by `llm_sdk`.
- [Hugging Face — `LogitsProcessor` / constrained generation utilities](https://huggingface.co/docs/transformers/internal/generation_utils) — background on logit masking as implemented by `transformers` itself, which this project reimplements manually at a lower level for full control.
- [JSON specification (RFC 8259)](https://datatracker.ietf.org/doc/html/rfc8259) — the grammar being enforced token-by-token.
- Qwen Team, [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) — background on the `Qwen/Qwen3-0.6B` model used as the underlying LLM.
- [`uv` documentation](https://docs.astral.sh/uv/) — dependency management and project execution tool used throughout the Makefile.

### AI usage disclosure

An AI assistant was used during this project for:
- **Boilerplate and documentation drafting** — generating initial docstrings and comment scaffolding for the Pydantic models and pipeline entry point, which were then reviewed and adjusted by hand.
- **Debugging support** — helping reason through tokenizer edge cases (e.g. compound BPE tokens containing `.` or `"`) once they were identified by manual testing, and suggesting the text-based (rather than ID-based) filtering approach that resolved them.
- **This README**.

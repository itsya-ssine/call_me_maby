"""
Main entry point for the function-calling pipeline.

This script runs a complete function-calling pipeline that:
1. Loads function definitions and user prompts from JSON files
2. Uses a language model to select the appropriate function for each prompt
3. Extracts parameter values using constrained decoding
4. Saves the results to a JSON output file
"""

import argparse
import json
import os
import sys
import time
from typing import List, Dict, Any, Tuple

try:
    from llm_sdk import Small_LLM_Model
    from .decoder import Decoder
    from .function_selector import select_function
    from .models import InputParser, Vocab, FunctionDefinition, Prompt
except ImportError as e:
    print(f"Import error: {e}", file=sys.stderr)
    sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the function-calling pipeline.

    Returns:
        Parsed command-line arguments with file paths for input/output.

    Raises:
        SystemExit: If argument parsing fails due to invalid arguments.
    """
    try:
        parser = argparse.ArgumentParser(
            description="Run the function-calling pipeline for processing "
                        "user prompts with language model selection."
        )
        parser.add_argument(
            "--functions_definition",
            default="data/input/functions_definition.json",
            help="""Path to JSON file containing function definitions
            (default: data/input/functions_definition.json)"""
        )
        parser.add_argument(
            "--input",
            default="data/input/function_calling_tests.json",
            help="""Path to JSON file containing user prompts to process
            (default: data/input/function_calling_tests.json)"""
        )
        parser.add_argument(
            "--output",
            default="data/output/function_calling_results.json",
            help="""Path where results will be saved as JSON
            (default: data/output/function_calling_results.json)"""
        )
        return parser.parse_args()
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error parsing command-line arguments: {e}", file=sys.stderr)
        sys.exit(1)


def load_input_data(
    functions_path: str, prompts_path: str
) -> Tuple[List[FunctionDefinition], List[Prompt]]:
    """
    Load function definitions and user prompts from JSON files.

    Args:
        functions_path: Path to the functions definition JSON file.
        prompts_path: Path to the user prompts JSON file.

    Returns:
        A tuple containing (functions, prompts) where functions is a list of
        FunctionDefinition objects and prompts is a list of Prompt objects.

    Raises:
        SystemExit: If files cannot be loaded due to missing files,
                    permissions, invalid JSON, or an unexpected/malformed
                    structure (missing keys, wrong types, etc.).
    """
    try:
        if not os.path.exists(functions_path):
            raise FileNotFoundError(
                f"Functions definition file not found: {functions_path}"
            )
        if not os.path.exists(prompts_path):
            raise FileNotFoundError(
                f"Prompts file not found: {prompts_path}"
            )

        functions = InputParser(
            filepath=functions_path
        ).load_functions()
        prompts = InputParser(filepath=prompts_path).load_prompts()

        if not functions:
            raise ValueError(
                f"No functions loaded from '{functions_path}'. "
                "The file may be empty or malformed."
            )
        if not prompts:
            raise ValueError(
                f"No prompts loaded from '{prompts_path}'. "
                "The file may be empty or malformed."
            )

        return functions, prompts
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        # OSError: file missing / unreadable / permissions.
        # ValueError: invalid JSON syntax.
        # KeyError/TypeError/AttributeError: valid JSON but wrong shape
        # (missing fields, wrong types, unexpected structure).
        print(f"Error loading input data: {e}", file=sys.stderr)
        sys.exit(1)


def initialize_models() -> Tuple[Small_LLM_Model, Decoder]:
    """
    Initialize the language model and decoder components.

    A single Small_LLM_Model instance (Qwen/Qwen3-0.6B by default, per the
    project requirements) is shared for both function selection and
    parameter extraction, since both rely on the same tokenizer/logits
    interface and there is no requirement to use two different models.

    Returns:
        A tuple containing (model, vocabulary, decoder).

    Raises:
        Exception: If model initialization or decoder creation fails.
    """
    try:
        model = Small_LLM_Model()
        vocabulary = Vocab.from_model(model)
        decoder = Decoder(model=model, vocabulary=vocabulary)
        return model, decoder
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize language model and decoder: {str(e)}"
        ) from e


def process_prompts(
    prompts: List[Prompt],
    functions: List[FunctionDefinition],
    model: Small_LLM_Model,
    decoder: Decoder
) -> List[Dict[str, Any]]:
    """
    Process all prompts through the function-calling pipeline.

    For each prompt, selects the appropriate function and extracts
    parameter values, collecting all results.

    Args:
        prompts: List of Prompt objects to process.
        functions: List of FunctionDefinition objects to choose from.
        model: Model used for function selection.
        decoder: Decoder used for parameter extraction.

    Returns:
        A list of result dictionaries, each containing:
        - prompt: The original user prompt text
        - name: The selected function name
        - parameters: Extracted parameter values

    Note:
        Handles KeyboardInterrupt gracefully, saving partial results.
        A failure on a single prompt (bad generation, malformed output,
        tokenizer edge case, etc.) is logged and skipped rather than
        aborting the whole batch, so one bad prompt can never crash the
        program or discard already-computed results. An outer guard also
        catches any error occurring outside the per-prompt loop body
        (e.g. a malformed `prompts` iterable, or a printing/encoding
        failure), so partial results are preserved in that case too.
    """
    results: List[Dict[str, Any]] = []
    total_prompts = len(prompts)

    print(f"Processing {total_prompts} prompt(s)...")
    start_time = time.time()

    try:
        for i, prompt in enumerate(prompts, 1):
            try:
                # Validate prompt object
                if not hasattr(prompt, 'prompt') or not prompt.prompt:
                    raise ValueError(
                        f"Invalid prompt object at index {i-1}: "
                        "missing or empty 'prompt' attribute"
                    )

                # Select the appropriate function for this prompt
                try:
                    function = select_function(
                        prompt.prompt, functions, model
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Function selection failed: {str(e)}"
                    ) from e

                # Extract parameter values using constrained decoding
                try:
                    parameters = decoder.generate(prompt.prompt, function)
                except Exception as e:
                    raise RuntimeError(
                        f"Parameter extraction failed: {str(e)}"
                    ) from e

                # Validate extracted parameters
                if not isinstance(parameters, dict):
                    raise TypeError(
                        "Expected parameters to be a dict, "
                        f"got {type(parameters)}"
                    )

                # Collect results
                results.append({
                    "prompt": prompt.prompt,
                    "name": function.name,
                    "parameters": parameters
                })

                # Display progress
                print(f"[{i}/{total_prompts}] User request: {prompt.prompt}")
                print(f"    Function selected: {function.name}")
                print(f"    Parameters extracted: {parameters}\n")

            except Exception as e:
                # Never let a single prompt's failure abort the batch.
                print(
                    f"[{i}/{total_prompts}] Error processing prompt "
                    f"{e}",
                    file=sys.stderr
                )
                continue

    except KeyboardInterrupt:
        print(
            "\nInterrupted by user — saving "
            f"{len(results)} partial result(s)...",
            file=sys.stderr
        )

    except Exception as e:
        # Last-resort guard for anything outside the per-prompt try/except
        # (e.g. `prompts` isn't iterable as expected, a print/encoding
        # error, or any other unforeseen failure). Partial results
        # computed so far are preserved rather than lost.
        print(
            f"Unexpected error during batch processing: {e}. "
            f"Saving {len(results)} partial result(s)...",
            file=sys.stderr
        )

    # Display timing information
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60
    print(f"Processing time: {minutes}m {seconds:.1f}s")

    return results


def save_results(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save pipeline results to a JSON file.

    Args:
        results: List of result dictionaries to save.
        output_path: Path where the JSON file will be written.

    Raises:
        SystemExit: If the output file cannot be written, the output
                    directory cannot be created, or the results cannot
                    be serialized to JSON.
    """
    try:
        # Validate results before writing
        if not results:
            print(
                "Warning: No results to save. Output file will be empty.",
                file=sys.stderr
            )

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                raise OSError(
                    "Failed to create output directory "
                    f"'{output_dir}': {str(e)}"
                ) from e

        # Write results to JSON file
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except UnicodeEncodeError as e:
            raise ValueError(
                f"Failed to encode results as UTF-8: {str(e)}"
            ) from e
        except OSError as e:
            raise OSError(
                f"Failed to write to output file '{output_path}': {str(e)}"
            ) from e

        print(f"Done — {len(results)} result(s) written to '{output_path}'.")

    except (OSError, ValueError, TypeError) as e:
        # OSError: covers FileNotFoundError, PermissionError, directory
        # creation failures, and any other filesystem-related error.
        # TypeError: a non-JSON-serializable object ended up in results.
        # ValueError: includes UnicodeEncodeError and other encoding/
        # serialization issues raised by json.dump.
        print(f"Error saving results to '{output_path}': {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """
    Run the complete function-calling pipeline.

    Workflow:
    1. Parse command-line arguments
    2. Load function definitions and user prompts
    3. Initialize language models and decoder
    4. Process each prompt through function selection and parameter extraction
    5. Save all results to the output file

    This function wraps the entire pipeline in a top-level guard so that
    no exception, expected or not, can ever reach the caller as a raw
    traceback. Deliberate `sys.exit()` calls made by the helper functions
    above are allowed to propagate with their intended exit codes.
    """
    try:
        # Parse command-line arguments
        args = parse_arguments()

        # Validate input paths
        if not isinstance(args.functions_definition, str):
            raise TypeError(
                f"Expected functions_definition to be a string, "
                f"got {type(args.functions_definition)}"
            )
        if not isinstance(args.input, str):
            raise TypeError(
                f"Expected input to be a string, got {type(args.input)}"
            )
        if not isinstance(args.output, str):
            raise TypeError(
                f"Expected output to be a string, got {type(args.output)}"
            )

        # Load input data
        functions, prompts = load_input_data(
            args.functions_definition, args.input
        )

        # Initialize model and decoder
        try:
            model, decoder = initialize_models()
        except Exception as e:
            print(f"Error initializing language model: {e}", file=sys.stderr)
            sys.exit(1)

        # Process all prompts through the pipeline
        results = process_prompts(
            prompts, functions, model, decoder
        )

        # Save results to output file
        save_results(results, args.output)

    except SystemExit:
        # Re-raise intentional exits (e.g. sys.exit(1) from the helpers
        # above) unchanged, so their exit codes are preserved.
        raise
    except KeyboardInterrupt:
        # Interrupt during setup/teardown outside process_prompts'
        # own KeyboardInterrupt handling (e.g. during argument parsing
        # or model initialization).
        print("\nInterrupted by user before completion.", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        # Absolute last resort: nothing should ever escape main() as an
        # unhandled traceback.
        print(f"Fatal unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

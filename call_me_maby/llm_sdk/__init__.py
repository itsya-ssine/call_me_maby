# ABOUTME: LLM SDK for local model inference using Hugging Face transformers.
# ABOUTME: Provides Small_LLM_Model class for loading and running causal language models.

import time
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, logging
from huggingface_hub import hf_hub_download
import os


logging.set_verbosity_error()  # keep the console clean


class Small_LLM_Model:
    """Utility class wrapping a lightweight Hugging Face causal-LM for fast, low-memory experimentation.

    Parameters
    ----------
    model_name: str, default="Qwen/Qwen3-0.6B"
        Identifier of the model on the HF Hub.
    device: str | None, default=None
        Computation device. If *None* we automatically select ``mps`` when available on macOS,
        ``cuda`` when available, otherwise we fall back to ``cpu``.
    dtype: torch.dtype | None, default=None
        Numerical precision. When using a GPU or MPS we default to ``float16`` to keep memory
        usage reasonable; on CPU we keep ``float32`` for maximum compatibility.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self._model_name = model_name

        # Auto-select device with priority: mps > cuda > cpu
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self._device = device

        if dtype is None:
            dtype = torch.float16 if self._device in ["cuda", "mps"] else torch.float32
        self._dtype = dtype

        # --- load tokenizer & model -------------------------------------------------
        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self._tokenizer.pad_token_id is None:
            # ensure we have a pad token to keep batch helpers happy
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=trust_remote_code,
        )
        self._model.to(self._device)
        self._model.eval()

        # switch to inference-only mode
        for p in self._model.parameters():
            p.requires_grad = False


    def encode(self, text: str) -> torch.Tensor:
        """Tokenise *text* and return a 2-D ``input_ids`` tensor on the target device."""
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], device=self._device, dtype=torch.long)


    def decode(self, ids: torch.Tensor | list[int]) -> str:
        """Inverse of :py:meth:`encode`. Removes special tokens."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tokenizer.decode(ids, skip_special_tokens=True)


    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """
        Given a list of input token ids, return the raw logits (no softmax) for the next token.
        """
        input_tensor = torch.tensor([input_ids], device=self._device, dtype=torch.long)
        with torch.no_grad():
            out = self._model(input_ids=input_tensor)
        # Get logits for the last token in the sequence for the batch (batch size 1)
        logits = out.logits[0, -1].tolist()
        return [float(x) for x in logits]


    def get_path_to_vocab_file(self) -> str:
        vocab_file_name = self._tokenizer.vocab_files_names.get('vocab_file', "vocab.json")
        vocab_path = hf_hub_download(
            repo_id=self._model_name,
            filename=vocab_file_name
        )
        return vocab_path


    def get_path_to_merges_file(self) -> str:
        merges_file_name = self._tokenizer.vocab_files_names.get('merges_file', "merges.txt")
        merges_path = hf_hub_download(
            repo_id=self._model_name,
            filename=merges_file_name
        )
        return merges_path


    def get_path_to_tokenizer_file(self) -> str:
        tokenizer_file_name = self._tokenizer.vocab_files_names.get('tokenizer_file', "tokenizer.json")
        tokenizer_path = hf_hub_download(
            repo_id=self._model_name,
            filename=tokenizer_file_name
        )
        return tokenizer_path

if __name__ == "__main__":
    import json

    # Initialize the model
    model = Small_LLM_Model("Qwen/Qwen3-0.6B")

    # ============================================================
    # PART 1: EXPLORING THE VOCABULARY
    # ============================================================
    print("=" * 60)
    print("VOCABULARY EXPLORATION")
    print("=" * 60)

    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path) as f:
        vocab = json.load(f)

    print(f"Vocabulary size: {len(vocab)}")
    print(f"ID for 'the': {vocab.get('the')}")
    print(f"ID for 'cat': {vocab.get('cat')}")

    # Show some interesting tokens
    print("\nCommon tokens and their IDs:")
    sample_words = ['the', 'The', ' a', ' an', ' to', 'cat', 'dog', 'meaning', 'life']
    for word in sample_words:
        if word in vocab:
            print(f"  {word!r:15} → ID {vocab[word]}")

    # ============================================================
    # PART 2: TOKENIZATION DEMONSTRATION
    # ============================================================
    print("\n" + "=" * 60)
    print("TOKENIZATION")
    print("=" * 60)

    text = "The meaning of life is"
    input_ids = model.encode(text)
    print(f"Input text: {text!r}")
    print(f"Token IDs: {input_ids.tolist()}")

    # Show what each token represents
    print("\nIndividual tokens:")
    for i, token_id in enumerate(input_ids[0].tolist()):
        token_text = model.decode([token_id])
        print(f"  Position {i}: ID {token_id:5} → {token_text!r}")

    # ============================================================
    # PART 3: NEXT TOKEN PREDICTION (CORRECT VERSION)
    # ============================================================
    print("\n" + "=" * 60)
    print("NEXT TOKEN PREDICTION")
    print("=" * 60)

    # Get the current token IDs as a flat list
    current_ids = input_ids[0].tolist()
    print(f"Prompt: {model.decode(current_ids)!r}")

    # Get logits for the next token
    logits = model.get_logits_from_input_ids(current_ids)
    probs = torch.softmax(torch.tensor(logits), dim=-1)

    # Get top 10 most likely next tokens
    top_k = 10
    top_probs, top_indices = torch.topk(probs, k=top_k)

    print(f"\nTop {top_k} most likely next tokens:")
    print("-" * 40)
    for i, (prob, token_id) in enumerate(zip(top_probs, top_indices)):
        token_id = token_id.item()
        
        # Decode just this single token
        single_token_text = model.decode([token_id])
        
        # Build the full sequence with this token and decode it
        full_sequence = current_ids + [token_id]
        full_text = model.decode(full_sequence)
        
        print(f"{i+1:2}. {single_token_text!r:15} ({prob*100:5.2f}%) → {full_text!r}")

    # ============================================================
    # PART 4: SIMPLE AUTOREGRESSIVE GENERATION
    # ============================================================
    print("\n" + "=" * 60)
    print("AUTOREGRESSIVE GENERATION")
    print("=" * 60)

    def generate_text(model, prompt, max_new_tokens=10, temperature=1.0, top_k=50):
        """
        Simple autoregressive text generation.
        
        Args:
            model: Small_LLM_Model instance
            prompt: Input text string
            max_new_tokens: How many tokens to generate
            temperature: Controls randomness (1.0=normal, <1.0=more focused, >1.0=more random)
            top_k: Only sample from top k tokens
        """
        # Encode the prompt
        input_ids = model.encode(prompt)
        generated_ids = input_ids[0].tolist()
        
        print(f"Prompt: {prompt!r}")
        print(f"Generating {max_new_tokens} tokens...")
        print("-" * 40)
        
        for step in range(max_new_tokens):
            # Get predictions
            logits = model.get_logits_from_input_ids(generated_ids)
            logits_tensor = torch.tensor(logits)
            
            # Apply temperature
            logits_tensor = logits_tensor / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(logits_tensor, k=min(top_k, len(logits_tensor)))
                filtered_logits = torch.full_like(logits_tensor, float('-inf'))
                filtered_logits[top_k_indices] = top_k_values
                logits_tensor = filtered_logits
            
            # Convert to probabilities
            probs = torch.softmax(logits_tensor, dim=-1)
            
            # Sample from the distribution
            next_token_id = torch.multinomial(probs, num_samples=1).item()
            
            # Add to generated sequence
            generated_ids.append(next_token_id)
            
            # Show what we generated at each step
            token_text = model.decode([next_token_id])
            current_text = model.decode(generated_ids)
            print(f"Step {step+1}: Generated token {token_text!r:15} → {current_text!r}")
            
            # Stop if we hit EOS token
            if next_token_id == model._tokenizer.eos_token_id:
                print("(EOS token reached)")
                break
        
        return model.decode(generated_ids)

    # Test with different prompts
    prompts = [
        "The meaning of life is",
        "Once upon a time",
        "Python is a programming",
    ]

    for prompt in prompts:
        print(f"\n{'='*60}")
        generated = generate_text(model, prompt, max_new_tokens=5, temperature=0.8, top_k=50)
        print(f"\nFinal generated text: {generated!r}")

    # ============================================================
    # PART 5: EXPLORING THE MERGES FILE
    # ============================================================
    print("\n" + "=" * 60)
    print("BPE MERGES EXPLORATION")
    print("=" * 60)

    merges_path = model.get_path_to_merges_file()
    with open(merges_path) as f:
        merges_lines = f.read().splitlines()

    # Skip the version header
    actual_merges = [line for line in merges_lines if not line.startswith('#')]
    print(f"Total lines in merges file: {len(merges_lines)}")
    print(f"Actual merge rules: {len(actual_merges)}")

    print("\nFirst 10 merge rules (fundamental building blocks):")
    for i, merge in enumerate(actual_merges[:10]):
        print(f"  {i+1:4}. {merge!r}")

    print("\nLast 10 merge rules (complex, rare combinations):")
    for i, merge in enumerate(actual_merges[-10:]):
        print(f"  {len(actual_merges)-9+i:4}. {merge!r}")

    # ============================================================
    # PART 6: TOKENIZER CONFIGURATION
    # ============================================================
    print("\n" + "=" * 60)
    print("TOKENIZER CONFIGURATION")
    print("=" * 60)

    tokenizer_path = model.get_path_to_tokenizer_file()
    print(f"Full tokenizer config at: {tokenizer_path}")

    # Show model information
    print(f"\nModel name: {model._model_name}")
    print(f"Device: {model._device}")
    print(f"Data type: {model._dtype}")
    print(f"Number of parameters: {sum(p.numel() for p in model._model.parameters()):,}")
    print(f"Model memory (approx): {sum(p.numel() * p.element_size() for p in model._model.parameters()) / 1024**3:.2f} GB")

    # ============================================================
    # PART 7: ADVANCED: ANALYZING ATTENTION PATTERNS
    # ============================================================
    print("\n" + "=" * 60)
    print("LOGITS ANALYSIS")
    print("=" * 60)

    # Compare predictions for different prompts
    test_prompts = [
        "The cat sat on the",
        "The dog sat on the",
        "The bird sat on the",
    ]

    print("Comparing predictions for similar prompts:")
    print("-" * 40)

    for prompt in test_prompts:
        ids = model.encode(prompt)[0].tolist()
        logits = model.get_logits_from_input_ids(ids)
        probs = torch.softmax(torch.tensor(logits), dim=-1)
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probs, k=3)
        
        print(f"\nPrompt: {prompt!r}")
        for prob, token_id in zip(top_probs, top_indices):
            token_text = model.decode([token_id.item()])
            full_text = model.decode(ids + [token_id.item()])
            print(f"  → {token_text!r:12} ({prob*100:5.2f}%) : {full_text!r}")

    print("\n" + "=" * 60)
    print("Demo complete!")
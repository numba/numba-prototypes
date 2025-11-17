#!/usr/bin/env python3
"""
Test script for Llama3 NumPy implementation with golden reference comparison only.
"""

import os
import sys
import time
import json
import numpy as np

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ModelArgs
from tokenizer import Tokenizer
from utils import load_parameters


def load_golden_output(filename="golden_output.json"):
    """Load golden output from JSON file."""
    with open(filename, "r") as f:
        return json.load(f)


def compare_outputs(golden_output, test_output):
    """Compare golden output with test output bit-by-bit."""
    print("Comparing outputs bit-by-bit...")

    # Compare token IDs
    golden_tokens = golden_output["token_ids"]
    test_tokens = test_output["token_ids"]

    if golden_tokens == test_tokens:
        print("Token IDs: MATCH")
    else:
        print("Token IDs: MISMATCH")
        print(f"Golden: {golden_tokens}")
        print(f"Test:   {test_tokens}")
        return False

    # Compare generated text
    golden_text = golden_output["generated_text"]
    test_text = test_output["generated_text"]

    if golden_text == test_text:
        print("Generated text: MATCH")
    else:
        print("Generated text: MISMATCH")
        print(f"Golden: '{golden_text}'")
        print(f"Test:   '{test_text}'")
        return False

    print("All comparisons: PASSED")
    return True


def generate_test_output(model, tokenizer, llama_generate, prompt, max_tokens):
    input_ids = np.array([tokenizer.encode(prompt)])
    generated_tokens = []
    token_ids = []
    for id_val in llama_generate(model, input_ids, max_tokens):
        output_id = id_val[0].tolist()
        if output_id[-1] in [tokenizer.eos_id, tokenizer.bos_id]:
            break
        generated_tokens.append(output_id[-1])
        token_ids.append(output_id[-1])
    generated_text = tokenizer.decode(generated_tokens)
    full_text = prompt + generated_text
    return {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "token_ids": token_ids,
        "generated_text": generated_text,
        "full_text": full_text,
        "num_tokens": len(generated_tokens),
    }


def test_llama3_against_golden():
    """Test the Llama3 implementation against golden reference only."""
    model_path = "./stories15M.model.npz"
    tokenizer_path = "./tokenizer.model.np"
    golden_file = "golden_output.json"

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return False
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer file not found: {tokenizer_path}")
        return False
    if not os.path.exists(golden_file):
        print(f"Golden output file not found: {golden_file}")
        print("Please generate the golden output before running this test.")
        return False

    print("Model, tokenizer, and golden output files found")

    args = ModelArgs()
    print(f"Using precision: {args.dtype}")
    tokenizer = Tokenizer(tokenizer_path)
    print("Tokenizer loaded successfully")
    from llama3 import llama_init, llama_generate

    model = llama_init(model_path, args)
    print("Model loaded successfully")

    # Load golden output
    golden_output = load_golden_output(golden_file)
    print(f"Loaded golden output for prompt: '{golden_output['prompt']}'")
    np.random.seed(42)
    test_output = generate_test_output(
        model,
        tokenizer,
        llama_generate,
        prompt=golden_output["prompt"],
        max_tokens=golden_output["max_tokens"],
    )
    print("Comparing to golden output...")
    golden_test_passed = compare_outputs(golden_output, test_output)
    if golden_test_passed:
        print("Golden reference test: PASSED")
    else:
        print("Golden reference test: FAILED")
    return golden_test_passed


if __name__ == "__main__":
    print("Testing Llama3 NumPy Implementation (golden reference only)")
    print("-" * 40)
    success = test_llama3_against_golden()
    print("-" * 40)
    if success:
        print("All tests passed")
        sys.exit(0)
    else:
        print("Tests failed")
        sys.exit(1)

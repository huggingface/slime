"""Prepare the Nemotron competitive coding dataset for SLIME.

Replicates exactly the dataset loading and filtering from:
  prime-rl/benchmark_envs/nemotron_env/nemotron_env/env.py
"""

import json
import os

from datasets import load_dataset


def prepare_dataset():
    # 1. Load dataset (same split as prime-rl)
    dataset = load_dataset("nvidia/Nemotron-RL-coding-competitive_coding", split="train")

    # 2. Filter to rows with valid unit tests (exact same filter as prime-rl)
    dataset = dataset.filter(
        lambda ex: isinstance(ex.get("verifier_metadata"), dict) and isinstance(ex["verifier_metadata"].get("unit_tests"), dict) and len(ex["verifier_metadata"]["unit_tests"].get("inputs", [])) > 0 and len(ex["verifier_metadata"]["unit_tests"]["inputs"]) == len(ex["verifier_metadata"]["unit_tests"].get("outputs", []))
    )

    # 3. Format examples (exact same logic as prime-rl's format_example)
    out_path = os.path.join(os.path.dirname(__file__), "nemotron_rl.jsonl")
    count = 0
    with open(out_path, "w") as f:
        for row in dataset:
            # Extract prompt from responses_create_params (same as prime-rl)
            prompt = row["responses_create_params"]["input"][0]["content"]
            parts = prompt.split("```python\n# Your code here\n```")
            if len(parts) > 1:
                prompt = parts[-1].strip()
            else:
                prompt = prompt.strip()

            # Format prompt identically to prime-rl nemotron_env
            formatted_prompt = f"You are an expert Python programmer.\n\n{prompt}\n\nPlease write a python code to solve the problem and use the execute_python_code tool to test it."

            # Store unit_tests in metadata (inputs/outputs for stdin/stdout testing)
            unit_tests = row["verifier_metadata"]["unit_tests"]
            item = {
                "prompt": formatted_prompt,
                "metadata": {
                    "unit_tests": {
                        "inputs": unit_tests["inputs"],
                        "outputs": unit_tests["outputs"],
                    },
                },
            }
            f.write(json.dumps(item) + "\n")
            count += 1

    print(f"Prepared {count} Nemotron competitive coding examples at {out_path}")


if __name__ == "__main__":
    prepare_dataset()

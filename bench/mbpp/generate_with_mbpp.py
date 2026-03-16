from __future__ import annotations

import asyncio
import copy
import os
import re
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from slime.utils.types import Sample

SYSTEM_PROMPT = """You are an expert Python programmer.
You must output your final code strictly enclosed in <code> and </code> tags.
Do not use standard markdown code blocks."""


def run_code_in_sandbox(code: str, test_list: list[str], timeout: int = 3):
    # Combine the generated code and the tests
    full_code = code + "\n\n" + "\n".join(test_list)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full_code)
        temp_path = f.name

    try:
        # Note: In a production environment, this should use a secure sandbox (e.g., Docker, seccomp).
        # For benchmarking purposes, we use a basic subprocess.
        result = subprocess.run([sys.executable, temp_path], capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return True, "Tests passed."
        else:
            # Truncate stderr to prevent context bloat
            return False, result.stderr[-2000:]
    except subprocess.TimeoutExpired:
        return False, "Execution timeout."
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


async def generate(args, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Multi-turn generation loop that provides test feedback to the model."""
    from slime.rollout.sglang_rollout import GenerateState
    from slime.rollout.sglang_rollout import generate as default_generate
    from slime.utils.types import Sample

    max_turns = getattr(args, "max_turns", 3)
    state = GenerateState(args)
    tokenizer = state.tokenizer

    test_list = sample.metadata.get("test_list", []) if sample.metadata else []

    for turn in range(max_turns):
        prev_resp_len = len(sample.response)

        if sample.metadata is None:
            sample.metadata = {}

        # 1. Apply system prompt chat template on the first turn
        if turn == 0 and not sample.metadata.get("_is_templated", False):
            if isinstance(sample.prompt, str):
                messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": sample.prompt}]
                sample.prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            sample.metadata["_is_templated"] = True

        # Deepcopy to prevent mutating max_new_tokens for the whole group reference
        current_sampling_params = copy.deepcopy(sampling_params)

        # 2. Generate code using the default SGLang runner
        sample = await default_generate(args, sample, current_sampling_params)

        if sample.status not in [Sample.Status.PENDING, Sample.Status.COMPLETED]:
            break

        new_text = sample.response[prev_resp_len:]

        # 3. Extract code block robustly
        code_blocks = re.findall(r"<code>(.*?)</code>", new_text, re.DOTALL)
        if code_blocks:
            code = code_blocks[-1].strip()
        else:
            unclosed_blocks = re.findall(r"<code>(.*)", new_text, re.DOTALL)
            if unclosed_blocks:
                code = unclosed_blocks[-1].strip()
            else:
                legacy_blocks = re.findall(r"```(?:python)?\n(.*?)\n```", new_text, re.DOTALL)
                code = legacy_blocks[-1].strip() if legacy_blocks else new_text.strip()

        passed, feedback = await asyncio.to_thread(run_code_in_sandbox, code, test_list)

        if sample.metadata is None:
            sample.metadata = {}
        sample.metadata["final_passed"] = passed

        # Check stop conditions
        if passed or turn == max_turns - 1:
            if passed:
                sample.status = Sample.Status.COMPLETED
            break

        # 5. Provide environment feedback for the next turn
        prompt_feedback = f"\n<environment_feedback>\nExecution failed with error:\n{feedback}\nPlease fix the code and try again.\n</environment_feedback>\n"
        feedback_ids = tokenizer.encode(prompt_feedback, add_special_tokens=False)

        if sample.response_length + len(feedback_ids) >= sampling_params.get("max_new_tokens", 4096):
            sample.status = Sample.Status.TRUNCATED
            break

        sample.tokens.extend(feedback_ids)
        sample.response += prompt_feedback
        sample.response_length += len(feedback_ids)

        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs.extend([0.0] * len(feedback_ids))

        if sample.loss_mask is not None:
            sample.loss_mask.extend([0] * len(feedback_ids))

        # Reset the status to PENDING so default_generate accepts it in the next turn
        sample.status = Sample.Status.PENDING

    return sample


async def reward_func(args, sample: Sample) -> float:
    """Binary reward: 1.0 if the final tests passed, 0.0 otherwise."""
    passed = sample.metadata.get("final_passed", False) if sample.metadata else False
    return 1.0 if passed else 0.0


if __name__ == "__main__":

    def simple_test():
        test_code = """
def reverse_words(s):
    return " ".join(s.split()[::-1])
"""
        test_list = ['assert reverse_words("python program")==("program python")', 'assert reverse_words("java language")==("language java")', 'assert reverse_words("indian man")==("man indian")']

        print("Running test code in sandbox...")
        passed, feedback = run_code_in_sandbox(test_code, test_list)
        print(f"Passed: {passed}")
        print(f"Feedback: {feedback}")

    simple_test()

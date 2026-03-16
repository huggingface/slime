"""Custom generate and reward functions for Nemotron competitive coding benchmark.

Replicates exactly the behavior of prime-rl/benchmark_envs/nemotron_env:
- Hermes-format tool calling with execute_python_code tool
- stdin/stdout code execution with identical normalize_output logic
- Up to max_turns=5 multi-turn tool calling
- Same stop conditions: tests_passed, no_tools_called, max_turns_reached
- Binary reward: 1.0 if all tests pass, 0.0 otherwise
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from slime.utils.types import Sample


TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "execute_python_code",
        "description": ("Execute python code to test against the hidden test cases. Provide the complete python code.\n\nargs:\n    code (str): The complete python code to execute."),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                }
            },
            "required": ["code"],
        },
    },
}


def normalize_output(out: str) -> str:
    return "\n".join(line.rstrip() for line in out.replace("\r\n", "\n").split("\n")).strip()


async def execute_python_code(code: str, unit_tests: dict) -> str:
    """Execute code against unit tests using stdin/stdout.

    Returns the same feedback strings for every case:
    - ``"Tests passed."`` on success
    - ``"Execution failed on test case {idx+1}..."`` on runtime error
    - ``"Test case {idx+1} failed..."`` on wrong output
    - ``"Execution timeout on test case {idx+1}."`` on timeout
    """
    inputs = unit_tests.get("inputs", [])
    outputs = unit_tests.get("outputs", [])

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = f.name

    try:
        for idx, (test_input, expected_output) in enumerate(zip(inputs, outputs)):
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                temp_path,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=test_input.encode("utf-8")),
                    timeout=10.0,
                )

                if proc.returncode != 0:
                    feedback = stderr.decode("utf-8")[-2000:]
                    return f"Execution failed on test case {idx + 1} with error:\n{feedback}\nPlease fix the code and try again."

                actual_output = normalize_output(stdout.decode("utf-8"))
                expected_output_clean = normalize_output(expected_output)

                if actual_output != expected_output_clean:
                    return f"Test case {idx + 1} failed.\nInput:\n{test_input}\nExpected Output:\n{expected_output_clean}\nActual Output:\n{actual_output}\nPlease fix the code and try again."

            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    # race condition occurs when the subprocess
                    pass  # process already exited
                await proc.communicate()
                return f"Execution timeout on test case {idx + 1}."

        return "Tests passed."
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _append_between_turn_tokens(
    sample: "Sample",
    tokenizer,
    tool_result: str,
    im_end_id: int,
    include_assistant_prompt: bool = True,
) -> None:
    """Append tool-response tokens between turns.

    Produces the exact Qwen-3 / Hermes chat-template sequence::

        <|im_end|>                  # already in sample.tokens (skipped)
        \\n<|im_start|>user
        <tool_response>
        {result}
        </tool_response><|im_end|>
        <|im_start|>assistant       # only when include_assistant_prompt=True

    All injected tokens get ``loss_mask=0`` and ``rollout_log_probs=0.0``
    so they are excluded from the policy-gradient loss.
    """
    # Check whether <|im_end|> is already the last token from generation
    has_im_end = sample.tokens and sample.tokens[-1] == im_end_id

    parts: list[str] = []
    if not has_im_end:
        parts.append("<|im_end|>")
    parts.append("\n<|im_start|>user\n<tool_response>\n")
    parts.append(tool_result)
    parts.append("\n</tool_response><|im_end|>\n")
    if include_assistant_prompt:
        parts.append("<|im_start|>assistant\n")

    between_text = "".join(parts)
    between_ids = tokenizer.encode(between_text, add_special_tokens=False)

    sample.tokens.extend(between_ids)
    sample.response += between_text
    sample.response_length += len(between_ids)

    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs.extend([0.0] * len(between_ids))

    if sample.loss_mask is not None:
        sample.loss_mask.extend([0] * len(between_ids))


# ---------------------------------------------------------------------------
# Custom generate function
# ---------------------------------------------------------------------------

# Regex to extract a Hermes-style tool call from the model response.
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


async def generate(args, sample: "Sample", sampling_params: dict[str, Any]) -> "Sample":
    """Multi-turn tool-calling generation for Nemotron competitive coding.

    1. **Turn 0** -- apply the Qwen-3 chat template with the
       ``execute_python_code`` tool definition (Hermes format).
    2. Generate via the default SGLang rollout.
    3. Parse ``<tool_call>`` tags from the response.
    4. Execute the extracted code against stdin/stdout unit tests
    5. Stitch the tool response back into the token stream using the
       correct chat-template framing and repeat.

    Stop conditions (matching prime-rl / verifiers):

    * **tests_passed** -- all unit tests pass (``has_final_env_response``)
    * **no_tools_called** -- the model responds without calling any tool
    * **max_turns_reached** -- turn count reaches ``max_turns`` (default 5)
    """
    from slime.rollout.sglang_rollout import GenerateState
    from slime.rollout.sglang_rollout import generate as default_generate
    from slime.utils.types import Sample

    max_turns = getattr(args, "max_turns", 5)
    state = GenerateState(args)
    tokenizer = state.tokenizer

    unit_tests = sample.metadata.get("unit_tests", {}) if sample.metadata else {}

    # Special-token ID used for between-turn stitching
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    for turn in range(max_turns):
        prev_resp_len = len(sample.response)

        if sample.metadata is None:
            sample.metadata = {}

        # -- Turn 0: apply chat template with tool definitions ---------------
        if turn == 0 and not sample.metadata.get("_is_templated", False):
            if isinstance(sample.prompt, str):
                messages = [{"role": "user", "content": sample.prompt}]
                sample.prompt = tokenizer.apply_chat_template(
                    messages,
                    tools=[TOOL_DEFINITION],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            sample.metadata["_is_templated"] = True

        # -- Generate --------------------------------------------------------
        current_sampling_params = copy.deepcopy(sampling_params)
        sample = await default_generate(args, sample, current_sampling_params)

        if sample.status not in [Sample.Status.PENDING, Sample.Status.COMPLETED]:
            break

        new_text = sample.response[prev_resp_len:]

        # -- Parse Hermes-format tool call -----------------------------------
        tool_call_match = _TOOL_CALL_RE.search(new_text)

        if not tool_call_match:
            # Stop condition: no_tools_called (model gave a direct answer)
            sample.status = Sample.Status.COMPLETED
            break

        try:
            tool_call_data = json.loads(tool_call_match.group(1))
            tool_name = tool_call_data.get("name", "")
            tool_args = tool_call_data.get("arguments", {})
            if isinstance(tool_args, str):
                tool_args = json.loads(tool_args)
        except (json.JSONDecodeError, TypeError):
            # Malformed tool call -- equivalent to no_tools_called
            sample.status = Sample.Status.COMPLETED
            break

        if tool_name != "execute_python_code" or "code" not in tool_args:
            sample.status = Sample.Status.COMPLETED
            break

        code = tool_args["code"]

        result = await execute_python_code(code, unit_tests)
        tests_passed = result == "Tests passed."

        sample.metadata["tests_passed"] = tests_passed

        if tests_passed:
            # Stop condition: has_final_env_response
            _append_between_turn_tokens(sample, tokenizer, result, im_end_id, include_assistant_prompt=False)
            sample.status = Sample.Status.COMPLETED
            break

        if turn == max_turns - 1:
            # Stop condition: max_turns_reached
            _append_between_turn_tokens(sample, tokenizer, result, im_end_id, include_assistant_prompt=False)
            break

        # -- Append tool-response tokens for next turn -----------------------
        _append_between_turn_tokens(sample, tokenizer, result, im_end_id, include_assistant_prompt=True)

        # Check remaining token budget
        if sample.response_length >= sampling_params.get("max_new_tokens", 4096):
            sample.status = Sample.Status.TRUNCATED
            break

        # Reset status so default_generate accepts the sample on the next turn
        sample.status = Sample.Status.PENDING

    return sample


async def reward_func(args, sample: "Sample") -> float:
    """Binary reward: 1.0 if all unit tests passed, 0.0 otherwise.

    Identical to prime-rl nemotron_env's ``tests_passed_reward`` rubric.
    """
    passed = sample.metadata.get("tests_passed", False) if sample.metadata else False
    return 1.0 if passed else 0.0

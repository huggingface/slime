#!/bin/bash
# run-qwen3-4b-mbpp.sh
# Benchmark script for SLIME on MBPP using 2 train + 6 infer GPUs (Disaggregated mode)

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PYTHON="python3"

MODEL_ARGS=(
   --swiglu
   --num-layers 36
   --hidden-size 2560
   --ffn-hidden-size 9728
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --use-rotary-position-embeddings
   --disable-bias-linear
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   --rotary-base 1000000
   --vocab-size 151936
   --kv-channels 128
   --qk-layernorm
)


# --- Auto-Convert HF Checkpoint to Megatron ---
HF_PATH="/fsx/amine_dirhoussi/bench_rl/Qwen3-4B"
SAVE_PATH="/fsx/amine_dirhoussi/bench_rl/Qwen3-4B_torch_dist"

# Check if the conversion was already done by looking for the Megatron directory
if [ ! -d "$SAVE_PATH" ] || [ -z "$(ls -A $SAVE_PATH)" ]; then
    echo "Megatron weights not found in $SAVE_PATH. Starting conversion..."

    # We need to temporarily source the default qwen3-4B args for the conversion script
    # to know the architecture parameters
    source ${SCRIPT_DIR}/../scripts/models/qwen3-4B.sh

    PYTHONPATH="${SCRIPT_DIR}/..:${SCRIPT_DIR}:/root/Megatron-LM/" $PYTHON ${SCRIPT_DIR}/../tools/convert_hf_to_torch_dist.py \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint ${HF_PATH} \
        --save ${SAVE_PATH}

    echo "Conversion complete."
else
    echo "Megatron weights already exist at $SAVE_PATH. Skipping conversion."
fi

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


# Launch Ray head on the single node with 8 GPUs (locally)
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${SCRIPT_DIR}/..:${SCRIPT_DIR}:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

PROMPT_SET="${SCRIPT_DIR}/mbpp_rl.jsonl"

# Check if dataset exists, if not run prepare script
if [ ! -f "$PROMPT_SET" ]; then
    "$PYTHON" "${SCRIPT_DIR}/prepare_mbpp.py"
fi


CKPT_ARGS=(
   --hf-checkpoint /fsx/amine_dirhoussi/bench_rl/Qwen3-4B
   --ref-load /fsx/amine_dirhoussi/bench_rl/Qwen3-4B_torch_dist
   --load /fsx/amine_dirhoussi/bench_rl/Qwen3-4B_torch_dist
   --save /fsx/amine_dirhoussi/bench_rl/Qwen3-4B_slime/
   --no-load-optim
   --no-load-rng
   --save-interval 20
)

ROLLOUT_ARGS=(
   --rollout-function-path slime.rollout.sglang_rollout.generate_rollout
   --custom-generate-function-path bench.generate_with_mbpp.generate
   --custom-rm-path bench.generate_with_mbpp.reward_func
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --metadata-key metadata
   --apply-chat-template
   --rollout-shuffle
   --num-epoch 2
   --rollout-batch-size 16
   --n-samples-per-prompt 16
   --rollout-max-response-len 4096
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data
)

EVAL_ARGS=(
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --weight-decay 0.1
)

WANDB_ARGS=(
)

SGLANG_ARGS=(
    # tp_size
   --rollout-num-gpus-per-engine 1
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# Submit Ray job. Notice:
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- "$PYTHON" ${SCRIPT_DIR}/../train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 2 \
   --rollout-num-gpus 4 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}

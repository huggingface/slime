#!/bin/bash
# run-qwen3-4b-nemotron.sh
# Benchmark script for SLIME on Nemotron competitive coding
# using 2 train + 6 infer GPUs (Disaggregated mode)
#
# Replicates the exact setup from prime-rl/bench/rl_nemotron_cp.toml:
#   - Qwen3-4B model, TP=2
#   - 16 rollouts per prompt, batch_size=128 (8 prompts/batch)
#   - 24576 max context/response length
#   - lr=1e-6, weight_decay=0.1
#   - Hermes tool calling with execute_python_code
#   - max_turns=5 multi-turn tool calling

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
BASE_PATH="/fsx/amine_dirhoussi/bench_rl"
HF_PATH="${BASE_PATH}/Qwen3-4B"
SAVE_PATH="${BASE_PATH}/Qwen3-4B_torch_dist"
SLIME_SAVE_PATH="${BASE_PATH}/Qwen3-4B_slime_nemotron/"

# Check if the conversion was already done by looking for the Megatron directory
if [ ! -d "$SAVE_PATH" ] || [ -z "$(ls -A $SAVE_PATH)" ]; then
    echo "Megatron weights not found in $SAVE_PATH. Starting conversion..."

    # We need to temporarily source the default qwen3-4B args for the conversion script
    # to know the architecture parameters
    source ${SCRIPT_DIR}/../../scripts/models/qwen3-4B.sh

    PYTHONPATH="${SCRIPT_DIR}/../..:${SCRIPT_DIR}/..:/root/Megatron-LM/" $PYTHON ${SCRIPT_DIR}/../../tools/convert_hf_to_torch_dist.py \
        ${MODEL_ARGS[@]} \
        --hf-checkpoint ${HF_PATH} \
        --save ${SAVE_PATH}

    echo "Conversion complete."
else
    echo "Megatron weights already exist at $SAVE_PATH. Skipping conversion."
fi

export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


# Launch Ray head on the single node with 8 GPUs (2 train + 6 infer)
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 6 --disable-usage-stats

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${SCRIPT_DIR}/../..:${SCRIPT_DIR}/..:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

PROMPT_SET="${SCRIPT_DIR}/nemotron_rl.jsonl"

# Check if dataset exists, if not run prepare script
if [ ! -f "$PROMPT_SET" ]; then
    "$PYTHON" "${SCRIPT_DIR}/prepare_nemotron.py"
fi


CKPT_ARGS=(
   --hf-checkpoint ${HF_PATH}
   --ref-load ${SAVE_PATH}
   --load ${SAVE_PATH}
   --save ${SLIME_SAVE_PATH}
   --no-load-optim
   --no-load-rng
   --save-interval 10
)

ROLLOUT_ARGS=(
   --rollout-function-path slime.rollout.sglang_rollout.generate_rollout
   --custom-generate-function-path bench.nemotron.generate_with_nemotron.generate
   --custom-rm-path bench.nemotron.generate_with_nemotron.reward_func
   --prompt-data ${PROMPT_SET}
   --input-key prompt
   --metadata-key metadata

   --rollout-shuffle
   --num-epoch 1
   --rollout-batch-size 8
   --n-samples-per-prompt 16
   --rollout-max-context-len 24576
   --rollout-max-response-len 24576
   --rollout-temperature 1

   --global-batch-size 128
   --balance-data
)

EVAL_ARGS=(
)

PERF_ARGS=(
    # DP = world_size // (TP + PP + CP)
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1 # each layer independently checkpointed (lower peak backward memory)

    # dynamic microbatch based on max tokens per gpus
    --use-dynamic-batch-size
    --max-tokens-per-gpu 9216

    # SLIME doesn't chunk the linear projection !!
    --log-probs-chunk-size 1024
    # Save memory
    --recompute-loss-function
)

GRPO_ARGS=(
   --advantage-estimator grpo
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --weight-decay 0.1

   # 24k support
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
   --optimizer-cpu-offload
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-bench-qwen3-4b
   --wandb-group qwen3-4b-nemotron
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

# Submit Ray job
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- "$PYTHON" ${SCRIPT_DIR}/../../train_async.py \
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

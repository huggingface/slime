import wandb
import pandas as pd
import numpy as np
import json
from scipy.stats import linregress

RUN_PATH = "slime-bench-qwen3-4b/runs/3uw3gu4b"
OUTPUT_PREFIX = "slime_metrics"


def _compute_pipeline_bubble(df):
    """Compute pure pipeline bubble ratio, excluding weight sync time.

    The logged ``perf/wait_time_ratio`` equals ``train_wait_time / step_time``,
    but ``train_wait_time`` includes ``update_weights_time`` because weight sync
    runs while the ``train_wait`` timer is still accumulating.  We subtract it
    to isolate true idle time where the trainer does no useful work.

    Falls back to the raw ``wait_time_ratio`` when the columns needed for the
    corrected calculation are not all present.
    """
    needed = ["perf/train_wait_time", "perf/update_weights_time", "perf/step_time"]
    if all(c in df.columns for c in needed):
        bubble_df = df.dropna(subset=needed)
        if len(bubble_df) > 0:
            pure_wait = (bubble_df["perf/train_wait_time"] - bubble_df["perf/update_weights_time"]).clip(lower=0)
            return (pure_wait / bubble_df["perf/step_time"]).mean()
    # Fallback: use the raw ratio if corrected columns are unavailable.
    if "perf/wait_time_ratio" in df.columns:
        return df["perf/wait_time_ratio"].mean()
    return None


def main():
    api = wandb.Api()
    run = api.run(RUN_PATH)

    # Relevant keys for the benchmark plan
    keys_to_fetch = [
        "perf/step_time",
        "perf/train_time",
        "perf/tokens_per_gpu_per_sec",
        "perf/effective_tokens_per_gpu_per_sec",
        "perf/actor_train_tok_per_s",
        "perf/actor_train_time",
        "perf/train_wait_time",
        "perf/update_weights_time",
        "perf/wait_time_ratio",
        "perf/rollout_time",
        "rollout/raw_reward",
        "train/ppo_kl",
        "train/train_rollout_logprob_abs_diff",
        "_runtime",
    ]

    print(f"Fetching history for run: {RUN_PATH}...")

    # Download metrics history
    history_df = run.history(samples=100000, pandas=True)
    system_metrics = run.history(stream="events", pandas=True)

    # Since metrics are logged asynchronously from different processes (rollout vs train),
    # we shouldn't drop rows based on a single column. We compute mean() per column,
    # which naturally ignores NaNs.
    df_clean = history_df

    # Extract rollout_num_gpus from config (default 6 if not found)
    rollout_num_gpus = run.config.get("rollout_num_gpus", 6)

    # Extract data_parallel_size to scale per-rank train tokens to global.
    # perf/actor_train_tok_per_s is logged from a single DP rank, so its token
    # count must be multiplied by dp_size to match the global gen token count.
    data_parallel_size = run.config.get("data_parallel_size", 1)

    # ---- Throughput Calculations (T1, T2, T3) ----

    # Generation metrics
    if "perf/tokens_per_gpu_per_sec" in df_clean.columns and "perf/rollout_time" in df_clean.columns:
        gen_df = df_clean.dropna(subset=["perf/tokens_per_gpu_per_sec", "perf/rollout_time"])
        total_gen_tokens = (gen_df["perf/tokens_per_gpu_per_sec"] * gen_df["perf/rollout_time"] * rollout_num_gpus).sum()

        t3_gen_tokens_per_sec_mean = (gen_df["perf/tokens_per_gpu_per_sec"] * rollout_num_gpus).mean()

        if "perf/effective_tokens_per_gpu_per_sec" in gen_df.columns:
            total_effective_gen_tokens = (gen_df["perf/effective_tokens_per_gpu_per_sec"] * gen_df["perf/rollout_time"] * rollout_num_gpus).sum()
            t3_gen_effective_tokens_per_sec_mean = (gen_df["perf/effective_tokens_per_gpu_per_sec"] * rollout_num_gpus).mean()
        else:
            total_effective_gen_tokens = 0
            t3_gen_effective_tokens_per_sec_mean = None
    else:
        total_gen_tokens = 0
        total_effective_gen_tokens = 0
        t3_gen_tokens_per_sec_mean = None
        t3_gen_effective_tokens_per_sec_mean = None

    # Training metrics
    if "perf/actor_train_tok_per_s" in df_clean.columns and "perf/actor_train_time" in df_clean.columns and "perf/train_time" in df_clean.columns and "perf/step_time" in df_clean.columns:
        train_df = df_clean.dropna(subset=["perf/actor_train_tok_per_s", "perf/actor_train_time", "perf/train_time", "perf/step_time"])
        # actor_train_tok_per_s is logged from one DP rank; scale to global token count.
        total_train_tokens = (train_df["perf/actor_train_tok_per_s"] * train_df["perf/actor_train_time"]).sum() * data_parallel_size
        total_step_time = train_df["perf/step_time"].sum()
        total_train_time = train_df["perf/train_time"].sum()

        # T2: overall trainer side throughput
        if total_train_time > 0:
            t2_train_tokens_per_sec_mean = total_train_tokens / total_train_time
        else:
            t2_train_tokens_per_sec_mean = None
    else:
        total_train_tokens = 0
        total_step_time = 1  # prevent division by zero
        t2_train_tokens_per_sec_mean = None

    # T1: End-to-end tokens/second
    if total_step_time > 0 and total_gen_tokens > 0:
        t1_e2e_tokens_per_sec_mean = (total_gen_tokens + total_train_tokens) / total_step_time
        t1_e2e_effective_tokens_per_sec_mean = (total_effective_gen_tokens + total_train_tokens) / total_step_time
    else:
        t1_e2e_tokens_per_sec_mean = None
        t1_e2e_effective_tokens_per_sec_mean = None

    # L2: Reward trend (slope of reward over steps)
    # Use raw_reward instead of rewards, because rollout/rewards may contain
    # group-normalized values (GRPO/GSPO) that are ~0 by construction.
    if "rollout/raw_reward" in df_clean.columns:
        df_rewards = df_clean.dropna(subset=["rollout/raw_reward"])
        if len(df_rewards) > 1:
            # We use the index (which is just the wandb row index) or _step
            slope, _, _, _, _ = linregress(df_rewards.index, df_rewards["rollout/raw_reward"])
        else:
            slope = None
    else:
        slope = None

    # L4: Importance Sampling Ratio Bound
    # logprob_abs_diff = |log(pi_train) - log(pi_rollout)| = |log(rho)|
    # exp(|log(rho)|) = max(rho, 1/rho)  -> This captures worst-case off-policy drift
    #
    # NOTE: The wandb value is already a per-sample mean of |delta log p| at each step.
    # We compute exp(mean(|delta|)) which, by Jensen's inequality (exp is convex),
    # is a lower bound on the true mean(exp(|delta|)).  This gives a conservative
    # estimate of the IS ratio bound.
    if "train/train_rollout_logprob_abs_diff" in df_clean.columns:
        is_ratio_bounds = np.exp(df_clean["train/train_rollout_logprob_abs_diff"].dropna())
        l4_is_ratio_mean = is_ratio_bounds.mean()
        l4_is_ratio_max = is_ratio_bounds.max()
    else:
        l4_is_ratio_mean = None
        l4_is_ratio_max = None

    # ---- System Metrics Processing ----
    # Find relevant columns dynamically
    gpu_util_cols = [c for c in system_metrics.columns if "system.gpu." in c and ".gpu/l:" in c]
    gpu_mem_cols = [c for c in system_metrics.columns if "system.gpu." in c and ".memoryAllocatedBytes/l:" in c]
    cpu_mem_cols = [c for c in system_metrics.columns if "system.memory_percent/l:" in c]

    # U1: GPU Utilization
    if gpu_util_cols:
        u1_gpu_util_mean = system_metrics[gpu_util_cols].mean().mean()
    else:
        u1_gpu_util_mean = None

    # U3: GPU Memory Peak (GB)
    if gpu_mem_cols:
        u3_gpu_mem_peak_gb = system_metrics[gpu_mem_cols].max().max() / (1024**3)
    else:
        u3_gpu_mem_peak_gb = None

    # U4: GPU Idle Time
    if gpu_util_cols:
        # Check if ANY gpu at a given timestamp is <= 1.0 (treating <=1% as idle to avoid noise)
        idle_mask = (system_metrics[gpu_util_cols] <= 1.0).any(axis=1)
        u4_gpu_idle_time_pct = idle_mask.mean() * 100
    else:
        u4_gpu_idle_time_pct = None

    # R1: CPU Memory Usage Peak (%)
    if cpu_mem_cols:
        r1_cpu_mem_peak_pct = system_metrics[cpu_mem_cols].max().max()
    else:
        r1_cpu_mem_peak_pct = None

    # Compute aggregates as per benchmark plan
    summary = {
        # 4.1 Throughput
        "T1_e2e_tokens_per_sec_mean": t1_e2e_tokens_per_sec_mean,
        "T1_e2e_effective_tokens_per_sec_mean": t1_e2e_effective_tokens_per_sec_mean,
        "T2_train_tokens_per_sec_mean": t2_train_tokens_per_sec_mean,
        "T3_gen_tokens_per_sec_mean": t3_gen_tokens_per_sec_mean,
        "T3_gen_effective_tokens_per_sec_mean": t3_gen_effective_tokens_per_sec_mean,
        "T5_steps_per_hour_mean": 3600 / df_clean["perf/step_time"].mean() if "perf/step_time" in df_clean.columns else None,
        "T6_wall_clock_time_100_steps": df_clean["_runtime"].dropna().iloc[-1] if "_runtime" in df_clean.columns and len(df_clean["_runtime"].dropna()) > 0 else None,
        # 4.2 Hardware Utilization
        "U1_gpu_utilization_mean_pct": u1_gpu_util_mean,
        "U3_gpu_memory_peak_gb": u3_gpu_mem_peak_gb,
        "U4_gpu_idle_time_pct": u4_gpu_idle_time_pct,
        # 4.3 Async Pipeline Efficiency
        "A1_weight_sync_latency_sec_mean": df_clean.get("perf/update_weights_time", pd.Series(dtype=float)).mean(),
        # A3: Pure pipeline bubble excluding weight sync.
        # perf/wait_time_ratio = train_wait_time / step_time, but train_wait_time
        # includes update_weights_time (weight sync runs while the train_wait
        # timer is accumulating). Subtract it to get true idle time.
        "A3_pipeline_bubble_ratio_mean": _compute_pipeline_bubble(df_clean),
        # 4.4 Multi-turn & Straggler
        "M2_rollout_time_sec_mean": df_clean.get("perf/rollout_time", pd.Series(dtype=float)).mean(),
        # 4.5 Learning Sanity Checks
        # Use raw_reward to avoid group-normalized values (GRPO/GSPO) that are ~0.
        "L1_reward_mean": df_clean.get("rollout/raw_reward", pd.Series(dtype=float)).mean(),
        "L1_reward_std": df_clean.get("rollout/raw_reward", pd.Series(dtype=float)).std(),
        "L2_reward_trend_slope": slope,
        "L3_kl_divergence_mean": df_clean.get("train/ppo_kl", pd.Series(dtype=float)).mean(),
        "L4_is_ratio_bound_mean": l4_is_ratio_mean,
        "L4_is_ratio_bound_max": l4_is_ratio_max,
        # 4.6 Resource
        "R1_cpu_memory_peak_pct": r1_cpu_mem_peak_pct,
    }

    # Save outputs
    summary_file = f"{OUTPUT_PREFIX}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Saved aggregated summary to {summary_file}")

    history_file = f"{OUTPUT_PREFIX}_history.csv"

    # Save only the available keys we care about
    available_keys = [k for k in keys_to_fetch if k in history_df.columns]
    history_df[available_keys].to_csv(history_file, index=False)
    print(f"Saved step-by-step history to {history_file}")

    # Generate Markdown Report
    markdown = f"""# SLIME Benchmark Metrics Report

## 4.1 Throughput

| Metric | Value | Description |
|---|---|---|
| **T1: End-to-end tokens/second (Mean)** | `{summary.get("T1_e2e_tokens_per_sec_mean", 0) or 0:.2f}` tok/s | Total tokens (generated + trained) / total wall-clock time |
| **T1: End-to-end effective tokens/second (Mean)** | `{summary.get("T1_e2e_effective_tokens_per_sec_mean", 0) or 0:.2f}` tok/s | Effective total tokens / total wall-clock time |
| **T2: Training tokens/second (Mean)** | `{summary.get("T2_train_tokens_per_sec_mean", 0) or 0:.2f}` tok/s | Tokens consumed by gradient steps per second |
| **T3: Generation tokens/second (Mean)** | `{summary.get("T3_gen_tokens_per_sec_mean", 0) or 0:.2f}` tok/s | Tokens produced by inference engine per second |
| **T3: Generation effective tokens/second (Mean)** | `{summary.get("T3_gen_effective_tokens_per_sec_mean", 0) or 0:.2f}` tok/s | Effective tokens produced by inference engine per second |
| **T5: Training steps/hour (Mean)** | `{summary.get("T5_steps_per_hour_mean", 0) or 0:.2f}` steps/h | Gradient update steps completed per hour |
| **T6: Total wall-clock time for 100 steps** | `{summary.get("T6_wall_clock_time_100_steps", 0) or 0:.2f}` seconds | Bottom-line runtime for the benchmark |

## 4.2 Hardware Utilization

| Metric | Value | Description |
|---|---|---|
| **U1: GPU utilization (Mean)** | `{summary.get("U1_gpu_utilization_mean_pct", 0) or 0:.2f}` % | Average SM utilization across all GPUs |
| **U3: GPU memory peak** | `{summary.get("U3_gpu_memory_peak_gb", 0) or 0:.2f}` GB | Max allocated memory observed on any single GPU |
| **U4: GPU idle time** | `{summary.get("U4_gpu_idle_time_pct", 0) or 0:.2f}` % | Fraction of wall-clock where at least one GPU has <= 1% SM utilization |

## 4.3 Async Pipeline Efficiency

| Metric | Value | Description |
|---|---|---|
| **A1: Weight sync latency (Mean)** | `{(summary.get("A1_weight_sync_latency_sec_mean", 0) or 0) * 1000:.4f}` ms | Wall-clock time to broadcast new weights to inference engine |
| **A3: Pipeline bubble (Mean)** | `{(summary.get("A3_pipeline_bubble_ratio_mean", 0) or 0) * 100:.2f}` % | Pure idle time (train_wait minus weight sync) / step time |

## 4.4 Multi-turn & Straggler

| Metric | Value | Description |
|---|---|---|
| **M2: Rollout completion time (Mean)** | `{summary.get("M2_rollout_time_sec_mean", 0) or 0:.2f}` seconds | Wall-clock from prompt dispatch to final reward received |

## 4.5 Learning Sanity Checks

| Metric | Value | Description |
|---|---|---|
| **L1: Reward (Mean)** | `{summary.get("L1_reward_mean", 0) or 0:.6f}` | Average reward across all rollouts per step |
| **L1: Reward (Std Dev)** | `{summary.get("L1_reward_std", 0) or 0:.6f}` | Standard deviation of reward per step |
| **L2: Reward trend (Slope)** | `{summary.get("L2_reward_trend_slope", 0) or 0:.2e}` | Linear regression slope of reward over training steps |
| **L3: KL divergence (Mean)** | `{summary.get("L3_kl_divergence_mean", 0) or 0:.6f}` nats | Between current policy and behavior policy |
| **L4: IS Ratio Bound (Mean)** | `{summary.get("L4_is_ratio_bound_mean", 0) or 0:.4f}` | Lower bound via `exp(mean(abs(log(pi_train) - log(pi_rollout))))` (Jensen) |
| **L4: IS Ratio Bound (Max)** | `{summary.get("L4_is_ratio_bound_max", 0) or 0:.4f}` | Worst-case off-policy drift bound |

## 4.6 Resource

| Metric | Value | Description |
|---|---|---|
| **R1: CPU memory usage peak** | `{summary.get("R1_cpu_memory_peak_pct", 0) or 0:.2f}` % | Peak host RAM usage percentage |
"""

    report_file = f"{OUTPUT_PREFIX}_report.md"
    with open(report_file, "w") as f:
        f.write(markdown)
    print(f"Saved markdown report to {report_file}")


if __name__ == "__main__":
    main()

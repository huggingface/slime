#!/bin/bash
#SBATCH --job-name=slime-qwen3-4b-mbpp
#SBATCH --partition=hopper-prod
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1999G
#SBATCH --time=1-0
#SBATCH -o /fsx/amine_dirhoussi/logs/%n-%j.out

# Ensure log directory exists
mkdir -p /fsx/amine_dirhoussi/logs/

echo "Starting SLIME MBPP benchmark container..."

# Execute the benchmark script inside the enroot container
srun \
    --container-image="registry.hpc-cluster-hopper.hpc.internal.huggingface.tech#library/slimerl:latest" \
    --container-mounts="/fsx:/fsx" \
    --container-workdir="/fsx/amine_dirhoussi/slime/bench" \
    bash -c "bash run-qwen3-4b-mbpp.sh"

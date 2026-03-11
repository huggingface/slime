#!/bin/bash

# Configuration
ENVIRONMENT_PATH="./env"
SGLANG_COMMIT="24c91001cf99ba642be791e099d358f4dfe955f5"
MEGATRON_COMMIT="3714d81d418c9f1bca4594fc35f9e8289f652862"

set -ex

# Determine SLIME_DIR and BASE_DIR
export SLIME_DIR=$(pwd)
export BASE_DIR=$(realpath ..)

# Detect Micromamba
if command -v micromamba >/dev/null 2>&1; then
    MAMBA_EXE=$(command -v micromamba)
elif [ -x "/fsx/amine_dirhoussi/bin/micromamba" ]; then
    MAMBA_EXE="/fsx/amine_dirhoussi/bin/micromamba"
else
    echo "micromamba not found at /fsx/amine_dirhoussi/bin/micromamba or in PATH."
    exit 1
fi

# Initialize micromamba for this shell session
eval "$($MAMBA_EXE shell hook --shell bash)"

# Create local environment if it doesn't exist
if [ ! -d "$ENVIRONMENT_PATH" ]; then
    echo "Creating environment in $ENVIRONMENT_PATH..."
    $MAMBA_EXE create -p "$ENVIRONMENT_PATH" python=3.12 pip -c conda-forge -y
fi

# Activating the environment using the path and setting Mamba's internal variables
# We use the prefix explicitly to avoid issues with named environments
micromamba activate "$SLIME_DIR/$ENVIRONMENT_PATH"

# Set CUDA_HOME to the environment's prefix if not already set
# This is often needed for compiling extensions like flash-attn or apex
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$LD_LIBRARY_PATH"

# install cuda 12.9 as it's the default cuda version for torch
echo "Installing CUDA, NCCL, and cuDNN via micromamba..."
$MAMBA_EXE install -p "$ENVIRONMENT_PATH" \
    cuda \
    cuda-nvtx \
    cuda-nvtx-dev \
    nccl \
    cudnn \
    -c nvidia/label/cuda-12.9.1 \
    -c nvidia \
    -c conda-forge \
    -y

# prevent installing cuda 13.0 for sglang
echo "Installing PyTorch and core dependencies..."
pip install cuda-python==13.1.0
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129

# install sglang
echo "Installing sglang..."
cd "$BASE_DIR"
if [ ! -d "sglang" ]; then
    git clone https://github.com/sgl-project/sglang.git
fi
cd sglang
git fetch origin
git checkout ${SGLANG_COMMIT}
# Install the python packages
pip install -e "python[all]"

pip install cmake ninja

# flash attn
# the newest version megatron supports is v2.7.4.post1
echo "Installing flash-attn (this may take a while)..."
MAX_JOBS=$(nproc) pip -v install flash-attn==2.7.4.post1 --no-build-isolation

echo "Installing RL and Megatron bridge dependencies..."
pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"
pip install flash-linear-attention==0.4.0

NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall
pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation
pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation

# megatron
echo "Installing Megatron-LM..."
cd "$BASE_DIR"
if [ ! -d "Megatron-LM" ]; then
    git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
fi
cd Megatron-LM/
git fetch origin
git checkout ${MEGATRON_COMMIT}
pip install -e .

# install slime and apply patches
echo "Installing slime..."
cd "$SLIME_DIR"
pip install -e .

# https://github.com/pytorch/pytorch/issues/168167
pip install nvidia-cudnn-cu12==9.16.0.29
pip install "numpy<2"

# apply patch
echo "Applying patches to dependencies..."
cd "$BASE_DIR/sglang"
git apply "$SLIME_DIR/docker/patch/v0.5.7/sglang.patch" || echo "sglang patch skip: already applied or incompatible"
cd "$BASE_DIR/Megatron-LM"
git apply "$SLIME_DIR/docker/patch/v0.5.7/megatron.patch" || echo "megatron patch skip: already applied or incompatible"

echo "--------------------------------------------------"
echo "Setup COMPLETED successfully."
echo "Environment location: $SLIME_DIR/env"
echo "To use this environment in the future, run:"
echo "  eval \"\$($MAMBA_EXE shell hook --shell bash)\""
echo "  micromamba activate $SLIME_DIR/env"
echo "--------------------------------------------------"

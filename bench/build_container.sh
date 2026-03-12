#!/bin/bash
# Slime Container Build Script
# Run this on the node where Docker is available (typically the login node)
# This script builds the Docker image and pushes it to your registry

set -e

echo "========================================="
echo "Slime Container Build Script"
echo "========================================="
echo "This script must run on a node with Docker access (login node)"
echo "Started at: $(date)"
echo "========================================="

# Parse command line arguments
SQSH_OUTPUT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --sqsh-output)
            SQSH_OUTPUT="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            echo ""
            echo "Usage: $0 [--sqsh-output <path>]"
            echo ""
            echo "Options:"
            echo "  --sqsh-output <path>  Create .sqsh file at specified path (requires enroot)"
            echo ""
            echo "Examples:"
            echo "  $0                                           # Build only"
            echo "  $0 --sqsh-output slime.sqsh                  # Build and create .sqsh"
            echo "  $0 --sqsh-output /fsx/amine_dirhoussi/docker_images/slime.sqsh"
            exit 1
            ;;
    esac
done

# Disable git pager to avoid interactive prompts
export GIT_PAGER=cat

# Configuration
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="slime"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Path to local Slime clone
SLIME_SOURCE="${SLIME_SOURCE:-$REPO_ROOT}"

# Registry configuration
REGISTRY="${REGISTRY:-registry.hpc-cluster-hopper.hpc.internal.huggingface.tech}"

# Container images directory (for relative paths in --sqsh-output)
CONTAINER_IMAGES_DIR="${CONTAINER_IMAGES_DIR:-/fsx/amine_dirhoussi/docker_images}"

if [ -z "$REGISTRY" ]; then
    echo "ERROR: REGISTRY environment variable not set"
    echo "Please set it to your cluster's Docker registry:"
    echo "  export REGISTRY=registry.hpc-cluster-hopper.hpc.internal.huggingface.tech"
    exit 1
fi

# Validate local source exists
if [ ! -d "$SLIME_SOURCE" ]; then
    echo "ERROR: Slime source directory not found: $SLIME_SOURCE"
    echo ""
    echo "Expected to find local Slime clone at: $SLIME_SOURCE"
    exit 1
fi

# Check if it's a git repository
if [ ! -d "$SLIME_SOURCE/.git" ]; then
    echo "ERROR: $SLIME_SOURCE is not a git repository"
    echo ""
    echo "The local source must be a git clone with .git directory"
    echo "Please clone it properly using git"
    exit 1
fi

LOCAL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
REGISTRY_IMAGE="${REGISTRY}/library/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Configuration:"
echo "  Repository root: $REPO_ROOT"
echo "  Local Slime source: $SLIME_SOURCE"
echo "  Local image: $LOCAL_IMAGE"
echo "  Registry image: $REGISTRY_IMAGE"
echo ""
echo "Using local Slime clone (commit info):"
cd "$SLIME_SOURCE"
git --no-pager log -1 --pretty=format:"  Commit: %h%n  Date: %ai%n  Message: %s%n" || echo "  (Unable to read git info)"
cd "$REPO_ROOT"
echo "========================================="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not available on this node"
    echo ""
    echo "Please run this script on a node with Docker access (typically the login node)"
    echo "If you are on a login node and Docker is still not available, contact your cluster admin"
    exit 1
fi

# Check Docker daemon connectivity
if ! docker info &> /dev/null; then
    echo "ERROR: Cannot connect to Docker daemon"
    echo ""
    echo "Docker is installed but the daemon is not accessible."
    echo "Please check:"
    echo "  1. Are you on the correct login node?"
    echo "  2. Is the Docker service running?"
    echo "  3. Do you have permissions to access Docker?"
    exit 1
fi

echo "✓ Docker is available and daemon is accessible"
echo ""

# Build Docker image using local source
echo "========================================="
echo "Building Docker image from local source..."
echo "This will take 10-20 minutes due to:"
echo "  - Using slimerl/sglang base image"
echo "  - Installing dependencies and patching Megatron/SGLang"
echo ""
echo "Build context: Using local Slime source"
echo "========================================="

cd "$REPO_ROOT"

# Determine versions from version.txt to match docker/justfile behavior
if [ -f "docker/version.txt" ]; then
    VERSION="$(cat docker/version.txt | tr -d '\n')"
    BUILD_ARGS="--build-arg HTTP_PROXY=$http_proxy --build-arg HTTPS_PROXY=$https_proxy --build-arg NO_PROXY=localhost,127.0.0.1"

    # If the user hasn't overridden IMAGE_TAG, we might want to use VERSION,
    # but to keep it simple and aligned with the prime-rl script we'll stick to IMAGE_TAG
    # unless IMAGE_TAG is 'latest' and version.txt exists.
    if [ "$IMAGE_TAG" = "latest" ]; then
        LOCAL_IMAGE="${IMAGE_NAME}:${VERSION}"
        REGISTRY_IMAGE="${REGISTRY}/library/${IMAGE_NAME}:${VERSION}"
        echo "Auto-detected version from docker/version.txt: ${VERSION}"
    fi
else
    BUILD_ARGS=""
fi

# Build with buildx if available, fallback to standard build
if docker buildx version &> /dev/null; then
    echo "Using docker buildx for build..."
    docker buildx build \
        $BUILD_ARGS \
        -f docker/Dockerfile \
        -t $LOCAL_IMAGE \
        --load \
        .
else
    echo "Using standard docker build..."
    docker build \
        $BUILD_ARGS \
        -f docker/Dockerfile \
        -t $LOCAL_IMAGE \
        .
fi

if [ $? -eq 0 ]; then
    echo "✓ Docker image built successfully: $LOCAL_IMAGE"
else
    echo "ERROR: Docker image build failed"
    exit 1
fi

# Tag for registry
echo ""
echo "========================================="
echo "Tagging image for registry..."
echo "========================================="

docker tag $LOCAL_IMAGE $REGISTRY_IMAGE

# If we used the version tag, also tag as latest
if [ "$IMAGE_TAG" = "latest" ] && [ -f "docker/version.txt" ]; then
    LATEST_REGISTRY_IMAGE="${REGISTRY}/library/${IMAGE_NAME}:latest"
    docker tag $LOCAL_IMAGE $LATEST_REGISTRY_IMAGE
    echo "Also tagged as: $LATEST_REGISTRY_IMAGE"
fi

if [ $? -eq 0 ]; then
    echo "✓ Image tagged: $REGISTRY_IMAGE"
else
    echo "ERROR: Failed to tag image"
    exit 1
fi

# Check if image already exists in registry
echo ""
echo "========================================="
echo "Checking registry for existing image..."
echo "========================================="

if docker manifest inspect $REGISTRY_IMAGE &> /dev/null; then
    echo "⚠️  WARNING: Image already exists in registry: $REGISTRY_IMAGE"
    echo ""
    echo "The existing image will be overwritten."
    echo "If you want to preserve it, consider using a different tag:"
    echo "  export IMAGE_TAG=v1.0.0"
    echo "  export IMAGE_TAG=\$(date +%Y%m%d-%H%M%S)"
    echo ""
    echo "Press any key to continue or Ctrl+C to cancel..."
    read -n 1 -s -r
else
    echo "✓ No existing image found in registry (new image)"
fi

# Push to registry
echo ""
echo "========================================="
echo "Pushing to registry..."
echo "This may take several minutes..."
echo "========================================="

docker push $REGISTRY_IMAGE

if [ "$IMAGE_TAG" = "latest" ] && [ -f "docker/version.txt" ]; then
    docker push $LATEST_REGISTRY_IMAGE
fi

if [ $? -eq 0 ]; then
    echo "✓ Image pushed successfully to registry"
else
    echo "ERROR: Failed to push image to registry"
    echo ""
    echo "Please check:"
    echo "  1. Is the registry URL correct? ($REGISTRY)"
    echo "  2. Do you have push permissions to the registry?"
    echo "  3. Is the registry accessible from this node?"
    exit 1
fi

# Verify the push
echo ""
echo "========================================="
echo "Verifying registry push..."
echo "========================================="

# Try to pull the image to verify it's available
docker pull $REGISTRY_IMAGE &> /dev/null

if [ $? -eq 0 ]; then
    echo "✓ Image verified in registry"
else
    echo "WARNING: Could not verify image in registry (may still be accessible)"
fi

# Optional: Create .sqsh file with enroot
if [ -n "$SQSH_OUTPUT" ]; then
    echo ""
    echo "========================================="
    echo "Creating .sqsh file with enroot..."
    echo "========================================="

    # Check if enroot is available
    if ! command -v enroot &> /dev/null; then
        echo "ERROR: enroot is not available on this node"
        echo ""
        echo "The --sqsh-output option requires enroot to be installed."
        echo "You can create the .sqsh file later on compute nodes:"
        echo "  enroot import \"docker://${REGISTRY_IMAGE}\""

        # Adjust expected name based on tags
        ACTUAL_TAG="${VERSION:-$IMAGE_TAG}"
        echo "  mv library+${IMAGE_NAME}+${ACTUAL_TAG}.sqsh ${SQSH_OUTPUT}"
        exit 1
    fi

    echo "✓ Enroot is available"

    # Set enroot environment variables to use writable locations
    export ENROOT_RUNTIME_PATH="${ENROOT_RUNTIME_PATH:-$HOME/.enroot/runtime}"
    export ENROOT_DATA_PATH="${ENROOT_DATA_PATH:-$HOME/.enroot/data}"
    export ENROOT_CACHE_PATH="${ENROOT_CACHE_PATH:-$HOME/.enroot/cache}"

    # Create directories if they don't exist
    mkdir -p "$ENROOT_RUNTIME_PATH" "$ENROOT_DATA_PATH" "$ENROOT_CACHE_PATH"

    echo "Using enroot directories:"
    echo "  Runtime: $ENROOT_RUNTIME_PATH"
    echo "  Data: $ENROOT_DATA_PATH"
    echo "  Cache: $ENROOT_CACHE_PATH"

    # Determine output path
    if [[ "$SQSH_OUTPUT" = /* ]]; then
        # Absolute path provided
        SQSH_FINAL_PATH="$SQSH_OUTPUT"
        SQSH_DIR="$(dirname "$SQSH_FINAL_PATH")"
    else
        # Relative filename provided, use CONTAINER_IMAGES_DIR
        SQSH_FINAL_PATH="${CONTAINER_IMAGES_DIR}/${SQSH_OUTPUT}"
        SQSH_DIR="$CONTAINER_IMAGES_DIR"
    fi

    # Create output directory if needed
    mkdir -p "$SQSH_DIR"

    # Import with enroot
    echo ""
    echo "Importing from registry to .sqsh format..."
    echo "This may take 5-10 minutes..."
    echo ""

    enroot import "docker://${REGISTRY_IMAGE}"

    if [ $? -eq 0 ]; then
        ACTUAL_TAG="${VERSION:-$IMAGE_TAG}"
        ENROOT_FILENAME="library+${IMAGE_NAME}+${ACTUAL_TAG}.sqsh"

        if [ -f "$ENROOT_FILENAME" ]; then
            # Remove existing file if it exists at destination
            if [ -f "$SQSH_FINAL_PATH" ]; then
                echo "Removing existing .sqsh file: $SQSH_FINAL_PATH"
                rm -f "$SQSH_FINAL_PATH"
            fi

            # Move to final location
            mv "$ENROOT_FILENAME" "$SQSH_FINAL_PATH"

            if [ $? -eq 0 ]; then
                SQSH_SIZE=$(du -h "$SQSH_FINAL_PATH" | cut -f1)
                echo "✓ .sqsh file created successfully: $SQSH_FINAL_PATH"
                echo "  Size: $SQSH_SIZE"
            else
                echo "ERROR: Failed to move .sqsh file to $SQSH_FINAL_PATH"
                exit 1
            fi
        else
            echo "ERROR: .sqsh file not found after import: $ENROOT_FILENAME"
            echo "The import may have failed or created the file with a different name"
            exit 1
        fi
    else
        echo "ERROR: enroot import failed"
        echo ""
        echo "Please check:"
        echo "  1. Is the registry accessible from this node?"
        echo "  2. Is the image available in the registry?"
        echo "  3. Do you have sufficient disk space?"
        exit 1
    fi
fi

echo ""
echo "========================================="
echo "Build Completed Successfully!"
echo "Finished at: $(date)"
echo "========================================="
echo ""
echo "Image pushed to: $REGISTRY_IMAGE"

if [ "$IMAGE_TAG" = "latest" ] && [ -f "docker/version.txt" ]; then
    echo "Also available as: $LATEST_REGISTRY_IMAGE"
fi

if [ -n "$SQSH_OUTPUT" ]; then
    echo ".sqsh file created: $SQSH_FINAL_PATH"
fi

echo ""
echo "Next steps:"
echo ""

ACTUAL_TAG="${VERSION:-$IMAGE_TAG}"

if [ -n "$SQSH_OUTPUT" ]; then
    echo "1. Test the .sqsh file on compute nodes:"
    echo ""
    echo "   sbatch test_container.slurm --sqsh \"${SQSH_FINAL_PATH}\""
    echo ""
    echo "2. Use the .sqsh file in your training jobs:"
    echo ""
    echo "   export CONTAINER_IMAGE=\"${SQSH_FINAL_PATH}\""
    echo ""
    echo "   srun --gpus-per-node=8 \\"
    echo "     --container-image=\"\${CONTAINER_IMAGE}\" \\"
    echo "     --container-mounts=\"/fsx:/fsx,/scratch:/scratch\" \\"
    echo "     --no-container-mount-home \\"
    echo "     bash -c 'export PATH=/app/.venv/bin:\$PATH && cd /app && sft @ configs/debug/sft/train.toml'"
else
    echo "1. Test the container on compute nodes:"
    echo ""
    echo "   export REGISTRY=\"${REGISTRY}\""
    echo "   sbatch test_container.slurm"
    echo ""
    echo "2. Use the container in your jobs:"
    echo ""
    echo "   Option A - Direct from registry (simplest):"
    echo "   export CONTAINER_IMAGE=\"docker://${REGISTRY_IMAGE}\""
    echo ""
    echo "   Option B - Create .sqsh file for better performance:"
    echo "   ./build_container.sh --sqsh-output slime.sqsh"
    echo "   # Or manually:"
    echo "   enroot import \"docker://${REGISTRY_IMAGE}\""
    echo "   mv library+slime+${ACTUAL_TAG}.sqsh /fsx/amine_dirhoussi/docker_images/"
    echo ""
    echo "3. Run training:"
    echo ""
    echo "   srun --gpus-per-node=8 \\"
    echo "     --container-image=\"\${CONTAINER_IMAGE}\" \\"
    echo "     --container-mounts=\"/fsx:/fsx,/scratch:/scratch\" \\"
    echo "     --no-container-mount-home \\"
    echo "     bash -c 'export PATH=/app/.venv/bin:\$PATH && cd /app && sft @ configs/debug/sft/train.toml'"
fi

echo ""

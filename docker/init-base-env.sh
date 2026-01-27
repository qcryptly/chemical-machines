#!/bin/bash
#
# Initialize the base conda environment with PyTorch + CUDA support
# This script runs on first Docker Compose startup to install PyTorch
# with the correct CUDA version detected at runtime.
#
# CUDA Detection: Uses nvidia-smi to detect CUDA version
#
# All other packages are pre-installed in the Dockerfile.
# This script only handles PyTorch which needs runtime CUDA detection.
#

set -e

CONDA_PATH="${CONDA_PATH:-/opt/conda}"
MARKER_FILE="/opt/conda/.base-pytorch-initialized"

# Check if already initialized
if [ -f "$MARKER_FILE" ]; then
    echo "[init-base-env] PyTorch already initialized, skipping..."
    exit 0
fi

echo "[init-base-env] Installing PyTorch in base environment..."

# =============================================================================
# Detect CUDA version from nvidia-smi
# =============================================================================
detect_cuda_version() {
    if command -v nvidia-smi &> /dev/null; then
        # Parse "CUDA Version: X.Y" from nvidia-smi output
        local cuda_ver=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version:\s*\K[\d.]+' | head -1)
        if [ -n "$cuda_ver" ]; then
            echo "$cuda_ver"
            return 0
        fi
    fi
    echo ""
    return 1
}

CUDA_VERSION=$(detect_cuda_version)

if [ -n "$CUDA_VERSION" ]; then
    echo "[init-base-env] Detected CUDA version: $CUDA_VERSION"
    # Remove dot from version for PyTorch wheel URL (e.g., 12.4 -> 124)
    CUDA_VERSION_NODOT=$(echo "$CUDA_VERSION" | tr -d '.')
    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu${CUDA_VERSION_NODOT}"
else
    echo "[init-base-env] No CUDA detected, using CPU-only PyTorch"
    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
fi

# =============================================================================
# Install PyTorch with CUDA support
# =============================================================================
echo "[init-base-env] Installing PyTorch..."
echo "[init-base-env] Using index: $PYTORCH_INDEX_URL"
$CONDA_PATH/bin/pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url "$PYTORCH_INDEX_URL"

# =============================================================================
# Verify installation
# =============================================================================
echo "[init-base-env] Verifying installation..."
$CONDA_PATH/bin/python -c "
import sys
print(f'Python: {sys.version}')

# PyTorch
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# Verify other packages
import numpy as np
print(f'NumPy: {np.__version__}')

import scipy
print(f'SciPy: {scipy.__version__}')

import sympy
print(f'SymPy: {sympy.__version__}')

import matplotlib
print(f'Matplotlib: {matplotlib.__version__}')

import jedi
print(f'Jedi: {jedi.__version__}')

print()
print('Base environment ready!')
"

# Create marker file
date > "$MARKER_FILE"
echo "[init-base-env] Base environment initialized successfully!"

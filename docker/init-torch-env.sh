#!/bin/bash
#
# Initialize the default PyTorch environment for Chemical Machines
# This script runs on first Docker Compose startup to create the 'torch' conda environment
# with PyTorch + CUDA support and all required dependencies.
#
# CUDA Detection: Uses nvidia-smi to detect CUDA version, same method as
# cm-compute/src/compute/jobs/create-conda-environment.js detectCudaVersion()
#
# Required packages for cm-libraries:
#   - cm/symbols.py: sympy, torch (lazy loaded)
#   - cm/views.py: standard library only
#   - cm/qm.py: standard library only
#   - equation_builder.py: sympy, numpy (sympy.physics.quantum)
#

set -e

CONDA_PATH="${CONDA_PATH:-/opt/conda}"
ENV_NAME="torch"
MARKER_FILE="/opt/conda/envs/.torch-initialized"

# Check if already initialized
if [ -f "$MARKER_FILE" ]; then
    echo "[init-torch-env] Environment '$ENV_NAME' already initialized, skipping..."
    exit 0
fi

echo "[init-torch-env] Creating PyTorch environment '$ENV_NAME'..."

# =============================================================================
# Detect CUDA version from nvidia-smi (same method as create-conda-environment.js)
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
    echo "[init-torch-env] Detected CUDA version: $CUDA_VERSION"
    # Remove dot from version for PyTorch wheel URL (e.g., 12.4 -> 124)
    CUDA_VERSION_NODOT=$(echo "$CUDA_VERSION" | tr -d '.')
    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/nightly/cu${CUDA_VERSION_NODOT}"
else
    echo "[init-torch-env] No CUDA detected, using CPU-only PyTorch"
    PYTORCH_INDEX_URL="https://download.pytorch.org/whl/nightly/cpu"
fi

# =============================================================================
# Create the conda environment with Python 3.12
# =============================================================================
echo "[init-torch-env] Creating conda environment with Python 3.12..."
$CONDA_PATH/bin/conda create -n "$ENV_NAME" python=3.12 -y

# =============================================================================
# Install PyTorch with CUDA support using nightly builds (pip)
# Same approach as create-conda-environment.js
# =============================================================================
echo "[init-torch-env] Installing PyTorch nightly with CUDA support..."
echo "[init-torch-env] Using index: $PYTORCH_INDEX_URL"
$CONDA_PATH/bin/conda run -n "$ENV_NAME" pip install --no-cache-dir \
    --pre \
    torch torchvision torchaudio \
    --index-url "$PYTORCH_INDEX_URL"

# =============================================================================
# Core scientific computing packages (required by cm-libraries)
# =============================================================================
echo "[init-torch-env] Installing core scientific computing packages..."
$CONDA_PATH/bin/conda run -n "$ENV_NAME" pip install --no-cache-dir \
    numpy \
    scipy \
    pandas

# =============================================================================
# SymPy - required by cm/symbols.py and equation_builder.py
# =============================================================================
echo "[init-torch-env] Installing SymPy (symbolic mathematics)..."
$CONDA_PATH/bin/conda run -n "$ENV_NAME" pip install --no-cache-dir \
    sympy

# =============================================================================
# Visualization packages
# =============================================================================
echo "[init-torch-env] Installing visualization packages..."
$CONDA_PATH/bin/conda run -n "$ENV_NAME" pip install --no-cache-dir \
    matplotlib \
    pillow

# =============================================================================
# Machine Learning packages
# =============================================================================
echo "[init-torch-env] Installing machine learning packages..."
$CONDA_PATH/bin/conda run -n "$ENV_NAME" pip install --no-cache-dir \
    scikit-learn

# =============================================================================
# Chemistry and bioinformatics packages
# =============================================================================
echo "[init-torch-env] Installing chemistry and bioinformatics packages..."
$CONDA_PATH/bin/conda run -n "$ENV_NAME" pip install --no-cache-dir \
    biopython \
    rdkit \
    openmm \
    mdanalysis

# =============================================================================
# Database clients (for integration with PostgreSQL and Elasticsearch)
# =============================================================================
echo "[init-torch-env] Installing database clients..."
$CONDA_PATH/bin/conda run -n "$ENV_NAME" pip install --no-cache-dir \
    psycopg2-binary \
    elasticsearch

# =============================================================================
# C++ binding support
# =============================================================================
echo "[init-torch-env] Installing pybind11..."
$CONDA_PATH/bin/conda run -n "$ENV_NAME" pip install --no-cache-dir \
    pybind11

# =============================================================================
# Verify installation
# =============================================================================
echo "[init-torch-env] Verifying installation..."
$CONDA_PATH/bin/conda run -n "$ENV_NAME" python -c "
import sys
print(f'Python: {sys.version}')

# Core: PyTorch
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# Core: NumPy, SciPy, Pandas
import numpy as np
print(f'NumPy: {np.__version__}')

import scipy
print(f'SciPy: {scipy.__version__}')

import pandas as pd
print(f'Pandas: {pd.__version__}')

# cm-libraries requirement: SymPy
import sympy
print(f'SymPy: {sympy.__version__}')

# Test sympy.physics.quantum (used by equation_builder.py)
from sympy.physics.quantum import Operator, Commutator
print('SymPy quantum physics: OK')

# Visualization
import matplotlib
print(f'Matplotlib: {matplotlib.__version__}')

# Machine learning
import sklearn
print(f'Scikit-learn: {sklearn.__version__}')

print()
print('All cm-libraries dependencies verified!')
"

# Create marker file
mkdir -p "$(dirname "$MARKER_FILE")"
date > "$MARKER_FILE"
echo "[init-torch-env] Environment '$ENV_NAME' initialized successfully!"

#!/bin/bash
# Setup script for Prodigy on ARM Mac (Apple Silicon)
# This script automates the installation process

set -e  # Exit on error

echo "======================================"
echo "Prodigy ARM Mac Setup Script"
echo "======================================"
echo ""

# Check if we're on ARM Mac
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
    echo "Warning: This script is optimized for ARM Mac (Apple Silicon)"
    echo "Detected architecture: $ARCH"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Check Xcode Command Line Tools
echo "Step 1: Checking Xcode Command Line Tools..."
if ! xcode-select -p &> /dev/null; then
    echo "Xcode Command Line Tools not found. Installing..."
    xcode-select --install
    echo "Please complete the installation and run this script again."
    exit 1
else
    echo "✓ Xcode Command Line Tools found: $(xcode-select -p)"
fi

# Create conda environment
ENV_NAME="prodigy-env"
echo ""
echo "Step 2: Creating conda environment '$ENV_NAME'..."

if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists."
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n $ENV_NAME
        conda create -n $ENV_NAME python=3.10 -y
    fi
else
    conda create -n $ENV_NAME python=3.10 -y
fi

echo "✓ Environment created"

# Activate environment
echo ""
echo "Step 3: Activating environment..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME
echo "✓ Environment activated"

# Install C++ compiler toolchain
echo ""
echo "Step 4: Installing C++ compiler toolchain..."
conda install -c conda-forge clang_osx-arm64 clangxx_osx-arm64 -y
echo "✓ Compilers installed"

# Install NumPy
echo ""
echo "Step 5: Installing NumPy <2.0..."
pip install "numpy<2.0"
echo "✓ NumPy installed"

# Install PyTorch and PyG
echo ""
echo "Step 6: Installing PyTorch 2.4.1 and PyTorch Geometric..."
pip install torch==2.4.1 torch-geometric
echo "✓ PyTorch installed"

# Install PyG extensions
echo ""
echo "Step 7: Installing torch-scatter and torch-sparse..."
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
echo "✓ PyG extensions installed"

# Install remaining dependencies
echo ""
echo "Step 8: Installing remaining dependencies..."
if [ -f "requirements-arm-mac.txt" ]; then
    pip install -r requirements-arm-mac.txt
elif [ -f "requirements.txt" ]; then
    echo "Warning: requirements-arm-mac.txt not found, using requirements.txt"
    pip install -r requirements.txt
else
    echo "Warning: No requirements file found, skipping..."
fi
echo "✓ Dependencies installed"

# Verify installation
echo ""
echo "======================================"
echo "Verifying Installation..."
echo "======================================"

echo ""
echo "Python version:"
python --version

echo ""
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

echo ""
echo "Testing torch-scatter and torch-sparse:"
python -c "import torch_scatter, torch_sparse; print('✓ torch-scatter and torch-sparse imported successfully!')"

echo ""
echo "NumPy version:"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

echo ""
echo "======================================"
echo "✓ Installation Complete!"
echo "======================================"
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To test the setup, run:"
echo "  python test_twitter_loader.py"
echo ""

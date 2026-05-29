# Prodigy Setup Guide

This guide provides detailed instructions for setting up the Prodigy environment, specifically optimized for **ARM-based Macs** (Apple Silicon).

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed
- Xcode Command Line Tools (will be installed if not present)

## Quick Start

### Option 1: Automated Setup (Recommended for ARM Mac)

```bash
# Clone the repository
git clone <repository-url>
cd prodigy

# Run the automated setup script
chmod +x setup_arm_mac.sh
./setup_arm_mac.sh
```

The script will:
- Verify Xcode Command Line Tools
- Create a conda environment
- Install all dependencies in the correct order
- Verify the installation

### Option 2: Manual Setup

```bash
# Clone the repository
git clone <repository-url>
cd prodigy

# Create and activate conda environment
conda create -n prodigy-env python=3.10 -y
conda activate prodigy-env

# Follow the detailed setup below
```

## Detailed Setup Instructions

### Step 1: Verify Xcode Command Line Tools

First, ensure Xcode Command Line Tools are installed:

```bash
xcode-select -p
```

If you see a path like `/Library/Developer/CommandLineTools`, you're good to go. Otherwise, install them:

```bash
xcode-select --install
```

### Step 2: Install C++ Compiler Toolchain

PyTorch extensions (like torch-scatter and torch-sparse) require proper C++ compilers on ARM Mac. Install the conda compiler toolchain:

```bash
conda install -c conda-forge clang_osx-arm64 clangxx_osx-arm64 -y
```

This will install:
- Modern Clang compiler (version 21+)
- C++ standard library headers
- ARM64-specific build tools

### Step 3: Install Core Dependencies

Install NumPy first (must be <2.0 for PyTorch compatibility):

```bash
pip install "numpy<2.0"
```

### Step 4: Install PyTorch and PyTorch Geometric

Install PyTorch 2.4.1 and torch-geometric:

```bash
pip install torch==2.4.1 torch-geometric
```

**Important**: PyTorch 2.4.1+ is required for ARM Mac compatibility with modern compilers. Earlier versions (like 2.1.0) will fail to compile extensions.

### Step 5: Install PyTorch Geometric Extensions

Install torch-scatter and torch-sparse from the PyTorch Geometric wheel repository:

```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
```

This command downloads pre-compiled wheels for ARM Mac, avoiding compilation issues.

### Step 6: Install Remaining Dependencies

For ARM Mac, use the optimized requirements file:

```bash
pip install -r requirements-arm-mac.txt
```

For other platforms, use the standard requirements file:

```bash
pip install -r requirements.txt
```

### Step 7: Verify Installation

Test that everything is working:

```bash
python -c "import torch; import torch_scatter; import torch_sparse; print(f'PyTorch: {torch.__version__}'); print('torch-scatter and torch-sparse imported successfully!')"
```

Expected output:
```
PyTorch: 2.4.1
torch-scatter and torch-sparse imported successfully!
```

## Common Issues and Solutions

### Issue 1: NumPy 2.x Compatibility Error

**Error**: `_ARRAY_API not found` or NumPy compatibility warnings

**Solution**: Ensure NumPy is <2.0:
```bash
pip install --force-reinstall "numpy<2.0"
```

### Issue 2: Compilation Errors for torch-scatter/torch-sparse

**Error**: `'functional' file not found` or compilation failures

**Solutions**:
1. Ensure you installed the conda C++ compilers (Step 2)
2. Use pre-built wheels instead of compiling from source:
   ```bash
   pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
   ```

### Issue 3: PyTorch Version Mismatch

**Error**: Incompatible PyTorch versions

**Solution**: Ensure you're using PyTorch 2.4.1:
```bash
pip install --force-reinstall torch==2.4.1
```

### Issue 4: AssertionError in SubgraphDataset

**Error**: `AssertionError` when loading Twitter data

**Solution**: This is already fixed in the codebase. Ensure you have the latest version of `data/twitter_csv.py` with `bidirectional=False`.

## Environment Verification Checklist

Run these commands to verify your setup:

```bash
# 1. Check Python version
python --version  # Should be 3.10.x

# 2. Check architecture
uname -m  # Should be arm64

# 3. Check PyTorch
python -c "import torch; print(torch.__version__)"  # Should be 2.4.1

# 4. Check PyTorch Geometric extensions
python -c "import torch_scatter, torch_sparse; print('Success!')"

# 5. Check NumPy version
python -c "import numpy; print(numpy.__version__)"  # Should be 1.26.x
```

## Package Versions

The following versions are known to work on ARM Mac (as of February 2026):

```
torch==2.4.1
torch-geometric==2.7.0
torch-scatter==2.1.2
torch-sparse==0.6.18
numpy==1.26.4
scipy==1.15.3
pandas (latest)
scikit-learn==1.0.2
sentence-transformers==2.2.2
transformers==4.29.2
```

## Testing the Setup

Test the Twitter dataset loader:

```bash
python test_twitter_loader.py
```

Expected output should show successful graph creation and dataset loading without errors.

## Additional Notes

### For Intel-based Macs

If you're on an Intel Mac, you can skip the C++ compiler installation (Step 2) and use standard pip installation:

```bash
pip install torch-scatter torch-sparse
```

### For Linux

On Linux, the setup is simpler:

```bash
conda create -n prodigy-env python=3.10 -y
conda activate prodigy-env
pip install -r requirements.txt
```

### For GPU Support

If you have an NVIDIA GPU and want CUDA support:

```bash
# Install PyTorch with CUDA
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121

# Install extensions with CUDA support
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

## Troubleshooting

If you encounter any issues:

1. **Clear pip cache**: `pip cache purge`
2. **Recreate environment**:
   ```bash
   conda deactivate
   conda env remove -n prodigy-env
   # Then start from Step 1
   ```
3. **Check conda version**: Ensure conda is up to date: `conda update -n base conda`

## Support

For issues specific to:
- PyTorch: [PyTorch Forums](https://discuss.pytorch.org/)
- PyTorch Geometric: [PyG GitHub Issues](https://github.com/pyg-team/pytorch_geometric/issues)
- Prodigy: [Open an issue](https://github.com/your-repo/issues)

## License

This setup guide is part of the Prodigy project.

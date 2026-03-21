# Protenix Environment Setup

## Issue

Protenix (ByteDance's AlphaFold3 implementation) has a strict dependency on `triton==3.3.1`, which conflicts with the triton version bundled with PyTorch from conda-forge.

**Error encountered:**
```
ERROR: Could not find a version that satisfies the requirement triton==3.3.1 (from protenix)
ERROR: No matching distribution found for triton==3.3.1
```

The main `ddi_experiment` environment uses PyTorch 2.10.0 from conda-forge, which comes with triton 3.2.0. Protenix's `setup.py` requires exactly triton 3.3.1.

## Solution

Create a separate conda environment specifically for Protenix experiments. This is acceptable because:
- Protenix experiments are independent from RFAA and ESMFold experiments
- Each environment can have its own optimized dependency set
- Avoids complex dependency resolution issues

## Setup Instructions

### Create Protenix Environment

```bash
# Load required modules
module load gcc-native/12.3
module load cudatoolkit/24.11_12.6

# Create new environment
conda create -n protenix_env python=3.11 -y
conda activate protenix_env

# Install PyTorch (Protenix may work better with pip-installed PyTorch)
# Check Protenix's recommended PyTorch version in their docs
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or if pip torch doesn't work on aarch64, use conda-forge:
# mamba install -c conda-forge pytorch torchvision torchaudio -y

# Install Protenix
cd /projects/u6bz/jude/ddi_experiment/repos/Protenix
pip install -e .

# Install additional tools if needed
pip install tmtools dockq
```

### Alternative: Modify Protenix Requirements

If you want to try installing Protenix in the main environment, you can relax the triton constraint:

1. Edit `/projects/u6bz/jude/ddi_experiment/repos/Protenix/setup.py`
2. Change `triton==3.3.1` to `triton>=3.2.0`
3. Attempt installation: `pip install -e .`

**Warning:** This may cause runtime issues if Protenix relies on specific triton 3.3.1 features.

## Environment Summary

| Environment | Purpose | Key Packages |
|-------------|---------|--------------|
| `ddi_experiment` | RFAA, ESMFold, OpenFold experiments | PyTorch 2.10.0, OpenFold, fair-esm |
| `protenix_env` | Protenix (AF3) experiments | PyTorch + triton 3.3.1, Protenix |

## Repository Locations

All model repositories are shared across environments:
- OpenFold: `/projects/u6bz/jude/ddi_experiment/repos/openfold`
- RFAA: `/projects/u6bz/jude/ddi_experiment/repos/RoseTTAFold-All-Atom`
- Protenix: `/projects/u6bz/jude/ddi_experiment/repos/Protenix`

## System Notes

- **Architecture:** aarch64 (ARM64)
- **OS:** SLES 15 SP6
- **CUDA:** 12.6/12.9 available via modules
- **GCC:** Use `module load gcc-native/12.3` when building extensions (system gcc 7.5 is too old for PyTorch extensions)

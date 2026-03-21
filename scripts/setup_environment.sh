#!/bin/bash
# Environment setup for DDI Experiment
# Training structure prediction models on domain-domain interfaces

set -e

# Configuration
ENV_NAME="ddi_experiment"
PYTHON_VERSION="3.11"
PROJECT_DIR="/projects/u6bz/jude/ddi_experiment"
REPOS_DIR="${PROJECT_DIR}/repos"

echo "=============================================="
echo "DDI Experiment Environment Setup"
echo "=============================================="

# Create directories
mkdir -p "${REPOS_DIR}"
mkdir -p "${PROJECT_DIR}/logs"
mkdir -p "${PROJECT_DIR}/outputs"
mkdir -p "${PROJECT_DIR}/splits"

# Create conda environment
echo "Creating conda environment: ${ENV_NAME}"
module load cudatoolkit
conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

echo "Installing core dependencies..."

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install atomworks (for RFAA integration)
pip install "atomworks[ml]"

# Install other dependencies
pip install \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    matplotlib \
    seaborn \
    tqdm \
    wandb \
    tensorboard \
    pyyaml \
    hydra-core \
    omegaconf \
    biopython \
    biotite \
    dm-tree \
    ml-collections \
    pytorch-lightning \
    einops \
    fair-esm

# Install MMseqs2 for sequence clustering
conda install -c conda-forge -c bioconda mmseqs2 -y

# Clone model repositories
echo "Cloning model repositories..."

cd "${REPOS_DIR}"

# OpenFold
if [ ! -d "openfold" ]; then
    echo "Cloning OpenFold..."
    git clone https://github.com/aqlaboratory/openfold.git
    cd openfold
    pip install -e .
    cd ..
fi

# RoseTTAFold-All-Atom
if [ ! -d "RoseTTAFold-All-Atom" ]; then
    echo "Cloning RoseTTAFold-All-Atom..."
    git clone https://github.com/baker-laboratory/RoseTTAFold-All-Atom.git
    cd RoseTTAFold-All-Atom
    # Install RFAA dependencies
    pip install -e .
    cd ..
fi

# Protenix (AlphaFold3 implementation)
if [ ! -d "Protenix" ]; then
    echo "Cloning Protenix..."
    git clone https://github.com/bytedance/Protenix.git
    cd Protenix
    pip install -e .
    cd ..
fi

# Additional structure prediction tools
pip install tmtools  # TM-score calculation
pip install dockq    # DockQ scoring

echo "=============================================="
echo "Environment setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Repository locations:"
echo "  OpenFold: ${REPOS_DIR}/openfold"
echo "  RFAA: ${REPOS_DIR}/RoseTTAFold-All-Atom"
echo "  Protenix: ${REPOS_DIR}/Protenix"
echo ""
echo "Next steps:"
echo "  1. Run download_pdb_monomers.py to download PDB data"
echo "  2. Run process_ddi_data.py to filter DDI data"
echo "  3. Run create_splits.py to create train/val/test splits"
echo "  4. Run convert_to_training_format.py to prepare data"

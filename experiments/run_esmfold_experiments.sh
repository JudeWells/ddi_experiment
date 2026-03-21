#!/bin/bash
# Script to submit ESMFold linker trick experiments
# These are expected to be faster than training from scratch
# Submits 2 experiments × 3 seeds = 6 jobs total

set -e

PROJECT_DIR="/projects/u6bz/jude/ddi_experiment"
EXPERIMENTS_DIR="${PROJECT_DIR}/experiments"
SEEDS=(42 123 456)

echo "=========================================="
echo "ESMFold Linker Trick Experiments"
echo "=========================================="
echo ""
echo "These experiments fine-tune pretrained ESMFold"
echo "using the linker trick for multimer prediction."
echo ""
echo "Experiment 4A: PDB monomers + multimers only"
echo "Experiment 4B: PDB + DDI pseudo-multimers"
echo ""
echo "Seeds: ${SEEDS[@]}"
echo "Total jobs: 6"
echo "Expected time per job: ~24 hours"
echo "=========================================="

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "--- Seed ${seed} ---"

    # Experiment 4A: ESMFold PDB Only
    pdb_job=$(sbatch --parsable "${EXPERIMENTS_DIR}/exp4_esmfold_pdb_only.batch" ${seed})
    echo "  ESMFold PDB-only: ${pdb_job}"

    # Experiment 4B: ESMFold PDB + DDI
    ddi_job=$(sbatch --parsable "${EXPERIMENTS_DIR}/exp4_esmfold_pdb_ddi.batch" ${seed})
    echo "  ESMFold PDB+DDI: ${ddi_job}"
done

echo ""
echo "=========================================="
echo "All ESMFold jobs submitted!"
echo "=========================================="
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs in:"
echo "  ${PROJECT_DIR}/logs/"
echo ""
echo "These experiments should complete faster than"
echo "training from scratch (~24h vs ~72h)."
echo ""

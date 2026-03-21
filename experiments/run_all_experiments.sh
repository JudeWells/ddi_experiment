#!/bin/bash
# Master script to submit all DDI experiments
# Submits experiments for 4 models × 3 seeds

set -e

PROJECT_DIR="/projects/u6bz/jude/ddi_experiment"
EXPERIMENTS_DIR="${PROJECT_DIR}/experiments"
SEEDS=(42 123 456)

echo "=========================================="
echo "DDI Experiment - Full Run"
echo "=========================================="
echo ""
echo "Models: 4 (ESMFold, OpenFold, RFAA, Protenix)"
echo "Seeds: ${SEEDS[@]}"
echo ""
echo "Experiment breakdown:"
echo "  - ESMFold (linker trick): 2 experiments × 3 seeds = 6 jobs (~24h each)"
echo "  - OpenFold/RFAA/Protenix: 3 experiments × 3 models × 3 seeds = 27 jobs"
echo "  - Fine-tuning: 3 models × 3 seeds = 9 jobs (after pretrain)"
echo ""
echo "Total jobs: 42"
echo "=========================================="

# Function to submit job and return job ID
submit_job() {
    local script=$1
    local seed=$2
    local dependency=$3

    if [ -z "$dependency" ]; then
        job_id=$(sbatch --parsable ${script} ${seed})
    else
        job_id=$(sbatch --parsable --dependency=afterok:${dependency} ${script} ${seed})
    fi
    echo $job_id
}

# Track job IDs for dependencies
declare -A pretrain_jobs

echo ""
echo "Phase 0: ESMFold Linker Trick (fastest - starts from pretrained)"
echo "================================================================="
echo "These experiments fine-tune pretrained ESMFold using the linker trick."
echo "Expected completion: ~24 hours (vs ~72h for training from scratch)"

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "--- Seed ${seed} ---"

    # Experiment 4A: ESMFold PDB Only
    esm_pdb=$(submit_job "${EXPERIMENTS_DIR}/exp4_esmfold_pdb_only.batch" ${seed})
    echo "  ESMFold PDB-only: ${esm_pdb}"

    # Experiment 4B: ESMFold PDB + DDI
    esm_ddi=$(submit_job "${EXPERIMENTS_DIR}/exp4_esmfold_pdb_ddi.batch" ${seed})
    echo "  ESMFold PDB+DDI: ${esm_ddi}"
done

echo ""
echo "Phase 1: Baseline and DDI Pre-training (can run in parallel)"
echo "============================================================"

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "--- Seed ${seed} ---"

    # Experiment 1: Baselines (all models, parallel)
    echo "Submitting baselines..."
    baseline_of=$(submit_job "${EXPERIMENTS_DIR}/exp1_baseline_openfold.batch" ${seed})
    baseline_rfaa=$(submit_job "${EXPERIMENTS_DIR}/exp1_baseline_rfaa.batch" ${seed})
    baseline_ptx=$(submit_job "${EXPERIMENTS_DIR}/exp1_baseline_protenix.batch" ${seed})
    echo "  OpenFold baseline: ${baseline_of}"
    echo "  RFAA baseline: ${baseline_rfaa}"
    echo "  Protenix baseline: ${baseline_ptx}"

    # Experiment 2A: DDI Pre-training (all models, parallel)
    echo "Submitting DDI pre-training..."
    pretrain_of=$(submit_job "${EXPERIMENTS_DIR}/exp2_ddi_pretrain_openfold.batch" ${seed})
    pretrain_rfaa=$(submit_job "${EXPERIMENTS_DIR}/exp2_ddi_pretrain_rfaa.batch" ${seed})
    pretrain_ptx=$(submit_job "${EXPERIMENTS_DIR}/exp2_ddi_pretrain_protenix.batch" ${seed})
    echo "  OpenFold DDI pretrain: ${pretrain_of}"
    echo "  RFAA DDI pretrain: ${pretrain_rfaa}"
    echo "  Protenix DDI pretrain: ${pretrain_ptx}"

    # Store pretrain job IDs for fine-tuning dependencies
    pretrain_jobs["of_${seed}"]=${pretrain_of}
    pretrain_jobs["rfaa_${seed}"]=${pretrain_rfaa}
    pretrain_jobs["ptx_${seed}"]=${pretrain_ptx}

    # Experiment 3: Joint Training (all models, parallel)
    echo "Submitting joint training..."
    joint_of=$(submit_job "${EXPERIMENTS_DIR}/exp3_joint_openfold.batch" ${seed})
    joint_rfaa=$(submit_job "${EXPERIMENTS_DIR}/exp3_joint_rfaa.batch" ${seed})
    joint_ptx=$(submit_job "${EXPERIMENTS_DIR}/exp3_joint_protenix.batch" ${seed})
    echo "  OpenFold joint: ${joint_of}"
    echo "  RFAA joint: ${joint_rfaa}"
    echo "  Protenix joint: ${joint_ptx}"
done

echo ""
echo "Phase 2: Fine-tuning (depends on pre-training completion)"
echo "=========================================================="

for seed in "${SEEDS[@]}"; do
    echo ""
    echo "--- Seed ${seed} ---"

    # Experiment 2B: Fine-tuning (depends on DDI pre-training)
    echo "Submitting fine-tuning (will start after pre-training completes)..."

    finetune_of=$(submit_job "${EXPERIMENTS_DIR}/exp2_finetune_openfold.batch" ${seed} ${pretrain_jobs["of_${seed}"]})
    finetune_rfaa=$(submit_job "${EXPERIMENTS_DIR}/exp2_finetune_rfaa.batch" ${seed} ${pretrain_jobs["rfaa_${seed}"]})
    finetune_ptx=$(submit_job "${EXPERIMENTS_DIR}/exp2_finetune_protenix.batch" ${seed} ${pretrain_jobs["ptx_${seed}"]})

    echo "  OpenFold finetune: ${finetune_of} (depends on ${pretrain_jobs["of_${seed}"]})"
    echo "  RFAA finetune: ${finetune_rfaa} (depends on ${pretrain_jobs["rfaa_${seed}"]})"
    echo "  Protenix finetune: ${finetune_ptx} (depends on ${pretrain_jobs["ptx_${seed}"]})"
done

echo ""
echo "=========================================="
echo "All jobs submitted!"
echo "=========================================="
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs in:"
echo "  ${PROJECT_DIR}/logs/"
echo ""

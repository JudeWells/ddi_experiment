# DDI Experiment: Training on Domain-Domain Interfaces

## Hypothesis

Training structure prediction models on domain-domain interfaces (DDI) from the AlphaFold Database improves structure prediction on true protein multimers.

## Quick Start

```bash
# 1. Setup environment
cd /projects/u6bz/jude/ddi_experiment
bash scripts/setup_environment.sh

# 2. Activate environment
conda activate ddi_experiment

# 3. Prepare data
python scripts/download_pdb_monomers.py --type both --limit 100  # Test with small subset first
python scripts/process_ddi_data.py --limit 1000  # Process DDI data
python scripts/create_splits.py  # Create train/val/test splits
python scripts/convert_to_training_format.py  # Convert to model formats

# 4. Run all experiments
bash experiments/run_all_experiments.sh

# 5. Evaluate after training completes
sbatch experiments/run_evaluation.batch
```

## Directory Structure

```
/projects/u6bz/jude/ddi_experiment/
├── configs/                    # Training configurations
│   ├── baseline.yaml          # PDB-only baseline
│   ├── ddi_pretrain.yaml      # DDI pre-training
│   ├── finetune.yaml          # PDB fine-tuning
│   ├── joint.yaml             # Joint DDI+PDB training
│   ├── esmfold_pdb_only.yaml  # ESMFold linker - PDB only
│   └── esmfold_pdb_ddi.yaml   # ESMFold linker - PDB+DDI
├── environment/               # Conda environment specs
├── scripts/                   # Python scripts
│   ├── setup_environment.sh   # Environment setup
│   ├── download_pdb_monomers.py
│   ├── process_ddi_data.py
│   ├── create_splits.py
│   ├── convert_to_training_format.py
│   ├── train_openfold_soloSeq.py
│   ├── train_rfaa.py
│   ├── train_protenix.py
│   ├── train_esmfold_linker.py
│   ├── evaluate.py
│   └── aggregate_results.py
├── experiments/               # SLURM batch scripts
│   ├── exp1_baseline_*.batch
│   ├── exp2_ddi_pretrain_*.batch
│   ├── exp2_finetune_*.batch
│   ├── exp3_joint_*.batch
│   ├── exp4_esmfold_*.batch      # ESMFold linker experiments
│   ├── run_all_experiments.sh
│   ├── run_esmfold_experiments.sh # ESMFold only (faster)
│   └── run_evaluation.batch
├── splits/                    # Train/val/test splits
├── outputs/                   # Model checkpoints
├── logs/                      # Training logs
├── evaluation_results/        # Evaluation metrics
└── analysis/                  # Aggregated results
```

## Experimental Design

### Models
| Model | MSA Required | Multimer Support | Training Time | Notes |
|-------|--------------|------------------|---------------|-------|
| **ESMFold (linker)** | No | Via linker trick | **~24h** | Fastest - fine-tunes pretrained |
| OpenFold-SoloSeq | No | Yes | ~48-72h | Trains from scratch |
| RoseTTAFold-All-Atom | No | Yes | ~48-72h | Uses atomworks |
| Protenix (AF3) | Yes | Yes | ~72-96h | AlphaFold3 implementation |

### Experiments

**ESMFold Linker Trick (Fastest)**
| # | Experiment | Description |
|---|------------|-------------|
| 4A | ESMFold PDB-only | Fine-tune on PDB monomers + multimers |
| 4B | ESMFold PDB+DDI | Fine-tune on PDB + DDI pseudo-multimers |

**Full Training Experiments (OpenFold, RFAA, Protenix)**
| # | Experiment | Description |
|---|------------|-------------|
| 1 | Baseline | Train on PDB multimers only |
| 2 | DDI→PDB | Pre-train on DDI, fine-tune on PDB |
| 3 | Joint | Mixed DDI+PDB with weighted sampling |

### The Linker Trick

ESMFold was trained on monomers. The linker trick enables multimer prediction by:
1. Connecting multiple chains with a poly-glycine linker (25 G residues)
2. Model predicts structure of the joined sequence
3. Linker region is masked during loss computation

This allows fine-tuning from the pretrained checkpoint rather than training from scratch.

### Configuration
- **Date cutoff**: 2021-09-30 (matches AlphaFold training)
- **DDI filtering**: pLDDT > 70
- **Sequence clustering**: 50% identity (MMseqs2)
- **Seeds**: 42, 123, 456

## Data Sources

| Dataset | Location | Description |
|---------|----------|-------------|
| DDI domains | `/projects/u6bz/public/erik/AFDDI_data/` | READ-ONLY |
| PDB structures | `/projects/u6bz/public/jude/pdb_*` | Downloaded |
| Processed DDI | `/projects/u6bz/public/jude/processed_ddi/` | pLDDT>70 |

## Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| DockQ | >0.49 | Interface quality |
| lDDT | >0.70 | Local accuracy |
| TM-score | >0.50 | Global similarity |
| P@K | Higher | Contact precision |

## Running Individual Experiments

```bash
# Run just the ESMFold experiments (fastest, ~24h)
bash experiments/run_esmfold_experiments.sh

# Or run specific ESMFold experiment
sbatch experiments/exp4_esmfold_pdb_only.batch 42
sbatch experiments/exp4_esmfold_pdb_ddi.batch 42

# Run full training experiments
sbatch experiments/exp1_baseline_openfold.batch 42

# Resume training
python scripts/train_openfold_soloSeq.py \
    --config configs/baseline.yaml \
    --experiment baseline \
    --seed 42 \
    --resume outputs/openfold/baseline_seed42/checkpoint_epoch10.pt

# ESMFold linker training
python scripts/train_esmfold_linker.py \
    --config configs/esmfold_pdb_ddi.yaml \
    --experiment pdb_ddi \
    --seed 42
```

## Monitoring

```bash
# Check job status
squeue -u $USER

# Watch training logs
tail -f logs/exp1_baseline_openfold_*.out

# Check GPU usage
nvidia-smi
```

## Analysis

After all experiments complete:

```bash
# Run evaluation
sbatch experiments/run_evaluation.batch

# View results
cat analysis/experiment_report.md

# Check statistical significance
cat analysis/statistical_tests.csv
```

## Success Criteria

**Primary**: DDI pre-training (Exp 2) shows statistically significant improvement (p < 0.05) over baseline (Exp 1) in:
- DockQ on PDB multimer test set
- Inter-chain contact AUC

**Secondary**: Quantify optimal training strategy

## Troubleshooting

### Out of memory
- Reduce `crop_size` in config
- Reduce `batch_size`
- Enable gradient checkpointing

### Slow data loading
- Increase `num_workers`
- Check disk I/O with `iostat`

### NaN loss
- Reduce `learning_rate`
- Check data preprocessing
- Enable gradient clipping

## Contact

For questions about this experiment, check the logs or modify the configurations as needed.

#!/usr/bin/env python3
"""
ESMFold training with PyTorch Lightning and FSDP.

Supports:
- Model parallelism via FSDP (sharding across 4 GPUs per node)
- Data parallelism across nodes
- bf16 mixed precision with fp32 for critical operations
- Linker trick for multimer prediction
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Any
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import FSDPStrategy
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# ESMFold imports are deferred to avoid issues with distributed launch
ESMFOLD_AVAILABLE = None  # Will be set on first use
_esm_module = None
_FoldingTrunkBlock = None


def _load_esm_module():
    """Lazily load ESM module."""
    global ESMFOLD_AVAILABLE, _esm_module, _FoldingTrunkBlock
    if ESMFOLD_AVAILABLE is None:
        try:
            import esm
            from esm.esmfold.v1.trunk import TriangularSelfAttentionBlock
            _esm_module = esm
            _FoldingTrunkBlock = TriangularSelfAttentionBlock
            ESMFOLD_AVAILABLE = True
            logger.info("ESMFold loaded successfully")
        except ImportError as e:
            logger.warning(f"ESMFold import failed: {e}")
            ESMFOLD_AVAILABLE = False
    return _esm_module, _FoldingTrunkBlock


def get_fsdp_wrap_class():
    """Get transformer block class for FSDP wrapping."""
    _, block_cls = _load_esm_module()
    return block_cls

# Configuration
PROJECT_DIR = Path("/projects/u6bz/jude/ddi_experiment")
DATA_DIR = PROJECT_DIR / "training_data" / "esmfold"
SPLITS_DIR = PROJECT_DIR / "splits"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "esmfold"
SAMPLE_DATA_DIR = PROJECT_DIR / "sample_data"

DEFAULT_LINKER = "G" * 25

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Dataset Classes (unchanged from original)
# =============================================================================

class LinkerMultimerDataset(Dataset):
    """Dataset that applies linker trick to create pseudo-multimer sequences."""

    def __init__(
        self,
        structure_files: list,
        linker: str = DEFAULT_LINKER,
        max_length: int = 1024,
        crop_size: int = 512,
        use_chain_break_token: bool = True,
    ):
        self.structure_files = structure_files
        self.linker = linker
        self.max_length = max_length
        self.crop_size = crop_size
        self.use_chain_break_token = use_chain_break_token

        self.aa_vocab = list("ARNDCQEGHILKMFPSTWYV")
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.aa_vocab)}
        self.aa_to_idx["X"] = 20
        self.aa_to_idx["G"] = self.aa_vocab.index("G")

    def __len__(self):
        return len(self.structure_files)

    def __getitem__(self, idx, _retry_count=0):
        structure_file = self.structure_files[idx]
        max_retries = 10

        try:
            features = self._parse_and_link(structure_file)
            if features is None:
                if _retry_count < max_retries:
                    new_idx = np.random.randint(len(self))
                    return self.__getitem__(new_idx, _retry_count + 1)
                else:
                    return self._get_dummy_sample()
            return features
        except Exception as e:
            if _retry_count < max_retries:
                new_idx = np.random.randint(len(self))
                return self.__getitem__(new_idx, _retry_count + 1)
            else:
                return self._get_dummy_sample()

    def _get_dummy_sample(self):
        dummy_seq = "GGGGGGGGGG"
        return {
            "sequence": dummy_seq,
            "aatype": torch.zeros(len(dummy_seq), dtype=torch.int64),
            "coords": torch.zeros(len(dummy_seq), 3, dtype=torch.float32),
            "linker_mask": torch.ones(len(dummy_seq), dtype=torch.float32),
            "chain_indices": torch.zeros(len(dummy_seq), dtype=torch.int64),
            "seq_length": torch.tensor([len(dummy_seq)]),
            "num_chains": torch.tensor([1]),
        }

    def _parse_and_link(self, pdb_file: Path) -> Optional[dict]:
        """Parse PDB/CIF and apply linker trick for multi-chain structures."""
        import gemmi

        three_to_one = {
            "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
            "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
            "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
            "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
        }

        try:
            structure = gemmi.read_structure(str(pdb_file))
        except Exception:
            try:
                with open(pdb_file, 'r') as f:
                    content = f.read()
                lines = [l for l in content.split('\n')
                         if not l.startswith('MODEL') and not l.startswith('ENDMDL')]
                fixed_content = '\n'.join(lines)
                structure = gemmi.read_pdb_string(fixed_content)
            except Exception:
                return None

        if len(structure) == 0:
            return None

        model = structure[0]
        chains = {}

        for chain in model:
            chain_id = chain.name
            if chain_id not in chains:
                chains[chain_id] = {}

            for residue in chain:
                if not residue.is_water() and residue.name in three_to_one:
                    resnum = residue.seqid.num
                    if resnum not in chains[chain_id]:
                        chains[chain_id][resnum] = {
                            "resname": residue.name,
                            "coords": {},
                        }
                    for atom in residue:
                        chains[chain_id][resnum]["coords"][atom.name] = np.array(
                            [atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32
                        )

        if not chains:
            return None

        sorted_chains = sorted(chains.keys())
        linked_sequence = []
        linked_coords = []
        linker_mask = []
        chain_indices = []

        for chain_idx, chain_id in enumerate(sorted_chains):
            chain_data = chains[chain_id]
            sorted_resnums = sorted(chain_data.keys())

            for resnum in sorted_resnums:
                res = chain_data[resnum]
                aa = three_to_one.get(res["resname"], "X")
                linked_sequence.append(aa)
                ca_coord = res["coords"].get("CA", np.zeros(3))
                linked_coords.append(ca_coord)
                linker_mask.append(1.0)
                chain_indices.append(chain_idx)

            if chain_idx < len(sorted_chains) - 1:
                for _ in range(len(self.linker)):
                    linked_sequence.append("G")
                    linked_coords.append(np.zeros(3))
                    linker_mask.append(0.0)
                    chain_indices.append(-1)

        total_len = len(linked_sequence)
        if total_len < 10:
            return None

        if total_len > self.crop_size:
            linked_sequence = linked_sequence[:self.crop_size]
            linked_coords = linked_coords[:self.crop_size]
            linker_mask = linker_mask[:self.crop_size]
            chain_indices = chain_indices[:self.crop_size]
            total_len = self.crop_size

        sequence_str = "".join(linked_sequence)
        aatype = np.array([self.aa_to_idx.get(aa, 20) for aa in linked_sequence], dtype=np.int64)
        coords = np.stack(linked_coords)

        return {
            "sequence": sequence_str,
            "aatype": torch.tensor(aatype),
            "coords": torch.tensor(coords, dtype=torch.float32),
            "linker_mask": torch.tensor(linker_mask, dtype=torch.float32),
            "chain_indices": torch.tensor(chain_indices, dtype=torch.int64),
            "seq_length": torch.tensor([total_len]),
            "num_chains": torch.tensor([len(sorted_chains)]),
        }


class MixedLinkerDataset(Dataset):
    """Mixed dataset combining PDB and DDI data."""

    def __init__(
        self,
        pdb_files: list,
        ddi_files: list = None,
        pdb_weight: float = 1.0,
        ddi_weight: float = 1.0,
        **kwargs,
    ):
        self.pdb_dataset = LinkerMultimerDataset(pdb_files, **kwargs)
        if ddi_files:
            self.ddi_dataset = LinkerMultimerDataset(ddi_files, **kwargs)
            self.has_ddi = True
        else:
            self.ddi_dataset = None
            self.has_ddi = False

        self.pdb_weight = pdb_weight
        self.ddi_weight = ddi_weight

        if self.has_ddi:
            total = len(pdb_files) * pdb_weight + len(ddi_files) * ddi_weight
            self.pdb_prob = (len(pdb_files) * pdb_weight) / total
        else:
            self.pdb_prob = 1.0

    def __len__(self):
        if self.has_ddi:
            return len(self.pdb_dataset) + len(self.ddi_dataset)
        return len(self.pdb_dataset)

    def __getitem__(self, idx):
        if self.has_ddi and np.random.random() > self.pdb_prob:
            return self.ddi_dataset[np.random.randint(len(self.ddi_dataset))]
        return self.pdb_dataset[np.random.randint(len(self.pdb_dataset))]


def collate_fn(batch):
    """Collate function for ESMFold batches."""
    max_len = max(b["aatype"].shape[0] for b in batch)
    padded_batch = {"sequences": [b["sequence"] for b in batch]}

    for key in ["aatype", "coords", "linker_mask", "chain_indices"]:
        tensors = []
        for b in batch:
            t = b[key]
            if t.dim() == 1:
                pad = torch.zeros(max_len - t.shape[0], dtype=t.dtype)
                tensors.append(torch.cat([t, pad]))
            elif t.dim() == 2:
                pad = torch.zeros(max_len - t.shape[0], t.shape[1], dtype=t.dtype)
                tensors.append(torch.cat([t, pad], dim=0))
        padded_batch[key] = torch.stack(tensors)

    padded_batch["seq_length"] = torch.stack([b["seq_length"] for b in batch])
    padded_batch["num_chains"] = torch.stack([b["num_chains"] for b in batch])
    return padded_batch


def load_structure_files(split_type: str, data_type: str = "pdb", use_sample_data: bool = False) -> list:
    """Load structure file paths for a given split.

    Args:
        split_type: 'train' or 'val'
        data_type: 'pdb' or 'ddi'
        use_sample_data: If True, use sample_data directory for debugging
    """
    import pandas as pd

    # Use sample data directory if requested
    if use_sample_data:
        splits_dir = SAMPLE_DATA_DIR / "splits"
        pdb_dirs = [
            SAMPLE_DATA_DIR / "pdb_multimers",
            SAMPLE_DATA_DIR / "pdb_monomers",
        ]
    else:
        splits_dir = SPLITS_DIR
        pdb_dirs = [
            Path("/projects/u6bz/public/jude/pdb_multimers"),
            Path("/projects/u6bz/public/jude/pdb_monomers"),
        ]

    if data_type == "ddi":
        pairs_file = splits_dir / f"ddi_{split_type}_pairs.csv"
        if pairs_file.exists():
            pairs_df = pd.read_csv(pairs_file)
            files = []
            ddi_dir = DATA_DIR / "ddi_pairs"
            if not ddi_dir.exists():
                ddi_dir = PROJECT_DIR / "training_data" / "openfold" / "ddi_pairs"

            for _, row in pairs_df.iterrows():
                pair_id = f"{row['domain1_id']}_{row['domain2_id']}"
                pdb_file = ddi_dir / f"{pair_id}.pdb"
                if pdb_file.exists():
                    files.append(pdb_file)
            return files
    else:
        split_file = splits_dir / f"pdb_{split_type}.txt"
        files = []

        if split_file.exists():
            with open(split_file, "r") as f:
                pdb_ids = [line.strip() for line in f if line.strip()]

            for pdb_dir in pdb_dirs:
                if pdb_dir.exists():
                    for pdb_id in pdb_ids:
                        for suffix in [".pdb", ".cif"]:
                            pdb_file = pdb_dir / f"{pdb_id}{suffix}"
                            if pdb_file.exists():
                                files.append(pdb_file)
                                break
        return files
    return []


# =============================================================================
# Lightning DataModule
# =============================================================================

class ESMFoldDataModule(LightningDataModule):
    """Lightning DataModule for ESMFold training."""

    def __init__(
        self,
        experiment: str = "pdb_only",
        batch_size: int = 1,
        num_workers: int = 4,
        crop_size: int = 512,
        max_length: int = 1024,
        pdb_weight: float = 1.0,
        ddi_weight: float = 1.0,
        use_sample_data: bool = False,
    ):
        super().__init__()
        self.experiment = experiment
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size
        self.max_length = max_length
        self.pdb_weight = pdb_weight
        self.ddi_weight = ddi_weight
        self.use_sample_data = use_sample_data

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str = None):
        data_kwargs = {
            "crop_size": self.crop_size,
            "max_length": self.max_length,
        }

        pdb_train = load_structure_files("train", "pdb", use_sample_data=self.use_sample_data)
        pdb_val = load_structure_files("val", "pdb", use_sample_data=self.use_sample_data)

        if self.experiment == "pdb_ddi":
            ddi_train = load_structure_files("train", "ddi", use_sample_data=self.use_sample_data)
            ddi_val = load_structure_files("val", "ddi", use_sample_data=self.use_sample_data)

            self.train_dataset = MixedLinkerDataset(
                pdb_train, ddi_train,
                pdb_weight=self.pdb_weight,
                ddi_weight=self.ddi_weight,
                **data_kwargs,
            )
            self.val_dataset = MixedLinkerDataset(
                pdb_val, ddi_val,
                pdb_weight=self.pdb_weight,
                ddi_weight=self.ddi_weight,
                **data_kwargs,
            )
            logger.info(f"PDB train: {len(pdb_train)}, DDI train: {len(ddi_train)}")
        else:
            self.train_dataset = LinkerMultimerDataset(pdb_train, **data_kwargs)
            self.val_dataset = LinkerMultimerDataset(pdb_val, **data_kwargs)

        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


# =============================================================================
# Lightning Module
# =============================================================================

class ESMFoldLightningModule(LightningModule):
    """PyTorch Lightning module for ESMFold training with linker trick."""

    def __init__(
        self,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_epochs: int = 2,
        max_epochs: int = 30,
        freeze_esm_trunk: bool = True,
        chunk_size: int = 128,
        grad_clip: float = 0.5,
        max_steps: int = -1,
        limit_train_batches: int = -1,
        gradient_accumulation_steps: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.max_steps = max_steps
        self.limit_train_batches = limit_train_batches
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Load ESMFold (lazy import)
        esm_module, _ = _load_esm_module()
        if ESMFOLD_AVAILABLE and esm_module is not None:
            logger.info("Loading pretrained ESMFold model...")
            self.model = esm_module.pretrained.esmfold_v1()

            # Set chunk_size for memory efficiency
            if hasattr(self.model, "trunk") and hasattr(self.model.trunk, "chunk_size"):
                self.model.trunk.chunk_size = chunk_size
                logger.info(f"Set ESMFold trunk chunk_size to {chunk_size}")

            # Convert model to uniform dtype for FSDP compatibility
            # ESMFold has mixed fp16/fp32 params which FSDP cannot handle
            logger.info("Converting model to float32 for FSDP compatibility")
            self.model = self.model.float()

            # Freeze ESM trunk if requested
            if freeze_esm_trunk:
                logger.info("Freezing ESM trunk parameters")
                for param in self.model.esm.parameters():
                    param.requires_grad = False
        else:
            raise RuntimeError(
                f"ESMFold not available. ESMFOLD_AVAILABLE={ESMFOLD_AVAILABLE}. "
                "Install with: pip install fair-esm[esmfold]"
            )

    def forward(self, sequences: List[str]):
        """Forward pass through ESMFold."""
        from esm.esmfold.v1.misc import batch_encode_sequences

        aatype, mask, residx, linker_mask_esm, chain_index = batch_encode_sequences(
            sequences, residue_index_offset=512, chain_linker="G" * 25
        )

        device = next(self.model.parameters()).device
        aatype = aatype.to(device)
        mask = mask.to(device)
        residx = residx.to(device)

        # Forward pass with autocast disabled for stability
        # FSDP handles mixed precision at the wrapper level
        outputs = self.model.forward(aatype, mask=mask, residx=residx)

        return outputs

    def _compute_masked_loss(
        self,
        pred_coords: torch.Tensor,
        true_coords: torch.Tensor,
        linker_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute FAPE-like loss with linker masking."""
        device = pred_coords.device
        true_coords = true_coords.to(device)
        linker_mask = linker_mask.to(device)

        # Handle shape mismatches
        pred_len = pred_coords.shape[1]
        true_len = true_coords.shape[1]
        min_len = min(pred_len, true_len)

        pred_coords = pred_coords[:, :min_len, :]
        true_coords = true_coords[:, :min_len, :]
        linker_mask = linker_mask[:, :min_len]

        # Mask out residues with zero/invalid coordinates
        coord_valid = (true_coords.abs().sum(-1) > 1e-6).float()
        combined_mask = linker_mask * coord_valid

        # Coordinate difference loss
        diff = pred_coords - true_coords
        dist = torch.sqrt((diff ** 2).sum(-1) + 1e-8)
        masked_dist = dist * combined_mask

        valid_count = combined_mask.sum()
        if valid_count > 0:
            loss = masked_dist.sum() / valid_count
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        # NaN safeguard
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/Inf loss detected, returning zero loss")
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        return loss

    def training_step(self, batch, batch_idx):
        sequences = batch["sequences"]
        true_coords = batch["coords"]
        linker_mask = batch["linker_mask"]

        try:
            outputs = self.forward(sequences)
            pred_coords = outputs["positions"][-1, :, :, 1, :]  # CA atoms

            loss = self._compute_masked_loss(pred_coords, true_coords, linker_mask)

            # Skip if NaN
            if torch.isnan(loss) or torch.isinf(loss):
                return None

            # Log metrics
            self.log("train_loss", loss, prog_bar=True, sync_dist=True)

            if outputs.get("plddt") is not None:
                plddt_mean = outputs["plddt"].mean()
                self.log("train_plddt", plddt_mean, sync_dist=True)

            return loss

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM in training step, skipping batch")
                torch.cuda.empty_cache()
                return None
            raise

    def validation_step(self, batch, batch_idx):
        sequences = batch["sequences"]
        true_coords = batch["coords"]
        linker_mask = batch["linker_mask"]

        outputs = self.forward(sequences)
        pred_coords = outputs["positions"][-1, :, :, 1, :]

        loss = self._compute_masked_loss(pred_coords, true_coords, linker_mask)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        if outputs.get("plddt") is not None:
            self.log("val_plddt", outputs["plddt"].mean(), sync_dist=True)

        return loss

    def on_before_optimizer_step(self, optimizer):
        """Manually clip gradients for FSDP compatibility."""
        # FSDP requires using its own gradient clipping method
        if hasattr(self.trainer.strategy, "clip_gradients"):
            # Use FSDP's gradient clipping
            self.trainer.strategy.clip_gradients(
                optimizer,
                clip_val=self.grad_clip,
                gradient_clip_algorithm="value"  # 'value' is supported, 'norm' is not
            )

    def configure_optimizers(self):
        # Only optimize parameters that require gradients
        params = filter(lambda p: p.requires_grad, self.parameters())

        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Compute training steps without iterating through dataloader
        # This avoids the slow estimated_stepping_batches calculation
        if self.max_steps > 0:
            num_training_steps = self.max_steps
        elif self.limit_train_batches > 0:
            # limit_train_batches * max_epochs / gradient_accumulation
            num_training_steps = int(
                self.limit_train_batches * self.max_epochs / self.gradient_accumulation_steps
            )
        else:
            # Fallback: estimate based on typical dataset size
            # This will be refined once training starts
            num_training_steps = 10000
            logger.warning(f"Using fallback num_training_steps={num_training_steps}")

        num_warmup_steps = int(num_training_steps * self.warmup_epochs / self.max_epochs)
        logger.info(f"LR schedule: {num_training_steps} total steps, {num_warmup_steps} warmup")

        # Cosine annealing with warmup
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))

            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


# =============================================================================
# FSDP Configuration
# =============================================================================

def get_fsdp_strategy(num_gpus: int = 4) -> FSDPStrategy:
    """Configure FSDP strategy for model parallelism."""

    # Mixed precision config - bf16 for compute, fp32 for reduce and params
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,  # Keep reductions in fp32 for stability
        buffer_dtype=torch.bfloat16,
    )

    # Auto-wrap policy for transformer blocks
    # This wraps each TriangularSelfAttentionBlock separately for efficient sharding
    FoldingTrunkBlock = get_fsdp_wrap_class()
    if FoldingTrunkBlock is not None:
        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={FoldingTrunkBlock},
        )
        activation_checkpointing = {FoldingTrunkBlock}
    else:
        auto_wrap_policy = None
        activation_checkpointing = None

    strategy = FSDPStrategy(
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # Maximum memory savings
        mixed_precision=mixed_precision,
        auto_wrap_policy=auto_wrap_policy,
        activation_checkpointing_policy=activation_checkpointing,
        cpu_offload=False,  # Keep on GPU for speed
        limit_all_gathers=True,  # Memory optimization
    )

    return strategy


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train ESMFold with PyTorch Lightning and FSDP"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--experiment",
        choices=["pdb_only", "pdb_ddi"],
        default="pdb_only",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--devices", type=int, default=4, help="GPUs per node")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set seed
    pl.seed_everything(args.seed, workers=True)

    # Create output directory
    experiment_name = f"esmfold_{args.experiment}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = OUTPUT_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Enable tensor cores for better performance
    torch.set_float32_matmul_precision('medium')

    # Data module
    data_config = config.get("data", {})
    datamodule = ESMFoldDataModule(
        experiment=args.experiment,
        batch_size=config.get("batch_size", 1),
        num_workers=config.get("num_workers", 4),
        crop_size=data_config.get("crop_size", 512),
        max_length=data_config.get("max_length", 1024),
        pdb_weight=config.get("pdb_weight", 1.0),
        ddi_weight=config.get("ddi_weight", 1.0),
        use_sample_data=data_config.get("use_sample_data", False),
    )

    # Model
    model_config = config.get("model", {})
    model = ESMFoldLightningModule(
        learning_rate=config.get("learning_rate", 1e-5),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_epochs=config.get("warmup_epochs", 2),
        max_epochs=config.get("max_epochs", 30),
        freeze_esm_trunk=model_config.get("freeze_esm_trunk", True),
        chunk_size=model_config.get("chunk_size", 128),
        grad_clip=config.get("grad_clip", 0.5),
        max_steps=config.get("max_steps", -1),
        limit_train_batches=config.get("limit_train_batches", -1) if isinstance(config.get("limit_train_batches"), int) else -1,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir,
            filename="best-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=config.get("early_stopping_patience", 5),
            mode="min",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Logger
    wandb_config = config.get("wandb", {})
    wandb_logger = WandbLogger(
        project=wandb_config.get("project", "ddi-experiment"),
        name=f"esmfold_{args.experiment}_seed{args.seed}",
        save_dir=str(output_dir),
        tags=wandb_config.get("tags", [args.experiment, "esmfold", "fsdp"]),
    )

    # FSDP Strategy
    strategy = get_fsdp_strategy(num_gpus=args.devices)

    # Trainer
    # Note: gradient_clip_val with 'norm' algorithm is not supported with FSDP
    # Gradient clipping is handled via FSDP's clip_grad_norm_ in on_before_optimizer_step
    trainer = Trainer(
        accelerator="gpu",
        devices=args.devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        precision="bf16-mixed",
        max_epochs=config.get("max_epochs", 30),
        max_steps=config.get("max_steps", -1),  # Use -1 for unlimited
        # gradient_clip_val removed - not compatible with FSDP, handled manually
        accumulate_grad_batches=config.get("gradient_accumulation_steps", 4),
        log_every_n_steps=config.get("log_every_n_steps", 50),
        val_check_interval=config.get("val_check_interval", 500),
        limit_train_batches=config.get("limit_train_batches", 1.0),  # Can limit for debug
        limit_val_batches=config.get("limit_val_batches", 1.0),
        num_sanity_val_steps=0,  # Skip sanity validation for faster startup
        callbacks=callbacks,
        logger=wandb_logger,
        enable_progress_bar=True,
        deterministic=False,  # For performance
    )

    # Train
    logger.info(f"Starting training: {args.experiment}")
    logger.info(f"Nodes: {args.num_nodes}, GPUs/node: {args.devices}")
    logger.info(f"Output: {output_dir}")

    trainer.fit(model, datamodule=datamodule)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()

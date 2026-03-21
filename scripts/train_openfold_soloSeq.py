#!/usr/bin/env python3
"""
Training script for OpenFold SoloSeq model.

SoloSeq is an MSA-free variant of OpenFold that uses only single sequences,
making it suitable for rapid training without MSA generation overhead.

Supports three training modes:
1. Baseline: PDB multimers only
2. DDI Pre-training + PDB Fine-tuning
3. Joint training: Mixed DDI + PDB data
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import yaml
from tqdm import tqdm

# Add openfold to path
OPENFOLD_PATH = Path("/projects/u6bz/jude/ddi_experiment/repos/openfold")
if OPENFOLD_PATH.exists():
    sys.path.insert(0, str(OPENFOLD_PATH))

try:
    from openfold.model.model import AlphaFold
    from openfold.config import model_config
    from openfold.utils.tensor_utils import tensor_tree_map
    OPENFOLD_AVAILABLE = True
except ImportError:
    OPENFOLD_AVAILABLE = False
    logging.warning("OpenFold not available. Install from: https://github.com/aqlaboratory/openfold")

# Configuration
PROJECT_DIR = Path("/projects/u6bz/jude/ddi_experiment")
DATA_DIR = PROJECT_DIR / "training_data" / "openfold"
SPLITS_DIR = PROJECT_DIR / "splits"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "openfold"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ProteinStructureDataset(Dataset):
    """Dataset for protein structures (DDI pairs or PDB monomers/multimers)."""

    def __init__(
        self,
        structure_files: list,
        max_length: int = 512,
        crop_size: int = 256,
    ):
        self.structure_files = structure_files
        self.max_length = max_length
        self.crop_size = crop_size

        # Amino acid vocabulary
        self.aa_to_idx = {
            aa: i for i, aa in enumerate("ARNDCQEGHILKMFPSTWYV")
        }
        self.aa_to_idx["X"] = 20  # Unknown

    def __len__(self):
        return len(self.structure_files)

    def __getitem__(self, idx):
        pdb_file = self.structure_files[idx]

        try:
            features = self._parse_structure(pdb_file)
            if features is None:
                # Return random valid sample if parsing fails
                return self.__getitem__(np.random.randint(len(self)))
            return features
        except Exception as e:
            logger.warning(f"Error loading {pdb_file}: {e}")
            return self.__getitem__(np.random.randint(len(self)))

    def _parse_structure(self, pdb_file: Path) -> Optional[dict]:
        """Parse PDB file and extract features for SoloSeq."""
        three_to_one = {
            "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
            "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
            "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
            "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
        }

        residues = {}  # (chain, resnum) -> resname
        coords = {}    # (chain, resnum) -> {atom_name: xyz}

        with open(pdb_file, "r") as f:
            for line in f:
                if line.startswith("ATOM"):
                    chain = line[21]
                    resnum = int(line[22:26].strip())
                    resname = line[17:20].strip()
                    atom_name = line[12:16].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])

                    key = (chain, resnum)
                    if key not in residues:
                        residues[key] = resname
                        coords[key] = {}

                    coords[key][atom_name] = np.array([x, y, z], dtype=np.float32)

        if len(residues) < 10:
            return None

        # Sort by chain and residue number
        sorted_keys = sorted(residues.keys())

        # Crop if too long
        if len(sorted_keys) > self.crop_size:
            start = np.random.randint(0, len(sorted_keys) - self.crop_size)
            sorted_keys = sorted_keys[start:start + self.crop_size]

        n_res = len(sorted_keys)

        # Build sequence and aatype
        sequence = "".join(
            three_to_one.get(residues[k], "X") for k in sorted_keys
        )
        aatype = np.array([
            self.aa_to_idx.get(aa, 20) for aa in sequence
        ], dtype=np.int64)

        # Build backbone coordinates (N, CA, C, O)
        atom_names = ["N", "CA", "C", "O"]
        all_atom_positions = np.zeros((n_res, 37, 3), dtype=np.float32)
        all_atom_mask = np.zeros((n_res, 37), dtype=np.float32)

        # Atom indices in OpenFold format
        atom_order = {
            "N": 0, "CA": 1, "C": 2, "CB": 3, "O": 4,
        }

        for i, key in enumerate(sorted_keys):
            for atom_name, xyz in coords[key].items():
                if atom_name in atom_order:
                    atom_idx = atom_order[atom_name]
                    all_atom_positions[i, atom_idx] = xyz
                    all_atom_mask[i, atom_idx] = 1.0

        # Chain index (for multimer support)
        chain_index = np.array([
            ord(k[0]) - ord("A") for k in sorted_keys
        ], dtype=np.int64)

        # Residue index
        residue_index = np.arange(n_res, dtype=np.int64)

        # Create features dict
        features = {
            "aatype": torch.tensor(aatype),
            "residue_index": torch.tensor(residue_index),
            "chain_index": torch.tensor(chain_index),
            "all_atom_positions": torch.tensor(all_atom_positions),
            "all_atom_mask": torch.tensor(all_atom_mask),
            "seq_length": torch.tensor([n_res]),
        }

        return features


class MixedDataset(Dataset):
    """Dataset that mixes DDI and PDB data with weighted sampling."""

    def __init__(
        self,
        ddi_files: list,
        pdb_files: list,
        pdb_weight: float = 2.0,
        **kwargs,
    ):
        self.ddi_dataset = ProteinStructureDataset(ddi_files, **kwargs)
        self.pdb_dataset = ProteinStructureDataset(pdb_files, **kwargs)

        # Oversample PDB data
        self.ddi_weight = 1.0
        self.pdb_weight = pdb_weight

        total_weight = len(ddi_files) * self.ddi_weight + len(pdb_files) * self.pdb_weight
        self.ddi_prob = (len(ddi_files) * self.ddi_weight) / total_weight

    def __len__(self):
        return len(self.ddi_dataset) + len(self.pdb_dataset)

    def __getitem__(self, idx):
        # Randomly sample based on weights
        if np.random.random() < self.ddi_prob:
            return self.ddi_dataset[np.random.randint(len(self.ddi_dataset))]
        else:
            return self.pdb_dataset[np.random.randint(len(self.pdb_dataset))]


def collate_fn(batch):
    """Collate function for batching protein structures."""
    # Find max length in batch
    max_len = max(b["aatype"].shape[0] for b in batch)

    # Pad all tensors to max length
    padded_batch = {}

    for key in batch[0].keys():
        if key == "seq_length":
            padded_batch[key] = torch.stack([b[key] for b in batch])
        else:
            tensors = []
            for b in batch:
                t = b[key]
                if t.dim() == 1:
                    pad = torch.zeros(max_len - t.shape[0], dtype=t.dtype)
                    tensors.append(torch.cat([t, pad]))
                elif t.dim() == 2:
                    pad = torch.zeros(max_len - t.shape[0], t.shape[1], dtype=t.dtype)
                    tensors.append(torch.cat([t, pad], dim=0))
                elif t.dim() == 3:
                    pad = torch.zeros(max_len - t.shape[0], t.shape[1], t.shape[2], dtype=t.dtype)
                    tensors.append(torch.cat([t, pad], dim=0))

            padded_batch[key] = torch.stack(tensors)

    return padded_batch


def load_structure_files(split_type: str, data_type: str = "ddi") -> list:
    """Load structure file paths for a given split."""
    if data_type == "ddi":
        pairs_file = SPLITS_DIR / f"ddi_{split_type}_pairs.csv"
        if pairs_file.exists():
            import pandas as pd
            pairs_df = pd.read_csv(pairs_file)

            files = []
            ddi_dir = DATA_DIR / "ddi_pairs"
            for _, row in pairs_df.iterrows():
                pair_id = f"{row['domain1_id']}_{row['domain2_id']}"
                pdb_file = ddi_dir / f"{pair_id}.pdb"
                if pdb_file.exists():
                    files.append(pdb_file)

            return files
    else:
        split_file = SPLITS_DIR / f"pdb_{split_type}.txt"
        if split_file.exists():
            with open(split_file, "r") as f:
                pdb_ids = [line.strip() for line in f if line.strip()]

            files = []
            struct_dir = DATA_DIR / "structures"
            for pdb_id in pdb_ids:
                for suffix in [".pdb", ".cif"]:
                    pdb_file = struct_dir / f"{pdb_id}{suffix}"
                    if pdb_file.exists():
                        files.append(pdb_file)
                        break

            return files

    return []


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])

        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)

        return rank, world_size, local_rank

    return 0, 1, 0


def train_epoch(
    model,
    dataloader,
    optimizer,
    scaler,
    device,
    epoch,
    config,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        # Move to device
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        with autocast(enabled=config.get("use_amp", True)):
            # Forward pass
            outputs = model(batch)
            loss = outputs["loss"]

        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 1.0))
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validating"):
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(batch)
        loss = outputs["loss"]

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def save_checkpoint(model, optimizer, scaler, epoch, loss, path):
    """Save training checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss": loss,
    }, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(model, optimizer, scaler, path):
    """Load training checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]


def main():
    parser = argparse.ArgumentParser(
        description="Train OpenFold SoloSeq model"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to training configuration YAML file",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["baseline", "ddi_pretrain", "finetune", "joint"],
        default="baseline",
        help="Experiment type",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=None,
        help="Path to pretrained model for fine-tuning",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        logger.info(f"Starting {args.experiment} training with seed {args.seed}")
        logger.info(f"Config: {config}")

    # Create output directory
    experiment_name = f"{args.experiment}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = OUTPUT_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Load data based on experiment type
    if args.experiment == "baseline":
        # PDB multimers only
        train_files = load_structure_files("train", "pdb")
        val_files = load_structure_files("val", "pdb")
        train_dataset = ProteinStructureDataset(train_files, **config.get("data", {}))
        val_dataset = ProteinStructureDataset(val_files, **config.get("data", {}))

    elif args.experiment == "ddi_pretrain":
        # DDI data only
        train_files = load_structure_files("train", "ddi")
        val_files = load_structure_files("val", "ddi")
        train_dataset = ProteinStructureDataset(train_files, **config.get("data", {}))
        val_dataset = ProteinStructureDataset(val_files, **config.get("data", {}))

    elif args.experiment == "finetune":
        # PDB multimers for fine-tuning (must have pretrained model)
        if args.pretrained is None:
            raise ValueError("Fine-tuning requires --pretrained argument")
        train_files = load_structure_files("train", "pdb")
        val_files = load_structure_files("val", "pdb")
        train_dataset = ProteinStructureDataset(train_files, **config.get("data", {}))
        val_dataset = ProteinStructureDataset(val_files, **config.get("data", {}))

    elif args.experiment == "joint":
        # Mixed DDI + PDB data
        ddi_train = load_structure_files("train", "ddi")
        pdb_train = load_structure_files("train", "pdb")
        ddi_val = load_structure_files("val", "ddi")
        pdb_val = load_structure_files("val", "pdb")

        train_dataset = MixedDataset(
            ddi_train, pdb_train,
            pdb_weight=config.get("pdb_weight", 2.0),
            **config.get("data", {}),
        )
        val_dataset = MixedDataset(
            ddi_val, pdb_val,
            pdb_weight=config.get("pdb_weight", 2.0),
            **config.get("data", {}),
        )

    if rank == 0:
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")

    # Create data loaders
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size", 1),
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("batch_size", 1),
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.get("num_workers", 4),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Create model
    if OPENFOLD_AVAILABLE:
        cfg = model_config("model_1")
        model = AlphaFold(cfg)
    else:
        # Placeholder model for testing
        logger.warning("Using placeholder model (OpenFold not available)")
        model = torch.nn.Linear(256, 256)

    model = model.to(device)

    # Load pretrained weights if specified
    if args.pretrained is not None:
        logger.info(f"Loading pretrained model from {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Wrap in DDP if distributed
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 0.01),
    )

    scaler = GradScaler(enabled=config.get("use_amp", True))

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume is not None:
        start_epoch, _ = load_checkpoint(model, optimizer, scaler, args.resume)
        logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    patience = config.get("early_stopping_patience", 10)

    for epoch in range(start_epoch, config.get("max_epochs", 100)):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, device, epoch, config
        )

        val_loss = validate(model, val_loader, device)

        if rank == 0:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            # Save checkpoint
            save_checkpoint(
                model, optimizer, scaler, epoch, val_loss,
                output_dir / f"checkpoint_epoch{epoch}.pt",
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    model, optimizer, scaler, epoch, val_loss,
                    output_dir / "best_model.pt",
                )
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping after {patience} epochs without improvement")
                break

    if rank == 0:
        logger.info(f"Training complete. Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Model saved to {output_dir}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

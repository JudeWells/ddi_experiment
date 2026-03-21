#!/usr/bin/env python3
"""
Training script for RoseTTAFold-All-Atom (RFAA) model.

RFAA natively supports multimer prediction and uses atomworks for
data processing. It doesn't require MSAs, making it suitable for
this experiment.

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

# Add RFAA to path
RFAA_PATH = Path("/projects/u6bz/jude/ddi_experiment/repos/RoseTTAFold-All-Atom")
if RFAA_PATH.exists():
    sys.path.insert(0, str(RFAA_PATH))

try:
    import atomworks
    ATOMWORKS_AVAILABLE = True
except ImportError:
    ATOMWORKS_AVAILABLE = False
    logging.warning("atomworks not available. Install with: pip install 'atomworks[ml]'")

# Configuration
PROJECT_DIR = Path("/projects/u6bz/jude/ddi_experiment")
DATA_DIR = PROJECT_DIR / "training_data" / "rfaa"
SPLITS_DIR = PROJECT_DIR / "splits"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "rfaa"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RFAADataset(Dataset):
    """Dataset for RFAA training using atomworks."""

    def __init__(
        self,
        structure_files: list,
        max_length: int = 512,
        crop_size: int = 256,
        use_atomworks: bool = True,
    ):
        self.structure_files = structure_files
        self.max_length = max_length
        self.crop_size = crop_size
        self.use_atomworks = use_atomworks and ATOMWORKS_AVAILABLE

        # Amino acid vocabulary
        self.aa_vocab = list("ARNDCQEGHILKMFPSTWYV")
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.aa_vocab)}
        self.aa_to_idx["X"] = 20

    def __len__(self):
        return len(self.structure_files)

    def __getitem__(self, idx):
        structure_file = self.structure_files[idx]

        try:
            if self.use_atomworks:
                features = self._parse_with_atomworks(structure_file)
            else:
                features = self._parse_pdb(structure_file)

            if features is None:
                return self.__getitem__(np.random.randint(len(self)))

            return features

        except Exception as e:
            logger.warning(f"Error loading {structure_file}: {e}")
            return self.__getitem__(np.random.randint(len(self)))

    def _parse_with_atomworks(self, structure_file: Path) -> Optional[dict]:
        """Parse structure using atomworks library."""
        try:
            structure = atomworks.load_structure(str(structure_file))

            # Extract features from atomworks structure
            # This is a simplified version - full RFAA uses more features

            coords = structure.get_atom_coords()
            sequence = structure.get_sequence()

            n_res = len(sequence)

            if n_res < 10:
                return None

            # Crop if too long
            if n_res > self.crop_size:
                start = np.random.randint(0, n_res - self.crop_size)
                sequence = sequence[start:start + self.crop_size]
                coords = coords[start:start + self.crop_size]
                n_res = self.crop_size

            aatype = np.array([
                self.aa_to_idx.get(aa, 20) for aa in sequence
            ], dtype=np.int64)

            features = {
                "aatype": torch.tensor(aatype),
                "coords": torch.tensor(coords, dtype=torch.float32),
                "seq_length": torch.tensor([n_res]),
            }

            return features

        except Exception as e:
            logger.warning(f"atomworks parsing failed for {structure_file}: {e}")
            return self._parse_pdb(structure_file)

    def _parse_pdb(self, pdb_file: Path) -> Optional[dict]:
        """Parse PDB file directly."""
        three_to_one = {
            "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
            "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
            "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
            "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
        }

        residues = {}
        ca_coords = {}

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

                    if atom_name == "CA":
                        ca_coords[key] = np.array([x, y, z], dtype=np.float32)

        if len(residues) < 10:
            return None

        sorted_keys = sorted(residues.keys())

        # Crop if too long
        if len(sorted_keys) > self.crop_size:
            start = np.random.randint(0, len(sorted_keys) - self.crop_size)
            sorted_keys = sorted_keys[start:start + self.crop_size]

        n_res = len(sorted_keys)

        sequence = "".join(
            three_to_one.get(residues[k], "X") for k in sorted_keys
        )
        aatype = np.array([
            self.aa_to_idx.get(aa, 20) for aa in sequence
        ], dtype=np.int64)

        # CA coordinates
        coords = np.stack([
            ca_coords.get(k, np.zeros(3)) for k in sorted_keys
        ])

        # Chain information
        chain_idx = np.array([
            ord(k[0]) - ord("A") for k in sorted_keys
        ], dtype=np.int64)

        features = {
            "aatype": torch.tensor(aatype),
            "coords": torch.tensor(coords),
            "chain_idx": torch.tensor(chain_idx),
            "seq_length": torch.tensor([n_res]),
        }

        return features


class MixedRFAADataset(Dataset):
    """Mixed DDI and PDB dataset for joint training."""

    def __init__(
        self,
        ddi_files: list,
        pdb_files: list,
        pdb_weight: float = 2.0,
        **kwargs,
    ):
        self.ddi_dataset = RFAADataset(ddi_files, **kwargs)
        self.pdb_dataset = RFAADataset(pdb_files, **kwargs)

        self.ddi_weight = 1.0
        self.pdb_weight = pdb_weight

        total = len(ddi_files) * self.ddi_weight + len(pdb_files) * self.pdb_weight
        self.ddi_prob = (len(ddi_files) * self.ddi_weight) / total

    def __len__(self):
        return len(self.ddi_dataset) + len(self.pdb_dataset)

    def __getitem__(self, idx):
        if np.random.random() < self.ddi_prob:
            return self.ddi_dataset[np.random.randint(len(self.ddi_dataset))]
        else:
            return self.pdb_dataset[np.random.randint(len(self.pdb_dataset))]


def collate_fn(batch):
    """Collate function for RFAA batches."""
    max_len = max(b["aatype"].shape[0] for b in batch)

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

            padded_batch[key] = torch.stack(tensors)

    return padded_batch


def load_structure_files(split_type: str, data_type: str = "ddi") -> list:
    """Load structure file paths for a given split."""
    import pandas as pd

    if data_type == "ddi":
        pairs_file = SPLITS_DIR / f"ddi_{split_type}_pairs.csv"
        if pairs_file.exists():
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
                for suffix in [".pdb", ".cif", ".atomworks"]:
                    pdb_file = struct_dir / f"{pdb_id}{suffix}"
                    if pdb_file.exists():
                        files.append(pdb_file)
                        break

            return files

    return []


class RFAAModel(torch.nn.Module):
    """Simplified RFAA model architecture for training."""

    def __init__(self, config: dict):
        super().__init__()

        self.embed_dim = config.get("embed_dim", 256)
        self.num_layers = config.get("num_layers", 12)
        self.num_heads = config.get("num_heads", 8)

        # Embedding layers
        self.aa_embed = torch.nn.Embedding(21, self.embed_dim)
        self.pos_embed = torch.nn.Embedding(1024, self.embed_dim)

        # Transformer encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
        )

        # Structure prediction head
        self.coord_head = torch.nn.Sequential(
            torch.nn.Linear(self.embed_dim, self.embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.embed_dim, 3),  # xyz coordinates
        )

        # Distance prediction head for contacts
        self.dist_head = torch.nn.Sequential(
            torch.nn.Linear(self.embed_dim * 2, self.embed_dim),
            torch.nn.GELU(),
            torch.nn.Linear(self.embed_dim, 64),  # distance bins
        )

    def forward(self, batch):
        aatype = batch["aatype"]
        seq_length = batch["seq_length"]
        batch_size, max_len = aatype.shape

        # Create position indices
        pos_idx = torch.arange(max_len, device=aatype.device).unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        x = self.aa_embed(aatype) + self.pos_embed(pos_idx)

        # Create attention mask
        mask = torch.arange(max_len, device=aatype.device).unsqueeze(0) >= seq_length
        mask = mask.unsqueeze(1).unsqueeze(2)

        # Transformer encoding
        x = self.encoder(x, src_key_padding_mask=mask.squeeze(1).squeeze(1))

        # Predict coordinates
        pred_coords = self.coord_head(x)

        # Compute loss if ground truth available
        loss = torch.tensor(0.0, device=aatype.device)

        if "coords" in batch:
            true_coords = batch["coords"]
            coord_mask = ~mask.squeeze(1).squeeze(1)

            # FAPE-like loss
            diff = pred_coords - true_coords
            dist_loss = torch.sqrt((diff ** 2).sum(-1) + 1e-8)
            loss = (dist_loss * coord_mask).sum() / coord_mask.sum().clamp(min=1)

        return {
            "loss": loss,
            "pred_coords": pred_coords,
        }


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


def train_epoch(model, dataloader, optimizer, scaler, device, epoch, config):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        with autocast(enabled=config.get("use_amp", True)):
            outputs = model(batch)
            loss = outputs["loss"]

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
        total_loss += outputs["loss"].item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def save_checkpoint(model, optimizer, scaler, epoch, loss, path):
    """Save training checkpoint."""
    state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save({
        "epoch": epoch,
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "loss": loss,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train RFAA model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--experiment",
        choices=["baseline", "ddi_pretrain", "finetune", "joint"],
        default="baseline",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--pretrained", type=Path, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        logger.info(f"Starting RFAA {args.experiment} training with seed {args.seed}")

    # Create output directory
    experiment_name = f"{args.experiment}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = OUTPUT_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Load data
    data_config = config.get("data", {})

    if args.experiment == "baseline":
        train_files = load_structure_files("train", "pdb")
        val_files = load_structure_files("val", "pdb")
        train_dataset = RFAADataset(train_files, **data_config)
        val_dataset = RFAADataset(val_files, **data_config)

    elif args.experiment == "ddi_pretrain":
        train_files = load_structure_files("train", "ddi")
        val_files = load_structure_files("val", "ddi")
        train_dataset = RFAADataset(train_files, **data_config)
        val_dataset = RFAADataset(val_files, **data_config)

    elif args.experiment == "finetune":
        if args.pretrained is None:
            raise ValueError("Fine-tuning requires --pretrained")
        train_files = load_structure_files("train", "pdb")
        val_files = load_structure_files("val", "pdb")
        train_dataset = RFAADataset(train_files, **data_config)
        val_dataset = RFAADataset(val_files, **data_config)

    elif args.experiment == "joint":
        ddi_train = load_structure_files("train", "ddi")
        pdb_train = load_structure_files("train", "pdb")
        ddi_val = load_structure_files("val", "ddi")
        pdb_val = load_structure_files("val", "pdb")

        train_dataset = MixedRFAADataset(
            ddi_train, pdb_train,
            pdb_weight=config.get("pdb_weight", 2.0),
            **data_config,
        )
        val_dataset = MixedRFAADataset(
            ddi_val, pdb_val,
            pdb_weight=config.get("pdb_weight", 2.0),
            **data_config,
        )

    if rank == 0:
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")

    # Data loaders
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
    model = RFAAModel(config.get("model", {}))
    model = model.to(device)

    # Load pretrained if specified
    if args.pretrained is not None:
        checkpoint = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded pretrained model from {args.pretrained}")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 0.01),
    )

    scaler = GradScaler(enabled=config.get("use_amp", True))

    # Resume if specified
    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    patience = config.get("early_stopping_patience", 10)

    for epoch in range(start_epoch, config.get("max_epochs", 100)):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch, config)
        val_loss = validate(model, val_loader, device)

        if rank == 0:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            save_checkpoint(model, optimizer, scaler, epoch, val_loss,
                          output_dir / f"checkpoint_epoch{epoch}.pt")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(model, optimizer, scaler, epoch, val_loss,
                              output_dir / "best_model.pt")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping after {patience} epochs")
                break

    if rank == 0:
        logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

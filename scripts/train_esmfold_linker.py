#!/usr/bin/env python3
"""
Training script for ESMFold with linker trick for multimer prediction.

The linker trick connects multiple chains with a poly-glycine linker,
allowing a monomer-trained model to predict multimer structures.

This approach starts from a pretrained ESMFold checkpoint and fine-tunes
on multimer data, which should be faster than training from scratch.

Supports two training modes:
1. PDB-only: Train on PDB monomers + multimers
2. PDB+DDI: Train on PDB monomers + multimers + AFDB DDI pseudo-multimers
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import yaml
from tqdm import tqdm
import wandb

# Try to import ESMFold
try:
    import esm
    from esm.esmfold.v1 import esmfold
    ESMFOLD_AVAILABLE = True
except ImportError:
    ESMFOLD_AVAILABLE = False
    logging.warning("ESMFold not available. Install with: pip install fair-esm")

# Configuration
PROJECT_DIR = Path("/projects/u6bz/jude/ddi_experiment")
DATA_DIR = PROJECT_DIR / "training_data" / "esmfold"
SPLITS_DIR = PROJECT_DIR / "splits"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "esmfold"

# Linker configuration
DEFAULT_LINKER = "G" * 25  # 25 glycine residues as linker
CHAIN_BREAK_TOKEN = ":"  # Used in ESMFold for chain breaks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class LinkerMultimerDataset(Dataset):
    """
    Dataset that applies the linker trick to create pseudo-multimer sequences.

    The linker trick:
    1. Takes multiple chain sequences
    2. Joins them with a poly-glycine linker
    3. Model predicts structure of the joined sequence
    4. Linker region is masked during loss computation
    """

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
        self.aa_to_idx["G"] = self.aa_vocab.index("G")  # Glycine for linker

    def __len__(self):
        return len(self.structure_files)

    def __getitem__(self, idx, _retry_count=0):
        structure_file = self.structure_files[idx]
        max_retries = 10  # Prevent infinite recursion

        try:
            features = self._parse_and_link(structure_file)
            if features is None:
                if _retry_count < max_retries:
                    new_idx = np.random.randint(len(self))
                    return self.__getitem__(new_idx, _retry_count + 1)
                else:
                    # Return a dummy sample as last resort
                    return self._get_dummy_sample()
            return features

        except Exception as e:
            if _retry_count < max_retries:
                logger.warning(f"Error loading {structure_file}: {e}")
                new_idx = np.random.randint(len(self))
                return self.__getitem__(new_idx, _retry_count + 1)
            else:
                logger.error(f"Max retries exceeded, returning dummy sample")
                return self._get_dummy_sample()

    def _get_dummy_sample(self):
        """Return a minimal dummy sample when all retries fail."""
        dummy_seq = "GGGGGGGGGG"  # 10 glycines
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

        # Parse using gemmi (handles both PDB and CIF)
        try:
            structure = gemmi.read_structure(str(pdb_file))
        except Exception as e:
            # Try reading as PDB with preprocessing to fix MODEL/ENDMDL issues
            try:
                with open(pdb_file, 'r') as f:
                    content = f.read()
                # Remove MODEL/ENDMDL lines that cause issues
                lines = [l for l in content.split('\n')
                         if not l.startswith('MODEL') and not l.startswith('ENDMDL')]
                fixed_content = '\n'.join(lines)
                structure = gemmi.read_pdb_string(fixed_content)
            except Exception as e2:
                logger.warning(f"Failed to read {pdb_file}: {e2}")
                return None

        if len(structure) == 0:
            return None

        model = structure[0]
        chains = {}  # chain_id -> {resnum: {"resname": str, "coords": {atom_name: np.array}}}

        for chain in model:
            chain_id = chain.name
            if chain_id not in chains:
                chains[chain_id] = {}

            for residue in chain:
                # Skip non-polymer residues (water, ligands, etc.)
                if not residue.is_water() and residue.name in three_to_one:
                    resnum = residue.seqid.num
                    key = resnum
                    if key not in chains[chain_id]:
                        chains[chain_id][key] = {
                            "resname": residue.name,
                            "coords": {},
                        }

                    for atom in residue:
                        chains[chain_id][key]["coords"][atom.name] = np.array(
                            [atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32
                        )

        if not chains:
            return None

        # Sort chains and residues
        sorted_chains = sorted(chains.keys())

        # Build linked sequence and coordinates
        linked_sequence = []
        linked_coords = []  # CA coordinates
        linker_mask = []  # 1 for real residues, 0 for linker
        chain_indices = []  # Which chain each residue belongs to

        for chain_idx, chain_id in enumerate(sorted_chains):
            chain_data = chains[chain_id]
            sorted_resnums = sorted(chain_data.keys())

            for resnum in sorted_resnums:
                res = chain_data[resnum]
                aa = three_to_one.get(res["resname"], "X")
                linked_sequence.append(aa)

                # Get CA coordinates
                ca_coord = res["coords"].get("CA", np.zeros(3))
                linked_coords.append(ca_coord)

                linker_mask.append(1.0)
                chain_indices.append(chain_idx)

            # Add linker between chains (except after last chain)
            if chain_idx < len(sorted_chains) - 1:
                for _ in range(len(self.linker)):
                    linked_sequence.append("G")
                    linked_coords.append(np.zeros(3))  # Placeholder coords
                    linker_mask.append(0.0)  # Mask out linker in loss
                    chain_indices.append(-1)  # -1 for linker

        # Check length
        total_len = len(linked_sequence)
        if total_len < 10:
            return None

        # Crop if too long
        if total_len > self.crop_size:
            # Try to crop while keeping chains intact
            # For simplicity, just crop from the end
            linked_sequence = linked_sequence[:self.crop_size]
            linked_coords = linked_coords[:self.crop_size]
            linker_mask = linker_mask[:self.crop_size]
            chain_indices = chain_indices[:self.crop_size]
            total_len = self.crop_size

        # Convert to tensors
        sequence_str = "".join(linked_sequence)

        # For ESMFold, we can also use chain break tokens
        if self.use_chain_break_token:
            # Replace linker with chain break token representation
            # ESMFold uses specific handling for multi-chain
            pass

        aatype = np.array([
            self.aa_to_idx.get(aa, 20) for aa in linked_sequence
        ], dtype=np.int64)

        coords = np.stack(linked_coords)

        features = {
            "sequence": sequence_str,
            "aatype": torch.tensor(aatype),
            "coords": torch.tensor(coords, dtype=torch.float32),
            "linker_mask": torch.tensor(linker_mask, dtype=torch.float32),
            "chain_indices": torch.tensor(chain_indices, dtype=torch.int64),
            "seq_length": torch.tensor([total_len]),
            "num_chains": torch.tensor([len(sorted_chains)]),
        }

        return features


class MixedLinkerDataset(Dataset):
    """
    Mixed dataset combining PDB and DDI data with linker trick.

    For DDI pairs, combines two domain structures as a pseudo-multimer.
    """

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

    padded_batch = {
        "sequences": [b["sequence"] for b in batch],
    }

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


def load_structure_files(split_type: str, data_type: str = "pdb") -> list:
    """Load structure file paths for a given split."""
    import pandas as pd

    if data_type == "ddi":
        pairs_file = SPLITS_DIR / f"ddi_{split_type}_pairs.csv"
        if pairs_file.exists():
            pairs_df = pd.read_csv(pairs_file)

            files = []
            # Look for combined DDI pair files
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
        # PDB monomers and multimers
        split_file = SPLITS_DIR / f"pdb_{split_type}.txt"
        files = []

        if split_file.exists():
            with open(split_file, "r") as f:
                pdb_ids = [line.strip() for line in f if line.strip()]

            for pdb_dir in [
                Path("/projects/u6bz/public/jude/pdb_multimers"),
                Path("/projects/u6bz/public/jude/pdb_monomers"),
            ]:
                if pdb_dir.exists():
                    for pdb_id in pdb_ids:
                        for suffix in [".pdb", ".cif"]:
                            pdb_file = pdb_dir / f"{pdb_id}{suffix}"
                            if pdb_file.exists():
                                files.append(pdb_file)
                                break

        return files

    return []


class ESMFoldWrapper(torch.nn.Module):
    """
    Wrapper around ESMFold for training with linker trick.

    Handles:
    - Loading pretrained ESMFold checkpoint
    - Forward pass with linked sequences
    - Loss computation with linker masking
    """

    def __init__(self, config: dict):
        super().__init__()

        self.config = config

        if ESMFOLD_AVAILABLE:
            # Load pretrained ESMFold
            logger.info("Loading pretrained ESMFold model...")
            self.model = esm.pretrained.esmfold_v1()

            # Set chunk_size for memory efficiency
            chunk_size = config.get("model", {}).get("chunk_size", 128)
            if hasattr(self.model, "trunk") and hasattr(self.model.trunk, "chunk_size"):
                self.model.trunk.chunk_size = chunk_size
                logger.info(f"Set ESMFold trunk chunk_size to {chunk_size}")

            # Optionally freeze ESM trunk
            if config.get("model", {}).get("freeze_esm_trunk", False) or config.get("freeze_esm_trunk", False):
                logger.info("Freezing ESM trunk parameters")
                for param in self.model.esm.parameters():
                    param.requires_grad = False
        else:
            # Placeholder for testing
            logger.warning("Using placeholder model (ESMFold not available)")
            self.model = None
            self.placeholder = torch.nn.Linear(256, 3)

    def forward(self, batch):
        sequences = batch["sequences"]
        true_coords = batch["coords"]
        linker_mask = batch["linker_mask"]
        seq_length = batch["seq_length"]
        debug_batch_idx = batch.get("_debug_batch_idx")

        if self.model is not None:
            # ESMFold forward pass - use forward directly (not infer) for gradients
            from esm.esmfold.v1.misc import batch_encode_sequences

            # Debug: log first few batches
            if debug_batch_idx is not None and debug_batch_idx < 3:
                logger.info(f"Batch {debug_batch_idx}: seq_len={len(sequences[0])}, seq_preview={sequences[0][:50]}...")

            aatype, mask, residx, linker_mask_esm, chain_index = batch_encode_sequences(
                sequences, residue_index_offset=512, chain_linker="G" * 25
            )
            aatype = aatype.to(next(self.model.parameters()).device)
            mask = mask.to(next(self.model.parameters()).device)
            residx = residx.to(next(self.model.parameters()).device)

            # Debug: check input tensors
            if debug_batch_idx is not None and debug_batch_idx < 3:
                logger.info(f"  aatype shape={aatype.shape}, mask sum={mask.sum().item()}")

            # Forward pass - ESM trunk params are frozen via requires_grad=False in __init__
            # Don't use torch.no_grad() here as it blocks gradients for structure module too
            # Disable autocast for ESMFold forward - it can produce NaN with mixed precision
            with torch.amp.autocast('cuda', enabled=False):
                # Cast inputs to float32 for ESMFold forward
                outputs = self.model.forward(aatype, mask=mask, residx=residx)

            pred_coords = outputs["positions"][-1, :, :, 1, :]  # CA atoms

            # Debug: check output
            if debug_batch_idx is not None and debug_batch_idx < 3:
                logger.info(f"  pred_coords: nan={torch.isnan(pred_coords).sum().item()}, range=[{pred_coords.min():.2f}, {pred_coords.max():.2f}]")

            # Compute masked loss (exclude linker residues)
            loss = self._compute_masked_loss(
                pred_coords, true_coords, linker_mask, seq_length
            )

            return {
                "loss": loss,
                "pred_coords": pred_coords,
                "plddt": outputs.get("plddt"),
            }
        else:
            # Placeholder forward
            batch_size, max_len = batch["aatype"].shape
            pred_coords = self.placeholder(
                torch.randn(batch_size, max_len, 256, device=batch["aatype"].device)
            )

            loss = self._compute_masked_loss(
                pred_coords, true_coords, linker_mask, seq_length
            )

            return {"loss": loss, "pred_coords": pred_coords}

    def _compute_masked_loss(
        self,
        pred_coords: torch.Tensor,
        true_coords: torch.Tensor,
        linker_mask: torch.Tensor,
        seq_length: torch.Tensor,
    ) -> torch.Tensor:
        """Compute FAPE-like loss with linker masking."""
        # Ensure same device
        device = pred_coords.device
        true_coords = true_coords.to(device)
        linker_mask = linker_mask.to(device)

        # Handle shape mismatches (ESMFold encoding may differ from parsed length)
        pred_len = pred_coords.shape[1]
        true_len = true_coords.shape[1]
        min_len = min(pred_len, true_len)

        pred_coords = pred_coords[:, :min_len, :]
        true_coords = true_coords[:, :min_len, :]
        linker_mask = linker_mask[:, :min_len]

        # Also mask out residues with zero/invalid true coordinates (missing CA atoms)
        coord_valid = (true_coords.abs().sum(-1) > 1e-6).float()
        combined_mask = linker_mask * coord_valid

        # Coordinate difference
        diff = pred_coords - true_coords
        dist = torch.sqrt((diff ** 2).sum(-1) + 1e-8)

        # Apply combined mask (0 for linker and missing coords, 1 for valid real residues)
        masked_dist = dist * combined_mask

        # Compute mean over valid positions
        valid_count = combined_mask.sum()
        if valid_count > 0:
            loss = masked_dist.sum() / valid_count
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        # NaN safeguard
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/Inf loss detected. valid_count={valid_count}, pred range=[{pred_coords.min():.2f}, {pred_coords.max():.2f}]")
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        return loss


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

    for batch_idx, batch in enumerate(pbar):
        # Move tensors to device (sequences stay as list)
        batch_device = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_device[k] = v.to(device)
            else:
                batch_device[k] = v

        optimizer.zero_grad()
        skip_batch = False

        try:
            # Pass batch_idx for debug logging in first few batches
            batch_device["_debug_batch_idx"] = batch_idx if epoch == 0 else None

            with autocast(enabled=config.get("use_amp", True)):
                outputs = model(batch_device)
                loss = outputs["loss"]
                pred_coords = outputs.get("pred_coords")

            # Check for NaN/Inf in loss or predictions
            loss_is_bad = torch.isnan(loss) or torch.isinf(loss)
            pred_has_nan = pred_coords is not None and torch.isnan(pred_coords).any()

            if loss_is_bad or pred_has_nan:
                logger.warning(f"Skipping batch: loss_bad={loss_is_bad}, pred_nan={pred_has_nan}")
                skip_batch = True
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 1.0))
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM on batch, clearing cache and skipping")
                torch.cuda.empty_cache()
                skip_batch = True
            else:
                raise

        if skip_batch:
            optimizer.zero_grad()
            torch.cuda.empty_cache()  # Free memory after skipped batches
            continue

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Validating"):
        batch_device = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch_device[k] = v.to(device)
            else:
                batch_device[k] = v

        outputs = model(batch_device)
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
    parser = argparse.ArgumentParser(
        description="Train ESMFold with linker trick for multimer prediction"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--experiment",
        choices=["pdb_only", "pdb_ddi"],
        default="pdb_only",
        help="pdb_only: PDB monomers+multimers; pdb_ddi: also include DDI pseudo-multimers",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        logger.info(f"Starting ESMFold linker training: {args.experiment}")
        logger.info(f"Seed: {args.seed}")

        # Initialize wandb
        wandb_config = config.get("wandb", {})
        wandb.init(
            project=wandb_config.get("project", "ddi-esmfold-training"),
            name=f"esmfold_{args.experiment}_seed{args.seed}",
            config={
                "experiment": args.experiment,
                "seed": args.seed,
                **config,
            },
            tags=wandb_config.get("tags", [args.experiment, "esmfold"]),
        )

    # Create output directory
    experiment_name = f"esmfold_{args.experiment}_seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = OUTPUT_DIR / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Load data
    data_config_raw = config.get("data", {})
    # Filter to only valid dataset kwargs
    valid_data_keys = {"linker", "max_length", "crop_size", "use_chain_break_token"}
    data_config = {k: v for k, v in data_config_raw.items() if k in valid_data_keys}

    pdb_train = load_structure_files("train", "pdb")
    pdb_val = load_structure_files("val", "pdb")

    if args.experiment == "pdb_ddi":
        ddi_train = load_structure_files("train", "ddi")
        ddi_val = load_structure_files("val", "ddi")

        train_dataset = MixedLinkerDataset(
            pdb_train, ddi_train,
            pdb_weight=config.get("pdb_weight", 1.0),
            ddi_weight=config.get("ddi_weight", 1.0),
            **data_config,
        )
        val_dataset = MixedLinkerDataset(
            pdb_val, ddi_val,
            pdb_weight=config.get("pdb_weight", 1.0),
            ddi_weight=config.get("ddi_weight", 1.0),
            **data_config,
        )

        if rank == 0:
            logger.info(f"PDB train: {len(pdb_train)}, DDI train: {len(ddi_train)}")
    else:
        train_dataset = LinkerMultimerDataset(pdb_train, **data_config)
        val_dataset = LinkerMultimerDataset(pdb_val, **data_config)

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

    # Create model (loads pretrained ESMFold)
    model = ESMFoldWrapper(config.get("model", {}))
    model = model.to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # Optimizer - lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.get("learning_rate", 1e-5),
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
        logger.info(f"Resumed from epoch {start_epoch}")

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    patience = config.get("early_stopping_patience", 10)

    for epoch in range(start_epoch, config.get("max_epochs", 50)):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch, config)
        val_loss = validate(model, val_loader, device)

        if rank == 0:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            # Log to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss if val_loss >= best_val_loss else val_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            })

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
        wandb.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

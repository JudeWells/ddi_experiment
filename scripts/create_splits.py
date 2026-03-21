#!/usr/bin/env python3
"""
Create train/validation/test splits using MMseqs2 clustering.

Uses 40% sequence identity clustering to ensure no leakage between splits.
Ensures DDI training data doesn't leak into PDB test set.
"""

import argparse
import json
import logging
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Configuration
DDI_DATA_DIR = Path("/projects/u6bz/public/jude/processed_ddi")
PDB_DATA_DIR = Path("/projects/u6bz/public/jude")
SPLITS_DIR = Path("/projects/u6bz/jude/ddi_experiment/splits")

SEQUENCE_IDENTITY = 0.4  # 40% identity clustering
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/projects/u6bz/jude/ddi_experiment/logs/create_splits.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def extract_sequences_from_pdbs(pdb_dir: Path, output_fasta: Path, force_extract: bool = False) -> dict:
    """
    Extract sequences from PDB/CIF files with caching.

    If output_fasta exists and force_extract is False, loads from cache.
    Otherwise extracts sequences and saves to output_fasta.

    Returns dict mapping PDB ID to sequence.
    """
    # Check for cached sequences
    if output_fasta.exists() and not force_extract:
        logger.info(f"Loading cached sequences from {output_fasta}")
        sequences = {}
        with open(output_fasta, "r") as f:
            current_id = None
            current_seq = []
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id:
                        sequences[current_id] = "".join(current_seq)
                    current_id = line[1:]
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_id:
                sequences[current_id] = "".join(current_seq)
        logger.info(f"Loaded {len(sequences)} cached sequences")
        return sequences

    three_to_one = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }

    sequences = {}

    pdb_files = list(pdb_dir.glob("*.pdb")) + list(pdb_dir.glob("*.cif"))
    logger.info(f"Extracting sequences from {len(pdb_files)} PDB files...")

    for pdb_file in tqdm(pdb_files, desc="Extracting sequences"):
        pdb_id = pdb_file.stem
        is_cif = pdb_file.suffix.lower() == ".cif"

        residues = {}  # (chain, resnum) -> resname

        try:
            with open(pdb_file, "r") as f:
                for line in f:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        if is_cif:
                            # CIF format: space-separated fields
                            # ATOM 1 N N . MET A 1 1 ? ...
                            # Fields: 0=ATOM, 5=resname, 6=chain, 8=resnum
                            fields = line.split()
                            if len(fields) >= 9:
                                resname = fields[5]
                                chain = fields[6]
                                try:
                                    resnum = int(fields[8])
                                except ValueError:
                                    continue

                                if resname in three_to_one:
                                    key = (chain, resnum)
                                    if key not in residues:
                                        residues[key] = resname
                        else:
                            # PDB format: fixed columns
                            if len(line) > 26:
                                resname = line[17:20].strip()
                                chain = line[21]
                                try:
                                    resnum = int(line[22:26].strip())
                                except ValueError:
                                    continue

                                if resname in three_to_one:
                                    key = (chain, resnum)
                                    if key not in residues:
                                        residues[key] = resname

            if residues:
                # Sort by chain and residue number
                sorted_keys = sorted(residues.keys())
                sequence = "".join(
                    three_to_one.get(residues[k], "X") for k in sorted_keys
                )
                if len(sequence) >= 10:  # Minimum length
                    sequences[pdb_id] = sequence

        except Exception as e:
            logger.warning(f"Error processing {pdb_file}: {e}")

    # Write FASTA file (cache for next time)
    with open(output_fasta, "w") as f:
        for pdb_id, seq in sequences.items():
            f.write(f">{pdb_id}\n{seq}\n")

    logger.info(f"Extracted and cached {len(sequences)} sequences to {output_fasta}")
    return sequences


def run_mmseqs2_clustering(
    fasta_file: Path,
    output_dir: Path,
    identity: float = 0.5,
    coverage: float = 0.8,
) -> dict:
    """
    Run MMseqs2 clustering on sequences.

    Returns dict mapping sequence ID to cluster ID.
    """
    logger.info(f"Running MMseqs2 clustering at {identity*100}% identity...")

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create database
        db_path = tmpdir / "seqdb"
        subprocess.run(
            ["mmseqs", "createdb", str(fasta_file), str(db_path)],
            check=True,
            capture_output=True,
        )

        # Cluster
        cluster_path = tmpdir / "cluster"
        subprocess.run(
            [
                "mmseqs", "cluster",
                str(db_path),
                str(cluster_path),
                str(tmpdir),
                "--min-seq-id", str(identity),
                "-c", str(coverage),
                "--cov-mode", "0",
                "--cluster-mode", "0",
            ],
            check=True,
            capture_output=True,
        )

        # Create TSV output
        tsv_path = output_dir / "clusters.tsv"
        subprocess.run(
            [
                "mmseqs", "createtsv",
                str(db_path),
                str(db_path),
                str(cluster_path),
                str(tsv_path),
            ],
            check=True,
            capture_output=True,
        )

    # Parse cluster assignments
    cluster_assignments = {}
    cluster_members = defaultdict(list)

    with open(tsv_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                cluster_rep = parts[0]
                member = parts[1]
                cluster_assignments[member] = cluster_rep
                cluster_members[cluster_rep].append(member)

    logger.info(f"Created {len(cluster_members)} clusters from {len(cluster_assignments)} sequences")

    return cluster_assignments, cluster_members


def split_clusters(
    cluster_members: dict,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[set, set, set]:
    """
    Split clusters into train/val/test sets.

    Ensures all members of a cluster go to the same split.
    """
    np.random.seed(seed)

    cluster_ids = list(cluster_members.keys())
    np.random.shuffle(cluster_ids)

    n_clusters = len(cluster_ids)
    n_train = int(n_clusters * train_ratio)
    n_val = int(n_clusters * val_ratio)

    train_clusters = set(cluster_ids[:n_train])
    val_clusters = set(cluster_ids[n_train:n_train + n_val])
    test_clusters = set(cluster_ids[n_train + n_val:])

    train_ids = set()
    val_ids = set()
    test_ids = set()

    for cluster_id in train_clusters:
        train_ids.update(cluster_members[cluster_id])

    for cluster_id in val_clusters:
        val_ids.update(cluster_members[cluster_id])

    for cluster_id in test_clusters:
        test_ids.update(cluster_members[cluster_id])

    logger.info(f"Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    return train_ids, val_ids, test_ids


def check_leakage(
    ddi_train_ids: set,
    pdb_test_ids: set,
    ddi_to_cluster: dict,
    pdb_to_cluster: dict,
) -> tuple[set, set]:
    """
    Check for sequence leakage between DDI training and PDB test.

    Returns sets of leaked IDs.
    """
    # Get cluster IDs for each set
    ddi_train_clusters = {ddi_to_cluster.get(id_) for id_ in ddi_train_ids if id_ in ddi_to_cluster}
    pdb_test_clusters = {pdb_to_cluster.get(id_) for id_ in pdb_test_ids if id_ in pdb_to_cluster}

    # Find overlapping clusters
    overlapping_clusters = ddi_train_clusters & pdb_test_clusters

    if overlapping_clusters:
        logger.warning(f"Found {len(overlapping_clusters)} overlapping clusters!")

        leaked_ddi = {id_ for id_ in ddi_train_ids if ddi_to_cluster.get(id_) in overlapping_clusters}
        leaked_pdb = {id_ for id_ in pdb_test_ids if pdb_to_cluster.get(id_) in overlapping_clusters}

        return leaked_ddi, leaked_pdb

    logger.info("No sequence leakage detected between DDI train and PDB test")
    return set(), set()


def main():
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits using MMseqs2 clustering"
    )
    parser.add_argument(
        "--identity",
        type=float,
        default=SEQUENCE_IDENTITY,
        help=f"Sequence identity threshold (default: {SEQUENCE_IDENTITY})",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
        help=f"Training set ratio (default: {TRAIN_RATIO})",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
        help=f"Validation set ratio (default: {VAL_RATIO})",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=TEST_RATIO,
        help=f"Test set ratio (default: {TEST_RATIO})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SPLITS_DIR,
        help=f"Output directory (default: {SPLITS_DIR})",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Force re-extraction of PDB sequences (ignore cache)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load DDI sequences
    logger.info("=" * 50)
    logger.info("Step 1: Loading DDI sequences")
    logger.info("=" * 50)

    ddi_fasta = DDI_DATA_DIR / "domain_sequences.fasta"
    if not ddi_fasta.exists():
        logger.error(f"DDI sequences not found at {ddi_fasta}")
        logger.error("Run process_ddi_data.py first")
        return

    ddi_sequences = {}
    with open(ddi_fasta, "r") as f:
        current_id = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    ddi_sequences[current_id] = "".join(current_seq)
                current_id = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id:
            ddi_sequences[current_id] = "".join(current_seq)

    logger.info(f"Loaded {len(ddi_sequences)} DDI sequences")

    # Step 2: Extract PDB sequences (if needed)
    logger.info("=" * 50)
    logger.info("Step 2: Extracting PDB sequences")
    logger.info("=" * 50)

    pdb_fasta = args.output_dir / "pdb_sequences.fasta"

    pdb_sequences = {}
    for pdb_type in ["pdb_monomers", "pdb_multimers"]:
        pdb_dir = PDB_DATA_DIR / pdb_type
        if pdb_dir.exists():
            type_fasta = args.output_dir / f"{pdb_type}_sequences.fasta"
            type_seqs = extract_sequences_from_pdbs(pdb_dir, type_fasta, force_extract=args.force_extract)
            pdb_sequences.update(type_seqs)

    # Combine all sequences for clustering
    all_sequences = {**ddi_sequences, **pdb_sequences}
    all_fasta = args.output_dir / "all_sequences.fasta"

    with open(all_fasta, "w") as f:
        for seq_id, seq in all_sequences.items():
            f.write(f">{seq_id}\n{seq}\n")

    logger.info(f"Total sequences for clustering: {len(all_sequences)}")

    # Step 3: Run MMseqs2 clustering
    logger.info("=" * 50)
    logger.info("Step 3: Running MMseqs2 clustering")
    logger.info("=" * 50)

    cluster_assignments, cluster_members = run_mmseqs2_clustering(
        all_fasta,
        args.output_dir / "mmseqs2_clusters",
        identity=args.identity,
    )

    # Step 4: Create DDI splits
    logger.info("=" * 50)
    logger.info("Step 4: Creating DDI splits")
    logger.info("=" * 50)

    # Get clusters containing DDI sequences
    ddi_cluster_members = {}
    for cluster_id, members in cluster_members.items():
        ddi_members = [m for m in members if m in ddi_sequences]
        if ddi_members:
            ddi_cluster_members[cluster_id] = ddi_members

    ddi_train, ddi_val, ddi_test = split_clusters(
        ddi_cluster_members,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    # Step 5: Create PDB splits
    logger.info("=" * 50)
    logger.info("Step 5: Creating PDB splits")
    logger.info("=" * 50)

    # Get clusters containing PDB sequences (excluding clusters in DDI train)
    ddi_train_clusters = {cluster_assignments.get(id_) for id_ in ddi_train if id_ in cluster_assignments}

    pdb_cluster_members = {}
    for cluster_id, members in cluster_members.items():
        # Exclude clusters that overlap with DDI training
        if cluster_id in ddi_train_clusters:
            continue
        pdb_members = [m for m in members if m in pdb_sequences]
        if pdb_members:
            pdb_cluster_members[cluster_id] = pdb_members

    pdb_train, pdb_val, pdb_test = split_clusters(
        pdb_cluster_members,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    # Step 6: Check for leakage
    logger.info("=" * 50)
    logger.info("Step 6: Checking for leakage")
    logger.info("=" * 50)

    leaked_ddi, leaked_pdb = check_leakage(
        ddi_train, pdb_test, cluster_assignments, cluster_assignments
    )

    if leaked_ddi or leaked_pdb:
        logger.warning(f"Removing {len(leaked_ddi)} leaked DDI and {len(leaked_pdb)} leaked PDB entries")
        ddi_train = ddi_train - leaked_ddi
        pdb_test = pdb_test - leaked_pdb

    # Step 7: Save splits
    logger.info("=" * 50)
    logger.info("Step 7: Saving splits")
    logger.info("=" * 50)

    splits = {
        "ddi_train": list(ddi_train),
        "ddi_val": list(ddi_val),
        "ddi_test": list(ddi_test),
        "pdb_train": list(pdb_train),
        "pdb_val": list(pdb_val),
        "pdb_test": list(pdb_test),
    }

    for split_name, ids in splits.items():
        split_file = args.output_dir / f"{split_name}.txt"
        with open(split_file, "w") as f:
            for id_ in sorted(ids):
                f.write(f"{id_}\n")
        logger.info(f"Saved {len(ids)} IDs to {split_file}")

    # Save pair-level splits for DDI
    ddi_pairs = pd.read_csv(DDI_DATA_DIR / "filtered_pairs.csv")

    ddi_train_pairs = ddi_pairs[
        ddi_pairs["domain1_id"].isin(ddi_train) &
        ddi_pairs["domain2_id"].isin(ddi_train)
    ]
    ddi_val_pairs = ddi_pairs[
        ddi_pairs["domain1_id"].isin(ddi_val) &
        ddi_pairs["domain2_id"].isin(ddi_val)
    ]
    ddi_test_pairs = ddi_pairs[
        ddi_pairs["domain1_id"].isin(ddi_test) &
        ddi_pairs["domain2_id"].isin(ddi_test)
    ]

    ddi_train_pairs.to_csv(args.output_dir / "ddi_train_pairs.csv", index=False)
    ddi_val_pairs.to_csv(args.output_dir / "ddi_val_pairs.csv", index=False)
    ddi_test_pairs.to_csv(args.output_dir / "ddi_test_pairs.csv", index=False)

    logger.info(f"DDI pairs: train={len(ddi_train_pairs)}, val={len(ddi_val_pairs)}, test={len(ddi_test_pairs)}")

    # Save metadata
    metadata = {
        "sequence_identity": args.identity,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "total_sequences": len(all_sequences),
        "total_clusters": len(cluster_members),
        "ddi_train": len(ddi_train),
        "ddi_val": len(ddi_val),
        "ddi_test": len(ddi_test),
        "pdb_train": len(pdb_train),
        "pdb_val": len(pdb_val),
        "pdb_test": len(pdb_test),
        "ddi_train_pairs": len(ddi_train_pairs),
        "ddi_val_pairs": len(ddi_val_pairs),
        "ddi_test_pairs": len(ddi_test_pairs),
    }

    with open(args.output_dir / "split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("=" * 50)
    logger.info("Split Summary")
    logger.info("=" * 50)
    for key, value in metadata.items():
        logger.info(f"{key}: {value}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Create train/validation/test splits using benchmark test sets.

Test set: Docking Benchmark 5.5 + CASP15 Multimers
Validation: 500 clusters at 30% identity from remaining data
Train: Everything else with >30% identity to test set filtered out
"""

import argparse
import json
import logging
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Optional
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

# Configuration
DDI_DATA_DIR = Path("/projects/u6bz/public/jude/processed_ddi")
PDB_DATA_DIR = Path("/projects/u6bz/public/jude")
SPLITS_DIR = Path("/projects/u6bz/jude/ddi_experiment/splits")
BENCHMARK_TEST_FILE = SPLITS_DIR / "benchmark_test_pdb_ids.txt"

SEQUENCE_IDENTITY = 0.3  # 30% identity for filtering and clustering
MIN_VAL_PAIRS = 100  # Minimum number of DDI pairs for validation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_fasta(fasta_path: Path) -> dict:
    """Load sequences from FASTA file."""
    sequences = {}
    with open(fasta_path, "r") as f:
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
    return sequences


def write_fasta(sequences: dict, fasta_path: Path):
    """Write sequences to FASTA file."""
    with open(fasta_path, "w") as f:
        for seq_id, seq in sequences.items():
            f.write(f">{seq_id}\n{seq}\n")


def run_mmseqs2_search(query_fasta: Path, target_fasta: Path, output_dir: Path,
                        identity: float = 0.3) -> set:
    """
    Find all target sequences with >identity similarity to any query sequence.
    Returns set of target sequence IDs that match.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create databases
        query_db = tmpdir / "querydb"
        target_db = tmpdir / "targetdb"
        result_db = tmpdir / "resultdb"

        subprocess.run(
            ["mmseqs", "createdb", str(query_fasta), str(query_db)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["mmseqs", "createdb", str(target_fasta), str(target_db)],
            check=True, capture_output=True,
        )

        # Search
        subprocess.run(
            [
                "mmseqs", "search",
                str(query_db), str(target_db), str(result_db), str(tmpdir),
                "--min-seq-id", str(identity),
                "-c", "0.8",
                "--cov-mode", "0",
                "-s", "7.5",
            ],
            check=True, capture_output=True,
        )

        # Convert to TSV
        tsv_path = output_dir / "search_results.tsv"
        subprocess.run(
            ["mmseqs", "convertalis", str(query_db), str(target_db),
             str(result_db), str(tsv_path)],
            check=True, capture_output=True,
        )

        # Parse results - get all target IDs that matched
        matched_targets = set()
        if tsv_path.exists():
            with open(tsv_path, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        matched_targets.add(parts[1])  # target ID

        return matched_targets


def run_mmseqs2_clustering(fasta_file: Path, output_dir: Path,
                           identity: float = 0.3) -> tuple[dict, dict]:
    """
    Run MMseqs2 clustering on sequences.
    Returns (sequence_to_cluster, cluster_to_members) dicts.
    """
    logger.info(f"Running MMseqs2 clustering at {identity*100}% identity...")

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create database
        db_path = tmpdir / "seqdb"
        subprocess.run(
            ["mmseqs", "createdb", str(fasta_file), str(db_path)],
            check=True, capture_output=True,
        )

        # Cluster
        cluster_path = tmpdir / "cluster"
        subprocess.run(
            [
                "mmseqs", "cluster",
                str(db_path), str(cluster_path), str(tmpdir),
                "--min-seq-id", str(identity),
                "-c", "0.8",
                "--cov-mode", "0",
                "--cluster-mode", "0",
            ],
            check=True, capture_output=True,
        )

        # Create TSV output
        tsv_path = output_dir / "clusters.tsv"
        subprocess.run(
            ["mmseqs", "createtsv", str(db_path), str(db_path),
             str(cluster_path), str(tsv_path)],
            check=True, capture_output=True,
        )

        # Parse clusters
        sequence_to_cluster = {}
        cluster_to_members = defaultdict(list)

        with open(tsv_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    cluster_rep = parts[0]
                    member = parts[1]
                    sequence_to_cluster[member] = cluster_rep
                    cluster_to_members[cluster_rep].append(member)

        logger.info(f"Created {len(cluster_to_members)} clusters from {len(sequence_to_cluster)} sequences")

        return sequence_to_cluster, dict(cluster_to_members)


def main():
    parser = argparse.ArgumentParser(
        description="Create benchmark-based train/val/test splits"
    )
    parser.add_argument(
        "--identity", type=float, default=SEQUENCE_IDENTITY,
        help=f"Sequence identity threshold (default: {SEQUENCE_IDENTITY})",
    )
    parser.add_argument(
        "--min-val-pairs", type=int, default=MIN_VAL_PAIRS,
        help=f"Minimum DDI pairs for validation (default: {MIN_VAL_PAIRS})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=SPLITS_DIR,
        help=f"Output directory (default: {SPLITS_DIR})",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Step 1: Load benchmark test set PDB IDs
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 1: Loading benchmark test set")
    logger.info("=" * 60)

    benchmark_pdb_ids = set()
    with open(BENCHMARK_TEST_FILE, "r") as f:
        for line in f:
            pdb_id = line.strip().lower()
            if pdb_id:
                benchmark_pdb_ids.add(pdb_id)

    logger.info(f"Loaded {len(benchmark_pdb_ids)} benchmark test PDB IDs")

    # =========================================================================
    # Step 2: Load all sequences
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 2: Loading all sequences")
    logger.info("=" * 60)

    # Load DDI sequences
    ddi_fasta = DDI_DATA_DIR / "domain_sequences.fasta"
    ddi_sequences = load_fasta(ddi_fasta)
    logger.info(f"Loaded {len(ddi_sequences)} DDI sequences")

    # Load PDB sequences (from cache)
    pdb_sequences = {}
    for pdb_type in ["pdb_monomers", "pdb_multimers"]:
        cache_fasta = args.output_dir / f"{pdb_type}_sequences.fasta"
        if cache_fasta.exists():
            type_seqs = load_fasta(cache_fasta)
            pdb_sequences.update(type_seqs)
            logger.info(f"Loaded {len(type_seqs)} {pdb_type} sequences from cache")

    logger.info(f"Total PDB sequences: {len(pdb_sequences)}")

    # =========================================================================
    # Step 3: Create test set from benchmark PDBs
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 3: Creating test set from benchmark PDBs")
    logger.info("=" * 60)

    # Find PDB sequences that are in benchmark
    test_pdb_sequences = {}
    for seq_id, seq in pdb_sequences.items():
        pdb_id = seq_id.lower()[:4]  # First 4 characters are PDB ID
        if pdb_id in benchmark_pdb_ids:
            test_pdb_sequences[seq_id] = seq

    logger.info(f"Found {len(test_pdb_sequences)} sequences from benchmark PDBs in our dataset")

    # Write test sequences to file
    test_fasta = args.output_dir / "test_sequences.fasta"
    write_fasta(test_pdb_sequences, test_fasta)

    # =========================================================================
    # Step 4: Find all sequences with >30% identity to test set
    # =========================================================================
    logger.info("=" * 60)
    logger.info(f"Step 4: Finding sequences with >{args.identity*100}% identity to test set")
    logger.info("=" * 60)

    # Combine all non-test sequences
    all_train_candidate_sequences = {}

    # Add DDI sequences
    all_train_candidate_sequences.update(ddi_sequences)

    # Add PDB sequences not in test
    for seq_id, seq in pdb_sequences.items():
        if seq_id not in test_pdb_sequences:
            all_train_candidate_sequences[seq_id] = seq

    logger.info(f"Total candidate sequences for train/val: {len(all_train_candidate_sequences)}")

    # Write candidate sequences
    candidate_fasta = args.output_dir / "train_candidate_sequences.fasta"
    write_fasta(all_train_candidate_sequences, candidate_fasta)

    # Search for sequences similar to test set
    logger.info("Searching for sequences similar to test set...")
    similar_to_test = run_mmseqs2_search(
        test_fasta,
        candidate_fasta,
        args.output_dir / "test_similarity_search",
        identity=args.identity,
    )

    logger.info(f"Found {len(similar_to_test)} sequences with >{args.identity*100}% identity to test set")

    # Filter out similar sequences
    filtered_sequences = {
        seq_id: seq for seq_id, seq in all_train_candidate_sequences.items()
        if seq_id not in similar_to_test
    }

    logger.info(f"Remaining sequences after filtering: {len(filtered_sequences)}")

    # =========================================================================
    # Step 5: Cluster remaining sequences at 30% identity
    # =========================================================================
    logger.info("=" * 60)
    logger.info(f"Step 5: Clustering at {args.identity*100}% identity")
    logger.info("=" * 60)

    filtered_fasta = args.output_dir / "filtered_sequences.fasta"
    write_fasta(filtered_sequences, filtered_fasta)

    seq_to_cluster, cluster_to_members = run_mmseqs2_clustering(
        filtered_fasta,
        args.output_dir / "mmseqs2_clusters_30",
        identity=args.identity,
    )

    # =========================================================================
    # Step 6: Select clusters for validation (ensuring enough DDI pairs)
    # =========================================================================
    logger.info("=" * 60)
    logger.info(f"Step 6: Selecting clusters for validation (target: {args.min_val_pairs} DDI pairs)")
    logger.info("=" * 60)

    # Load DDI pairs to guide cluster selection
    import pandas as pd
    ddi_pairs_file = DDI_DATA_DIR / "filtered_pairs.csv"
    ddi_pairs_df = pd.read_csv(ddi_pairs_file)

    # Filter to pairs where both domains are in filtered sequences
    valid_pairs = []
    for _, row in ddi_pairs_df.iterrows():
        d1, d2 = row["domain1_id"], row["domain2_id"]
        if d1 in filtered_sequences and d2 in filtered_sequences:
            valid_pairs.append((d1, d2))

    logger.info(f"Valid DDI pairs (both domains in filtered set): {len(valid_pairs)}")

    # Map domains to clusters
    domain_to_cluster = {}
    for seq_id in filtered_sequences:
        if seq_id in ddi_sequences:
            domain_to_cluster[seq_id] = seq_to_cluster.get(seq_id)

    # Find pairs where both domains are in the same cluster (easiest to keep together)
    same_cluster_pairs = []
    diff_cluster_pairs = []
    for d1, d2 in valid_pairs:
        c1 = domain_to_cluster.get(d1)
        c2 = domain_to_cluster.get(d2)
        if c1 and c2:
            if c1 == c2:
                same_cluster_pairs.append((d1, d2, c1))
            else:
                diff_cluster_pairs.append((d1, d2, c1, c2))

    logger.info(f"Same-cluster pairs: {len(same_cluster_pairs)}")
    logger.info(f"Different-cluster pairs: {len(diff_cluster_pairs)}")

    # Strategy: Select clusters that contain many same-cluster pairs first,
    # then add cluster pairs for different-cluster pairs

    # Count pairs per cluster
    cluster_pair_count = defaultdict(int)
    for d1, d2, c in same_cluster_pairs:
        cluster_pair_count[c] += 1

    # Sort clusters by number of pairs they contain
    sorted_clusters = sorted(cluster_pair_count.items(), key=lambda x: -x[1])

    # Select clusters greedily until we have enough pairs
    val_cluster_ids = set()
    val_pair_count = 0

    for cluster_id, pair_count in sorted_clusters:
        if val_pair_count >= args.min_val_pairs:
            break
        val_cluster_ids.add(cluster_id)
        val_pair_count += pair_count

    logger.info(f"Selected {len(val_cluster_ids)} clusters with {val_pair_count} same-cluster pairs")

    # If we don't have enough, also add cluster pairs from different-cluster pairs
    if val_pair_count < args.min_val_pairs:
        # Count how many pairs would be added by including each cluster pair
        cluster_pair_additions = defaultdict(int)
        for d1, d2, c1, c2 in diff_cluster_pairs:
            key = tuple(sorted([c1, c2]))
            cluster_pair_additions[key] += 1

        sorted_cluster_pairs = sorted(cluster_pair_additions.items(), key=lambda x: -x[1])

        for (c1, c2), pair_count in sorted_cluster_pairs:
            if val_pair_count >= args.min_val_pairs:
                break
            if c1 not in val_cluster_ids or c2 not in val_cluster_ids:
                val_cluster_ids.add(c1)
                val_cluster_ids.add(c2)
                val_pair_count += pair_count

    train_cluster_ids = set(cluster_to_members.keys()) - val_cluster_ids

    logger.info(f"Final validation clusters: {len(val_cluster_ids)}")
    logger.info(f"Training clusters: {len(train_cluster_ids)}")
    logger.info(f"Expected validation pairs: ~{val_pair_count}")

    # =========================================================================
    # Step 7: Assign sequences to splits
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 7: Assigning sequences to splits")
    logger.info("=" * 60)

    train_sequences = set()
    val_sequences = set()
    test_sequences = set(test_pdb_sequences.keys())

    for seq_id in filtered_sequences:
        cluster_id = seq_to_cluster.get(seq_id)
        if cluster_id in val_cluster_ids:
            val_sequences.add(seq_id)
        else:
            train_sequences.add(seq_id)

    # Separate DDI and PDB sequences
    ddi_train = train_sequences & set(ddi_sequences.keys())
    ddi_val = val_sequences & set(ddi_sequences.keys())
    ddi_test = set()  # DDI not in test set

    pdb_train = train_sequences & set(pdb_sequences.keys())
    pdb_val = val_sequences & set(pdb_sequences.keys())
    pdb_test = test_sequences

    logger.info(f"DDI - Train: {len(ddi_train)}, Val: {len(ddi_val)}, Test: {len(ddi_test)}")
    logger.info(f"PDB - Train: {len(pdb_train)}, Val: {len(pdb_val)}, Test: {len(pdb_test)}")

    # =========================================================================
    # Step 8: Save splits
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 8: Saving splits")
    logger.info("=" * 60)

    def save_ids(ids: set, path: Path):
        with open(path, "w") as f:
            for id_ in sorted(ids):
                f.write(f"{id_}\n")
        logger.info(f"Saved {len(ids)} IDs to {path}")

    save_ids(ddi_train, args.output_dir / "ddi_train.txt")
    save_ids(ddi_val, args.output_dir / "ddi_val.txt")
    save_ids(ddi_test, args.output_dir / "ddi_test.txt")
    save_ids(pdb_train, args.output_dir / "pdb_train.txt")
    save_ids(pdb_val, args.output_dir / "pdb_val.txt")
    save_ids(pdb_test, args.output_dir / "pdb_test.txt")

    # Also save the excluded sequences (similar to test)
    save_ids(similar_to_test, args.output_dir / "excluded_similar_to_test.txt")

    # =========================================================================
    # Step 9: Create DDI pair splits
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Step 9: Creating DDI pair splits")
    logger.info("=" * 60)

    # Load DDI pairs
    import pandas as pd
    ddi_pairs_file = DDI_DATA_DIR / "filtered_pairs.csv"
    ddi_pairs = pd.read_csv(ddi_pairs_file)

    def get_pair_split(row, train_ids, val_ids):
        """Assign pair to split based on both domains."""
        d1 = row["domain1_id"]
        d2 = row["domain2_id"]

        # Both must be in same split
        if d1 in train_ids and d2 in train_ids:
            return "train"
        elif d1 in val_ids and d2 in val_ids:
            return "val"
        elif d1 in train_ids and d2 in val_ids:
            return "mixed"
        elif d1 in val_ids and d2 in train_ids:
            return "mixed"
        else:
            return "excluded"

    ddi_pairs["split"] = ddi_pairs.apply(
        lambda row: get_pair_split(row, ddi_train, ddi_val), axis=1
    )

    train_pairs = ddi_pairs[ddi_pairs["split"] == "train"]
    val_pairs = ddi_pairs[ddi_pairs["split"] == "val"]

    logger.info(f"DDI pairs - Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Save pair splits
    train_pairs.to_csv(args.output_dir / "ddi_train_pairs.csv", index=False)
    val_pairs.to_csv(args.output_dir / "ddi_val_pairs.csv", index=False)

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Split Summary")
    logger.info("=" * 60)

    summary = {
        "sequence_identity": args.identity,
        "min_val_pairs": args.min_val_pairs,
        "val_clusters_selected": len(val_cluster_ids),
        "seed": args.seed,
        "benchmark_test_pdbs": len(benchmark_pdb_ids),
        "test_sequences_found": len(test_pdb_sequences),
        "excluded_similar_to_test": len(similar_to_test),
        "total_clusters": len(cluster_to_members),
        "ddi_train": len(ddi_train),
        "ddi_val": len(ddi_val),
        "ddi_test": len(ddi_test),
        "pdb_train": len(pdb_train),
        "pdb_val": len(pdb_val),
        "pdb_test": len(pdb_test),
        "ddi_train_pairs": len(train_pairs),
        "ddi_val_pairs": len(val_pairs),
    }

    for key, value in summary.items():
        logger.info(f"{key}: {value}")

    with open(args.output_dir / "benchmark_split_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Done!")


if __name__ == "__main__":
    main()

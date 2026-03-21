#!/usr/bin/env python3
"""
Process DDI (Domain-Domain Interface) data from AlphaFold Database.

Filters DDI pairs by pLDDT > 70 (high-confidence predictions only).
Reads from Erik's AFDB DDI data and outputs processed data.

Source: /projects/u6bz/public/erik/AFDDI_data/AFDB_DDI/
Output: /projects/u6bz/public/jude/processed_ddi/
"""

import argparse
import json
import logging
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Configuration
DDI_SOURCE_DIR = Path("/projects/u6bz/public/erik/AFDDI_data/AFDB_DDI")
DDI_DOMPDBS_DIR = DDI_SOURCE_DIR / "dompdbs"
DDI_PAIRS_FILE = DDI_SOURCE_DIR / "posi_all.csv"

OUTPUT_DIR = Path("/projects/u6bz/public/jude/processed_ddi")
PLDDT_THRESHOLD = 70.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/projects/u6bz/jude/ddi_experiment/logs/process_ddi.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def parse_pdb_plddt(pdb_file: Path) -> Optional[float]:
    """
    Extract mean pLDDT from PDB file B-factor column.

    In AlphaFold PDBs, the B-factor column contains pLDDT values.
    """
    plddt_values = []

    try:
        with open(pdb_file, "r") as f:
            for line in f:
                if line.startswith("ATOM"):
                    # B-factor is in columns 61-66 (0-indexed: 60-66)
                    try:
                        bfactor = float(line[60:66].strip())
                        plddt_values.append(bfactor)
                    except (ValueError, IndexError):
                        continue

        if plddt_values:
            return np.mean(plddt_values)
        return None

    except Exception as e:
        logger.warning(f"Error parsing {pdb_file}: {e}")
        return None


def get_domain_sequence(pdb_file: Path) -> Optional[str]:
    """Extract amino acid sequence from PDB file."""
    three_to_one = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }

    residues = {}  # (chain, resnum) -> resname

    try:
        with open(pdb_file, "r") as f:
            for line in f:
                if line.startswith("ATOM"):
                    chain = line[21]
                    resnum = int(line[22:26].strip())
                    resname = line[17:20].strip()
                    residues[(chain, resnum)] = resname

        if not residues:
            return None

        # Sort by chain and residue number
        sorted_keys = sorted(residues.keys())
        sequence = "".join(
            three_to_one.get(residues[k], "X") for k in sorted_keys
        )

        return sequence

    except Exception as e:
        logger.warning(f"Error extracting sequence from {pdb_file}: {e}")
        return None


def find_domain_pdb(domain_id: str, source_dir: Path) -> Optional[Path]:
    """Find domain PDB file, searching in subdirectories if needed."""
    # Try direct path first
    pdb_file = source_dir / f"{domain_id}.pdb"
    if pdb_file.exists():
        return pdb_file

    # Search in numbered subdirectories (00/, 01/, etc.)
    for subdir in source_dir.iterdir():
        if subdir.is_dir() and subdir.name.isdigit():
            pdb_file = subdir / f"{domain_id}.pdb"
            if pdb_file.exists():
                return pdb_file

    return None


def process_domain(domain_id: str, source_dir: Path) -> Optional[dict]:
    """
    Process a single domain: check pLDDT and extract metadata.

    Args:
        domain_id: Domain identifier (e.g., "A0A024U300_D1")
        source_dir: Directory containing domain PDB files

    Returns:
        Dictionary with domain metadata or None if doesn't pass filters
    """
    # Find PDB file
    pdb_file = find_domain_pdb(domain_id, source_dir)

    if pdb_file is None:
        return None

    # Parse pLDDT
    plddt = parse_pdb_plddt(pdb_file)

    if plddt is None or plddt < PLDDT_THRESHOLD:
        return None

    # Extract sequence
    sequence = get_domain_sequence(pdb_file)

    if sequence is None or len(sequence) < 10:  # Minimum domain length
        return None

    return {
        "domain_id": domain_id,
        "pdb_file": str(pdb_file),
        "plddt": plddt,
        "sequence": sequence,
        "length": len(sequence),
    }


def process_pair(
    row: pd.Series,
    source_dir: Path,
    domain_cache: dict,
) -> Optional[dict]:
    """
    Process a DDI pair: validate both domains pass pLDDT filter.

    Args:
        row: Row from posi_all.csv with domain pair info
        source_dir: Directory containing domain PDB files
        domain_cache: Cache of already-processed domains

    Returns:
        Dictionary with pair metadata or None if doesn't pass filters
    """
    # Extract domain IDs from the row
    # Handle PAIRID format: "domain1:domain2"
    if "PAIRID" in row.index:
        pair_id = row["PAIRID"]
        if ":" in str(pair_id):
            parts = str(pair_id).split(":")
            domain1_id = parts[0]
            domain2_id = parts[1] if len(parts) > 1 else None
        else:
            domain1_id = pair_id
            domain2_id = None
    else:
        # Try common column names
        domain1_id = None
        domain2_id = None

        for col in ["domain1", "dom1", "id1", "domain_1"]:
            if col in row.index:
                domain1_id = row[col]
                break

        for col in ["domain2", "dom2", "id2", "domain_2"]:
            if col in row.index:
                domain2_id = row[col]
                break

        if domain1_id is None or domain2_id is None:
            # Try using first two columns
            domain1_id = row.iloc[0]
            domain2_id = row.iloc[1]

    # Process domains (use cache)
    if domain1_id not in domain_cache:
        domain_cache[domain1_id] = process_domain(domain1_id, source_dir)

    if domain2_id not in domain_cache:
        domain_cache[domain2_id] = process_domain(domain2_id, source_dir)

    domain1 = domain_cache.get(domain1_id)
    domain2 = domain_cache.get(domain2_id)

    # Both domains must pass pLDDT filter
    if domain1 is None or domain2 is None:
        return None

    return {
        "domain1_id": domain1_id,
        "domain2_id": domain2_id,
        "domain1_plddt": domain1["plddt"],
        "domain2_plddt": domain2["plddt"],
        "domain1_length": domain1["length"],
        "domain2_length": domain2["length"],
        "domain1_sequence": domain1["sequence"],
        "domain2_sequence": domain2["sequence"],
    }


def copy_filtered_pdbs(
    filtered_pairs: pd.DataFrame,
    source_dir: Path,
    output_dir: Path,
):
    """Copy filtered domain PDBs to output directory."""
    domains_dir = output_dir / "domains"
    domains_dir.mkdir(parents=True, exist_ok=True)

    # Get unique domains
    all_domains = set(filtered_pairs["domain1_id"].tolist() +
                      filtered_pairs["domain2_id"].tolist())

    logger.info(f"Copying {len(all_domains)} unique domain PDBs...")

    copied = 0
    for domain_id in tqdm(all_domains, desc="Copying PDBs"):
        src = source_dir / f"{domain_id}.pdb"
        dst = domains_dir / f"{domain_id}.pdb"

        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

    logger.info(f"Copied {copied} PDB files")


def main():
    parser = argparse.ArgumentParser(
        description="Process DDI data with pLDDT filtering"
    )
    parser.add_argument(
        "--plddt-threshold",
        type=float,
        default=PLDDT_THRESHOLD,
        help=f"pLDDT threshold for filtering (default: {PLDDT_THRESHOLD})",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DDI_DOMPDBS_DIR,
        help="Source directory with domain PDBs",
    )
    parser.add_argument(
        "--pairs-file",
        type=Path,
        default=DDI_PAIRS_FILE,
        help="CSV file with positive DDI pairs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of pairs to process (for testing)",
    )
    parser.add_argument(
        "--copy-pdbs",
        action="store_true",
        help="Copy filtered PDBs to output directory",
    )
    args = parser.parse_args()

    # Update threshold from args
    plddt_threshold = args.plddt_threshold

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load positive pairs
    logger.info(f"Loading DDI pairs from {args.pairs_file}...")

    try:
        pairs_df = pd.read_csv(args.pairs_file)
    except Exception as e:
        logger.error(f"Error loading pairs file: {e}")
        # Try different separators
        for sep in ["\t", " ", ";"]:
            try:
                pairs_df = pd.read_csv(args.pairs_file, sep=sep)
                logger.info(f"Loaded with separator: '{sep}'")
                break
            except Exception:
                continue
        else:
            raise ValueError(f"Could not load {args.pairs_file}")

    logger.info(f"Loaded {len(pairs_df)} DDI pairs")
    logger.info(f"Columns: {list(pairs_df.columns)}")

    if args.limit:
        pairs_df = pairs_df.head(args.limit)
        logger.info(f"Limited to {len(pairs_df)} pairs for testing")

    # Process pairs
    logger.info(f"Processing pairs with pLDDT > {plddt_threshold}...")

    domain_cache = {}
    filtered_pairs = []

    for idx, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Processing"):
        result = process_pair(row, args.source_dir, domain_cache)
        if result is not None:
            filtered_pairs.append(result)

    logger.info(f"Filtered to {len(filtered_pairs)} pairs (pLDDT > {plddt_threshold})")

    # Create output DataFrame
    filtered_df = pd.DataFrame(filtered_pairs)

    # Save filtered pairs
    output_pairs = args.output_dir / "filtered_pairs.csv"
    filtered_df.to_csv(output_pairs, index=False)
    logger.info(f"Saved filtered pairs to {output_pairs}")

    # Save sequences for clustering
    sequences_file = args.output_dir / "domain_sequences.fasta"
    with open(sequences_file, "w") as f:
        seen = set()
        for domain_id, info in domain_cache.items():
            if info is not None and domain_id not in seen:
                f.write(f">{domain_id}\n{info['sequence']}\n")
                seen.add(domain_id)
    logger.info(f"Saved {len(seen)} sequences to {sequences_file}")

    # Copy PDBs if requested
    if args.copy_pdbs:
        copy_filtered_pdbs(filtered_df, args.source_dir, args.output_dir)

    # Save statistics
    stats = {
        "total_pairs_input": len(pairs_df),
        "filtered_pairs_output": len(filtered_pairs),
        "unique_domains": len(domain_cache),
        "domains_passing_filter": sum(1 for v in domain_cache.values() if v is not None),
        "plddt_threshold": plddt_threshold,
        "mean_plddt_domain1": filtered_df["domain1_plddt"].mean() if len(filtered_df) > 0 else 0,
        "mean_plddt_domain2": filtered_df["domain2_plddt"].mean() if len(filtered_df) > 0 else 0,
    }

    with open(args.output_dir / "processing_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("=" * 50)
    logger.info("Processing Summary")
    logger.info("=" * 50)
    for key, value in stats.items():
        logger.info(f"{key}: {value}")


if __name__ == "__main__":
    main()

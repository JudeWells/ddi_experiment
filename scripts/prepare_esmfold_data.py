#!/usr/bin/env python3
"""
Prepare training data for ESMFold linker experiments.

Creates combined DDI pair PDB files by joining two domain structures.
The linker trick will be applied during training.
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

# Configuration
PROJECT_DIR = Path("/projects/u6bz/jude/ddi_experiment")
SPLITS_DIR = PROJECT_DIR / "splits"
DDI_SOURCE_DIR = Path("/projects/u6bz/public/jude/processed_ddi")
PDB_MULTIMER_DIR = Path("/projects/u6bz/public/jude/pdb_multimers")
PDB_MONOMER_DIR = Path("/projects/u6bz/public/jude/pdb_monomers")
OUTPUT_DIR = PROJECT_DIR / "training_data" / "esmfold"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_domain_index(source_dirs: list) -> dict:
    """Build an index of domain_id -> file path for fast lookup."""
    logger.info(f"Building domain file index from {len(source_dirs)} directories...")
    index = {}

    for source_dir in tqdm(source_dirs, desc="Indexing directories"):
        if not source_dir.is_dir():
            continue
        for pdb_file in source_dir.glob("*.pdb"):
            domain_id = pdb_file.stem  # filename without extension
            if domain_id not in index:
                index[domain_id] = pdb_file

    logger.info(f"Indexed {len(index)} domain PDB files")
    return index


def find_domain_pdb(domain_id: str, source_dir: Path) -> Optional[Path]:
    """Find domain PDB file, searching in subdirectories if needed."""
    # Direct path
    pdb_file = source_dir / f"{domain_id}.pdb"
    if pdb_file.exists():
        return pdb_file

    # Search in numbered subdirectories
    for subdir in source_dir.iterdir():
        if subdir.is_dir():
            pdb_file = subdir / f"{domain_id}.pdb"
            if pdb_file.exists():
                return pdb_file

    return None


def renumber_pdb_chain(pdb_content: str, chain_id: str, start_resnum: int = 1) -> tuple[str, int]:
    """
    Renumber residues in PDB content and change chain ID.
    Returns (new_content, last_resnum).
    """
    lines = []
    current_resnum = None
    new_resnum = start_resnum - 1
    resnum_map = {}

    for line in pdb_content.split("\n"):
        if line.startswith(("ATOM", "HETATM")):
            try:
                old_resnum = int(line[22:26].strip())
                if old_resnum not in resnum_map:
                    new_resnum += 1
                    resnum_map[old_resnum] = new_resnum

                mapped_resnum = resnum_map[old_resnum]

                # Rebuild line with new chain and resnum
                new_line = (
                    line[:21] +
                    chain_id +
                    f"{mapped_resnum:4d}" +
                    line[26:]
                )
                lines.append(new_line)
            except (ValueError, IndexError):
                lines.append(line)
        elif line.startswith(("TER", "END")):
            continue  # Skip TER/END, we'll add them at the end
        elif line.strip():
            lines.append(line)

    return "\n".join(lines), new_resnum


def combine_domain_pdbs(domain1_pdb: Path, domain2_pdb: Path, output_pdb: Path):
    """
    Combine two domain PDBs into a single multi-chain PDB file.

    Domain 1 becomes chain A, Domain 2 becomes chain B.
    """
    with open(domain1_pdb, "r") as f:
        content1 = f.read()

    with open(domain2_pdb, "r") as f:
        content2 = f.read()

    # Renumber chains
    chain_a, last_res_a = renumber_pdb_chain(content1, "A", start_resnum=1)
    chain_b, last_res_b = renumber_pdb_chain(content2, "B", start_resnum=1)

    # Combine with TER between chains
    combined = f"{chain_a}\nTER\n{chain_b}\nTER\nEND\n"

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pdb, "w") as f:
        f.write(combined)


def prepare_ddi_pairs(split: str = "train", domain_index: dict = None):
    """Prepare combined DDI pair files for a given split."""
    pairs_file = SPLITS_DIR / f"ddi_{split}_pairs.csv"

    if not pairs_file.exists():
        logger.warning(f"No pairs file found: {pairs_file}")
        return 0

    pairs_df = pd.read_csv(pairs_file)
    logger.info(f"Processing {len(pairs_df)} DDI {split} pairs...")

    output_dir = OUTPUT_DIR / "ddi_pairs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build domain index if not provided
    if domain_index is None:
        domain_source_dirs = []

        # Check processed_ddi directory structure
        ddi_pdbs_dir = DDI_SOURCE_DIR / "domains"
        if ddi_pdbs_dir.exists():
            domain_source_dirs.append(ddi_pdbs_dir)

        # Also check source AFDB DDI dompdbs directory (has numbered subdirs)
        afdb_dompdbs_dir = Path("/projects/u6bz/public/erik/AFDDI_data/AFDB_DDI/dompdbs")
        if afdb_dompdbs_dir.exists():
            # Add all numbered subdirectories
            for subdir in sorted(afdb_dompdbs_dir.iterdir()):
                if subdir.is_dir() and subdir.name.isdigit():
                    domain_source_dirs.append(subdir)

        if not domain_source_dirs:
            logger.error("No domain PDB source directories found!")
            return 0

        domain_index = build_domain_index(domain_source_dirs)

    success_count = 0
    missing_count = 0
    missing_domains = set()

    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc=f"Creating {split} pairs"):
        domain1_id = row["domain1_id"]
        domain2_id = row["domain2_id"]
        pair_id = f"{domain1_id}_{domain2_id}"
        output_file = output_dir / f"{pair_id}.pdb"

        if output_file.exists():
            success_count += 1
            continue

        # Find domain PDBs using index
        domain1_pdb = domain_index.get(domain1_id)
        domain2_pdb = domain_index.get(domain2_id)

        if domain1_pdb is None:
            missing_domains.add(domain1_id)
        if domain2_pdb is None:
            missing_domains.add(domain2_id)

        if domain1_pdb is None or domain2_pdb is None:
            missing_count += 1
            continue

        try:
            combine_domain_pdbs(domain1_pdb, domain2_pdb, output_file)
            success_count += 1
        except Exception as e:
            logger.warning(f"Error combining {pair_id}: {e}")
            missing_count += 1

    logger.info(f"Created {success_count} combined PDB files, {missing_count} missing")
    if missing_domains:
        logger.info(f"Missing {len(missing_domains)} unique domain PDBs")
    return success_count, domain_index


def verify_pdb_files(split: str = "train"):
    """Verify PDB files exist for the given split."""
    split_file = SPLITS_DIR / f"pdb_{split}.txt"

    if not split_file.exists():
        logger.warning(f"No split file found: {split_file}")
        return 0, 0

    with open(split_file, "r") as f:
        pdb_ids = [line.strip() for line in f if line.strip()]

    found = 0
    missing = []

    for pdb_id in pdb_ids:
        found_file = False
        for pdb_dir in [PDB_MULTIMER_DIR, PDB_MONOMER_DIR]:
            for suffix in [".pdb", ".cif"]:
                if (pdb_dir / f"{pdb_id}{suffix}").exists():
                    found_file = True
                    break
            if found_file:
                break

        if found_file:
            found += 1
        else:
            missing.append(pdb_id)

    logger.info(f"PDB {split}: {found}/{len(pdb_ids)} files found, {len(missing)} missing")

    if missing and len(missing) <= 10:
        logger.info(f"Missing: {missing}")

    return found, len(missing)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for ESMFold linker experiments"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"],
        help="Splits to prepare (default: train val)"
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only verify file existence, don't create files"
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build domain index once for all splits
    domain_index = None
    if not args.verify_only:
        domain_source_dirs = []

        # Check processed_ddi directory structure
        ddi_pdbs_dir = DDI_SOURCE_DIR / "domains"
        if ddi_pdbs_dir.exists():
            domain_source_dirs.append(ddi_pdbs_dir)

        # Also check source AFDB DDI dompdbs directory (has numbered subdirs)
        afdb_dompdbs_dir = Path("/projects/u6bz/public/erik/AFDDI_data/AFDB_DDI/dompdbs")
        if afdb_dompdbs_dir.exists():
            for subdir in sorted(afdb_dompdbs_dir.iterdir()):
                if subdir.is_dir() and subdir.name.isdigit():
                    domain_source_dirs.append(subdir)

        if domain_source_dirs:
            domain_index = build_domain_index(domain_source_dirs)

    for split in args.splits:
        logger.info("=" * 60)
        logger.info(f"Processing {split} split")
        logger.info("=" * 60)

        # Verify PDB files
        verify_pdb_files(split)

        if not args.verify_only:
            # Prepare DDI pairs
            result = prepare_ddi_pairs(split, domain_index)
            if isinstance(result, tuple):
                _, domain_index = result  # Reuse index

    logger.info("Done!")


if __name__ == "__main__":
    main()

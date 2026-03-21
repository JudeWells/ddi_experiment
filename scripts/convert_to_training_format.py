#!/usr/bin/env python3
"""
Convert processed DDI and PDB data to model-specific training formats.

Supports:
- OpenFold: mmCIF + features
- RoseTTAFold-All-Atom: atomworks format
- Protenix: mmCIF
"""

import argparse
import json
import logging
import os
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Attempt to import model-specific libraries
try:
    import atomworks
    ATOMWORKS_AVAILABLE = True
except ImportError:
    ATOMWORKS_AVAILABLE = False

try:
    from Bio.PDB import PDBParser, MMCIFIO
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

# Configuration
DDI_DATA_DIR = Path("/projects/u6bz/public/jude/processed_ddi")
PDB_MONOMER_DIR = Path("/projects/u6bz/public/jude/pdb_monomers")
PDB_MULTIMER_DIR = Path("/projects/u6bz/public/jude/pdb_multimers")
SPLITS_DIR = Path("/projects/u6bz/jude/ddi_experiment/splits")
OUTPUT_DIR = Path("/projects/u6bz/jude/ddi_experiment/training_data")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/projects/u6bz/jude/ddi_experiment/logs/convert_data.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class OpenFoldConverter:
    """Convert data to OpenFold SoloSeq format."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "openfold"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_single(self, pdb_file: Path, pair_id: str = None) -> Optional[Path]:
        """Convert a single PDB to OpenFold format."""
        try:
            # For SoloSeq, we need the sequence and structure
            output_subdir = self.output_dir / "structures"
            output_subdir.mkdir(parents=True, exist_ok=True)

            # Copy PDB file (OpenFold can use PDB directly)
            output_file = output_subdir / pdb_file.name
            if not output_file.exists():
                shutil.copy2(pdb_file, output_file)

            return output_file

        except Exception as e:
            logger.warning(f"Error converting {pdb_file} for OpenFold: {e}")
            return None

    def convert_ddi_pair(
        self,
        domain1_pdb: Path,
        domain2_pdb: Path,
        pair_id: str,
    ) -> Optional[Path]:
        """Convert a DDI pair to a combined structure."""
        try:
            output_subdir = self.output_dir / "ddi_pairs"
            output_subdir.mkdir(parents=True, exist_ok=True)

            output_file = output_subdir / f"{pair_id}.pdb"

            if output_file.exists():
                return output_file

            # Combine two domains into a single PDB with different chains
            with open(output_file, "w") as out_f:
                atom_num = 1

                # Write domain 1 as chain A
                with open(domain1_pdb, "r") as f:
                    for line in f:
                        if line.startswith("ATOM") or line.startswith("HETATM"):
                            # Replace chain with A
                            new_line = line[:21] + "A" + line[22:]
                            # Update atom number
                            new_line = f"{new_line[:6]}{atom_num:5d}{new_line[11:]}"
                            out_f.write(new_line)
                            atom_num += 1

                out_f.write("TER\n")

                # Write domain 2 as chain B
                with open(domain2_pdb, "r") as f:
                    for line in f:
                        if line.startswith("ATOM") or line.startswith("HETATM"):
                            # Replace chain with B
                            new_line = line[:21] + "B" + line[22:]
                            # Update atom number
                            new_line = f"{new_line[:6]}{atom_num:5d}{new_line[11:]}"
                            out_f.write(new_line)
                            atom_num += 1

                out_f.write("TER\n")
                out_f.write("END\n")

            return output_file

        except Exception as e:
            logger.warning(f"Error combining DDI pair {pair_id}: {e}")
            return None

    def create_feature_dict(self, pdb_file: Path) -> Optional[dict]:
        """Create OpenFold feature dictionary for SoloSeq."""
        try:
            # Extract sequence
            three_to_one = {
                "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
                "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
                "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
                "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
            }

            residues = {}
            coords = {}

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

                        coords[key][atom_name] = np.array([x, y, z])

            if not residues:
                return None

            sorted_keys = sorted(residues.keys())
            sequence = "".join(
                three_to_one.get(residues[k], "X") for k in sorted_keys
            )

            # Create minimal feature dict for SoloSeq
            features = {
                "sequence": sequence,
                "residue_index": np.arange(len(sequence)),
                "aatype": np.array([
                    list("ARNDCQEGHILKMFPSTWYV").index(aa) if aa in "ARNDCQEGHILKMFPSTWYV" else 20
                    for aa in sequence
                ]),
            }

            return features

        except Exception as e:
            logger.warning(f"Error creating features for {pdb_file}: {e}")
            return None


class RFAAConverter:
    """Convert data to RoseTTAFold-All-Atom format using atomworks."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "rfaa"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_single(self, pdb_file: Path) -> Optional[Path]:
        """Convert a single PDB to RFAA format."""
        if not ATOMWORKS_AVAILABLE:
            logger.warning("atomworks not available, copying PDB directly")
            output_file = self.output_dir / "structures" / pdb_file.name
            output_file.parent.mkdir(parents=True, exist_ok=True)
            if not output_file.exists():
                shutil.copy2(pdb_file, output_file)
            return output_file

        try:
            output_subdir = self.output_dir / "structures"
            output_subdir.mkdir(parents=True, exist_ok=True)

            # Use atomworks to parse and convert
            structure = atomworks.load_structure(str(pdb_file))

            output_file = output_subdir / f"{pdb_file.stem}.atomworks"

            if not output_file.exists():
                atomworks.save_structure(structure, str(output_file))

            return output_file

        except Exception as e:
            logger.warning(f"Error converting {pdb_file} for RFAA: {e}")
            # Fallback to copying PDB
            output_file = self.output_dir / "structures" / pdb_file.name
            output_file.parent.mkdir(parents=True, exist_ok=True)
            if not output_file.exists():
                shutil.copy2(pdb_file, output_file)
            return output_file

    def convert_ddi_pair(
        self,
        domain1_pdb: Path,
        domain2_pdb: Path,
        pair_id: str,
    ) -> Optional[Path]:
        """Convert a DDI pair for RFAA."""
        try:
            output_subdir = self.output_dir / "ddi_pairs"
            output_subdir.mkdir(parents=True, exist_ok=True)

            output_file = output_subdir / f"{pair_id}.pdb"

            if output_file.exists():
                return output_file

            # Combine domains similar to OpenFold
            with open(output_file, "w") as out_f:
                atom_num = 1

                for chain_id, pdb_path in [("A", domain1_pdb), ("B", domain2_pdb)]:
                    with open(pdb_path, "r") as f:
                        for line in f:
                            if line.startswith("ATOM") or line.startswith("HETATM"):
                                new_line = line[:21] + chain_id + line[22:]
                                new_line = f"{new_line[:6]}{atom_num:5d}{new_line[11:]}"
                                out_f.write(new_line)
                                atom_num += 1
                    out_f.write("TER\n")

                out_f.write("END\n")

            return output_file

        except Exception as e:
            logger.warning(f"Error combining DDI pair {pair_id} for RFAA: {e}")
            return None


class ProtenixConverter:
    """Convert data to Protenix (AlphaFold3) format."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "protenix"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def pdb_to_mmcif(self, pdb_file: Path, output_file: Path) -> bool:
        """Convert PDB to mmCIF format."""
        if not BIOPYTHON_AVAILABLE:
            logger.warning("Biopython not available, cannot convert to mmCIF")
            return False

        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("structure", str(pdb_file))

            io = MMCIFIO()
            io.set_structure(structure)
            io.save(str(output_file))

            return True

        except Exception as e:
            logger.warning(f"Error converting {pdb_file} to mmCIF: {e}")
            return False

    def convert_single(self, pdb_file: Path) -> Optional[Path]:
        """Convert a single PDB to Protenix format (mmCIF)."""
        try:
            output_subdir = self.output_dir / "structures"
            output_subdir.mkdir(parents=True, exist_ok=True)

            # Check if already mmCIF
            if pdb_file.suffix.lower() in [".cif", ".mmcif"]:
                output_file = output_subdir / pdb_file.name
                if not output_file.exists():
                    shutil.copy2(pdb_file, output_file)
                return output_file

            # Convert PDB to mmCIF
            output_file = output_subdir / f"{pdb_file.stem}.cif"

            if output_file.exists():
                return output_file

            if self.pdb_to_mmcif(pdb_file, output_file):
                return output_file

            return None

        except Exception as e:
            logger.warning(f"Error converting {pdb_file} for Protenix: {e}")
            return None

    def convert_ddi_pair(
        self,
        domain1_pdb: Path,
        domain2_pdb: Path,
        pair_id: str,
    ) -> Optional[Path]:
        """Convert a DDI pair for Protenix."""
        try:
            output_subdir = self.output_dir / "ddi_pairs"
            output_subdir.mkdir(parents=True, exist_ok=True)

            # First combine as PDB
            combined_pdb = output_subdir / f"{pair_id}.pdb"

            if not combined_pdb.exists():
                with open(combined_pdb, "w") as out_f:
                    atom_num = 1

                    for chain_id, pdb_path in [("A", domain1_pdb), ("B", domain2_pdb)]:
                        with open(pdb_path, "r") as f:
                            for line in f:
                                if line.startswith("ATOM") or line.startswith("HETATM"):
                                    new_line = line[:21] + chain_id + line[22:]
                                    new_line = f"{new_line[:6]}{atom_num:5d}{new_line[11:]}"
                                    out_f.write(new_line)
                                    atom_num += 1
                        out_f.write("TER\n")
                    out_f.write("END\n")

            # Convert to mmCIF
            output_file = output_subdir / f"{pair_id}.cif"

            if output_file.exists():
                return output_file

            if self.pdb_to_mmcif(combined_pdb, output_file):
                return output_file

            # Fallback to PDB if mmCIF conversion fails
            return combined_pdb

        except Exception as e:
            logger.warning(f"Error converting DDI pair {pair_id} for Protenix: {e}")
            return None


def process_ddi_pairs(
    pairs_csv: Path,
    domain_dir: Path,
    converters: dict,
    split_name: str,
) -> dict:
    """Process DDI pairs for all models."""
    results = {model: {"success": 0, "failed": 0} for model in converters}

    pairs_df = pd.read_csv(pairs_csv)
    logger.info(f"Processing {len(pairs_df)} {split_name} DDI pairs...")

    for idx, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc=f"DDI {split_name}"):
        domain1_id = row["domain1_id"]
        domain2_id = row["domain2_id"]
        pair_id = f"{domain1_id}_{domain2_id}"

        domain1_pdb = domain_dir / f"{domain1_id}.pdb"
        domain2_pdb = domain_dir / f"{domain2_id}.pdb"

        if not domain1_pdb.exists() or not domain2_pdb.exists():
            for model in converters:
                results[model]["failed"] += 1
            continue

        for model, converter in converters.items():
            result = converter.convert_ddi_pair(domain1_pdb, domain2_pdb, pair_id)
            if result:
                results[model]["success"] += 1
            else:
                results[model]["failed"] += 1

    return results


def process_pdb_structures(
    pdb_ids: list,
    pdb_dir: Path,
    converters: dict,
    split_name: str,
) -> dict:
    """Process PDB structures for all models."""
    results = {model: {"success": 0, "failed": 0} for model in converters}

    logger.info(f"Processing {len(pdb_ids)} {split_name} PDB structures...")

    for pdb_id in tqdm(pdb_ids, desc=f"PDB {split_name}"):
        # Find PDB file
        pdb_file = None
        for suffix in [".pdb", ".cif", ".PDB", ".CIF"]:
            candidate = pdb_dir / f"{pdb_id}{suffix}"
            if candidate.exists():
                pdb_file = candidate
                break

        if pdb_file is None:
            for model in converters:
                results[model]["failed"] += 1
            continue

        for model, converter in converters.items():
            result = converter.convert_single(pdb_file)
            if result:
                results[model]["success"] += 1
            else:
                results[model]["failed"] += 1

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert data to model-specific training formats"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["openfold", "rfaa", "protenix", "all"],
        default=["all"],
        help="Models to convert for (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples (for testing)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize converters
    models = args.models if "all" not in args.models else ["openfold", "rfaa", "protenix"]

    converters = {}
    if "openfold" in models:
        converters["openfold"] = OpenFoldConverter(args.output_dir)
    if "rfaa" in models:
        converters["rfaa"] = RFAAConverter(args.output_dir)
    if "protenix" in models:
        converters["protenix"] = ProtenixConverter(args.output_dir)

    all_results = {}

    # Process DDI pairs
    logger.info("=" * 50)
    logger.info("Processing DDI pairs")
    logger.info("=" * 50)

    ddi_domain_dir = DDI_DATA_DIR / "domains"
    if not ddi_domain_dir.exists():
        ddi_domain_dir = Path("/projects/u6bz/public/erik/AFDDI_data/AFDB_DDI/dompdbs")

    for split in ["train", "val", "test"]:
        pairs_file = SPLITS_DIR / f"ddi_{split}_pairs.csv"
        if pairs_file.exists():
            results = process_ddi_pairs(pairs_file, ddi_domain_dir, converters, split)
            all_results[f"ddi_{split}"] = results

    # Process PDB structures
    logger.info("=" * 50)
    logger.info("Processing PDB structures")
    logger.info("=" * 50)

    for split in ["train", "val", "test"]:
        split_file = SPLITS_DIR / f"pdb_{split}.txt"
        if split_file.exists():
            with open(split_file, "r") as f:
                pdb_ids = [line.strip() for line in f if line.strip()]

            if args.limit:
                pdb_ids = pdb_ids[:args.limit]

            # Try both monomer and multimer directories
            for pdb_dir in [PDB_MONOMER_DIR, PDB_MULTIMER_DIR]:
                if pdb_dir.exists():
                    results = process_pdb_structures(pdb_ids, pdb_dir, converters, split)
                    all_results[f"pdb_{split}_{pdb_dir.name}"] = results

    # Create manifest files for each model
    logger.info("=" * 50)
    logger.info("Creating manifest files")
    logger.info("=" * 50)

    for model_name, converter in converters.items():
        manifest = {
            "model": model_name,
            "splits": {},
        }

        model_dir = args.output_dir / model_name

        for split_type in ["ddi_pairs", "structures"]:
            split_dir = model_dir / split_type
            if split_dir.exists():
                files = list(split_dir.glob("*"))
                manifest["splits"][split_type] = {
                    "directory": str(split_dir),
                    "count": len(files),
                }

        with open(model_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Created manifest for {model_name}")

    # Summary
    logger.info("=" * 50)
    logger.info("Conversion Summary")
    logger.info("=" * 50)

    for dataset, results in all_results.items():
        logger.info(f"\n{dataset}:")
        for model, counts in results.items():
            logger.info(f"  {model}: {counts['success']} success, {counts['failed']} failed")

    # Save overall results
    with open(args.output_dir / "conversion_results.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Download PDB monomers and multimers from RCSB PDB.

Uses date cutoff of 2021-09-30 to match AlphaFold training cutoff.
Downloads both monomers (for baseline) and multimers (for evaluation).
"""

import argparse
import gzip
import json
import logging
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# Configuration
DATE_CUTOFF = "2021-09-30"
RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download"

# Output directories
MONOMER_DIR = Path("/projects/u6bz/public/jude/pdb_monomers")
MULTIMER_DIR = Path("/projects/u6bz/public/jude/pdb_multimers")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/projects/u6bz/jude/ddi_experiment/logs/download_pdb.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def build_search_query(
    is_multimer: bool,
    date_cutoff: str = DATE_CUTOFF,
    min_resolution: float = 3.5,
) -> dict:
    """
    Build RCSB search query for monomers or multimers.

    Args:
        is_multimer: If True, search for multimers; otherwise monomers
        date_cutoff: Maximum release date (YYYY-MM-DD)
        min_resolution: Maximum resolution in Angstroms
    """
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                # Date filter
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_accession_info.initial_release_date",
                        "operator": "less_or_equal",
                        "value": date_cutoff,
                    },
                },
                # Resolution filter
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal",
                        "value": min_resolution,
                    },
                },
                # Polymer entity count (monomer vs multimer)
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
                        "operator": "greater_or_equal" if is_multimer else "equals",
                        "value": 2 if is_multimer else 1,
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": 100000},
        },
    }

    return query


def search_pdb(query: dict) -> list[str]:
    """Execute RCSB search and return list of PDB IDs with pagination."""
    logger.info("Executing RCSB search query...")

    all_pdb_ids = []
    page_size = 10000
    start = 0

    while True:
        # Update pagination
        query["request_options"]["paginate"] = {"start": start, "rows": page_size}

        response = requests.post(
            RCSB_SEARCH_URL,
            json=query,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        results = response.json()
        page_ids = [r["identifier"] for r in results.get("result_set", [])]
        all_pdb_ids.extend(page_ids)

        total_count = results.get("total_count", 0)
        logger.info(f"Fetched {len(all_pdb_ids)}/{total_count} entries...")

        # Check if we have all results
        if len(page_ids) < page_size or len(all_pdb_ids) >= total_count:
            break

        start += page_size

    logger.info(f"Found {len(all_pdb_ids)} PDB entries total")
    return all_pdb_ids


def download_pdb_file(
    pdb_id: str,
    output_dir: Path,
    format: str = "cif",
) -> Optional[Path]:
    """
    Download a single PDB file.

    Args:
        pdb_id: 4-letter PDB code
        output_dir: Directory to save file
        format: 'cif' for mmCIF or 'pdb' for legacy PDB format
    """
    pdb_id = pdb_id.lower()

    if format == "cif":
        url = f"{RCSB_DOWNLOAD_URL}/{pdb_id}.cif.gz"
        output_file = output_dir / f"{pdb_id}.cif"
    else:
        url = f"{RCSB_DOWNLOAD_URL}/{pdb_id}.pdb.gz"
        output_file = output_dir / f"{pdb_id}.pdb"

    # Skip if already downloaded
    if output_file.exists():
        return output_file

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Decompress and save
        compressed_file = output_dir / f"{pdb_id}.{format}.gz"
        with open(compressed_file, "wb") as f:
            f.write(response.content)

        with gzip.open(compressed_file, "rb") as f_in:
            with open(output_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove compressed file
        compressed_file.unlink()

        return output_file

    except Exception as e:
        logger.warning(f"Failed to download {pdb_id}: {e}")
        return None


def download_batch(
    pdb_ids: list[str],
    output_dir: Path,
    max_workers: int = 16,
    format: str = "cif",
) -> tuple[int, int]:
    """
    Download multiple PDB files in parallel.

    Returns:
        Tuple of (successful_downloads, failed_downloads)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(download_pdb_file, pdb_id, output_dir, format): pdb_id
            for pdb_id in pdb_ids
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            pdb_id = futures[future]
            try:
                result = future.result()
                if result:
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Error downloading {pdb_id}: {e}")
                failed += 1

    return successful, failed


def save_pdb_list(pdb_ids: list[str], output_file: Path):
    """Save list of PDB IDs to file."""
    with open(output_file, "w") as f:
        for pdb_id in pdb_ids:
            f.write(f"{pdb_id}\n")
    logger.info(f"Saved {len(pdb_ids)} PDB IDs to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download PDB monomers and multimers"
    )
    parser.add_argument(
        "--type",
        choices=["monomer", "multimer", "both"],
        default="both",
        help="Type of structures to download",
    )
    parser.add_argument(
        "--date-cutoff",
        default=DATE_CUTOFF,
        help=f"Date cutoff for structures (default: {DATE_CUTOFF})",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=3.5,
        help="Maximum resolution in Angstroms (default: 3.5)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel download workers (default: 16)",
    )
    parser.add_argument(
        "--format",
        choices=["cif", "pdb"],
        default="cif",
        help="Output format (default: cif)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only search and save PDB IDs, don't download",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of structures to download (for testing)",
    )
    args = parser.parse_args()

    # Create output directories
    MONOMER_DIR.mkdir(parents=True, exist_ok=True)
    MULTIMER_DIR.mkdir(parents=True, exist_ok=True)

    lists_dir = Path("/projects/u6bz/jude/ddi_experiment/splits")
    lists_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Download monomers
    if args.type in ["monomer", "both"]:
        logger.info("=" * 50)
        logger.info("Searching for monomers...")
        logger.info("=" * 50)

        query = build_search_query(
            is_multimer=False,
            date_cutoff=args.date_cutoff,
            min_resolution=args.resolution,
        )
        monomer_ids = search_pdb(query)

        if args.limit:
            monomer_ids = monomer_ids[:args.limit]

        save_pdb_list(monomer_ids, lists_dir / "pdb_monomers.txt")

        if not args.dry_run:
            logger.info(f"Downloading {len(monomer_ids)} monomers...")
            success, fail = download_batch(
                monomer_ids,
                MONOMER_DIR,
                max_workers=args.workers,
                format=args.format,
            )
            results["monomers"] = {"success": success, "failed": fail}
            logger.info(f"Monomers: {success} successful, {fail} failed")

    # Download multimers
    if args.type in ["multimer", "both"]:
        logger.info("=" * 50)
        logger.info("Searching for multimers...")
        logger.info("=" * 50)

        query = build_search_query(
            is_multimer=True,
            date_cutoff=args.date_cutoff,
            min_resolution=args.resolution,
        )
        multimer_ids = search_pdb(query)

        if args.limit:
            multimer_ids = multimer_ids[:args.limit]

        save_pdb_list(multimer_ids, lists_dir / "pdb_multimers.txt")

        if not args.dry_run:
            logger.info(f"Downloading {len(multimer_ids)} multimers...")
            success, fail = download_batch(
                multimer_ids,
                MULTIMER_DIR,
                max_workers=args.workers,
                format=args.format,
            )
            results["multimers"] = {"success": success, "failed": fail}
            logger.info(f"Multimers: {success} successful, {fail} failed")

    # Summary
    logger.info("=" * 50)
    logger.info("Download Summary")
    logger.info("=" * 50)
    for dtype, counts in results.items():
        logger.info(f"{dtype}: {counts['success']} successful, {counts['failed']} failed")

    # Save metadata
    metadata = {
        "date_cutoff": args.date_cutoff,
        "resolution_cutoff": args.resolution,
        "download_date": datetime.now().isoformat(),
        "results": results,
    }

    with open(lists_dir / "download_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()

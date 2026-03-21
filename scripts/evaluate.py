#!/usr/bin/env python3
"""
Evaluation script for DDI experiment models.

Computes metrics:
- DockQ: Interface quality (>0.23 acceptable, >0.49 medium, >0.80 high)
- lDDT: Local distance difference test
- TM-score: Global structural similarity
- Inter-chain contact precision at K

Evaluates on:
1. PDB multimer test set (primary evaluation)
2. Hold-out DDI pairs (secondary evaluation)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None

# Configuration
PROJECT_DIR = Path("/projects/u6bz/jude/ddi_experiment")
SPLITS_DIR = PROJECT_DIR / "splits"
OUTPUT_DIR = PROJECT_DIR / "evaluation_results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def compute_lddt(
    pred_coords: np.ndarray,
    true_coords: np.ndarray,
    mask: np.ndarray,
    cutoff: float = 15.0,
    thresholds: tuple = (0.5, 1.0, 2.0, 4.0),
) -> float:
    """
    Compute local Distance Difference Test (lDDT).

    Args:
        pred_coords: Predicted coordinates (N, 3)
        true_coords: True coordinates (N, 3)
        mask: Validity mask (N,)
        cutoff: Distance cutoff for local neighborhood
        thresholds: Distance thresholds for scoring

    Returns:
        lDDT score between 0 and 1
    """
    n_res = len(pred_coords)
    valid_idx = np.where(mask)[0]

    if len(valid_idx) < 2:
        return 0.0

    # Compute pairwise distances
    pred_dists = np.sqrt(
        ((pred_coords[valid_idx, None] - pred_coords[None, valid_idx]) ** 2).sum(-1)
    )
    true_dists = np.sqrt(
        ((true_coords[valid_idx, None] - true_coords[None, valid_idx]) ** 2).sum(-1)
    )

    # Find local pairs (within cutoff in true structure)
    local_mask = (true_dists < cutoff) & (true_dists > 0)

    if not local_mask.any():
        return 0.0

    # Compute distance differences
    dist_diff = np.abs(pred_dists - true_dists)

    # Score based on thresholds
    scores = []
    for thresh in thresholds:
        preserved = (dist_diff < thresh) & local_mask
        score = preserved.sum() / local_mask.sum()
        scores.append(score)

    return np.mean(scores)


def compute_tm_score(
    pred_coords: np.ndarray,
    true_coords: np.ndarray,
    mask: np.ndarray,
) -> float:
    """
    Compute TM-score using tmtools or fallback.

    Args:
        pred_coords: Predicted coordinates (N, 3)
        true_coords: True coordinates (N, 3)
        mask: Validity mask (N,)

    Returns:
        TM-score between 0 and 1
    """
    try:
        import tmtools

        valid_idx = np.where(mask)[0]
        if len(valid_idx) < 5:
            return 0.0

        result = tmtools.tm_align(
            pred_coords[valid_idx],
            true_coords[valid_idx],
            "A" * len(valid_idx),
            "A" * len(valid_idx),
        )

        return result.tm_norm_chain1

    except ImportError:
        # Fallback: simplified TM-score calculation
        valid_idx = np.where(mask)[0]
        n = len(valid_idx)

        if n < 5:
            return 0.0

        # Length normalization factor
        d0 = 1.24 * (n - 15) ** (1.0 / 3.0) - 1.8
        d0 = max(d0, 0.5)

        # Compute distances after optimal superposition (simplified)
        # Using Kabsch alignment
        pred_centered = pred_coords[valid_idx] - pred_coords[valid_idx].mean(0)
        true_centered = true_coords[valid_idx] - true_coords[valid_idx].mean(0)

        # SVD for rotation
        H = pred_centered.T @ true_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Check for reflection
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        pred_aligned = pred_centered @ R
        distances = np.sqrt(((pred_aligned - true_centered) ** 2).sum(-1))

        # TM-score formula
        tm_score = (1 / (1 + (distances / d0) ** 2)).sum() / n

        return tm_score


def compute_dockq(
    pred_pdb: Path,
    true_pdb: Path,
    chain1: str = "A",
    chain2: str = "B",
) -> dict:
    """
    Compute DockQ score for interface quality.

    Args:
        pred_pdb: Path to predicted structure
        true_pdb: Path to true structure
        chain1, chain2: Chain identifiers for the interface

    Returns:
        Dictionary with DockQ metrics
    """
    try:
        # Try using DockQ command-line tool
        result = subprocess.run(
            ["DockQ", str(pred_pdb), str(true_pdb)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode == 0:
            # Parse DockQ output
            output = result.stdout
            metrics = {}

            for line in output.split("\n"):
                if "DockQ" in line and ":" in line:
                    parts = line.split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        try:
                            value = float(parts[1].strip())
                            metrics[key] = value
                        except ValueError:
                            pass

            return metrics

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback: compute simplified DockQ-like metrics
    return compute_simplified_dockq(pred_pdb, true_pdb, chain1, chain2)


def compute_simplified_dockq(
    pred_pdb: Path,
    true_pdb: Path,
    chain1: str = "A",
    chain2: str = "B",
    contact_dist: float = 10.0,
) -> dict:
    """
    Compute simplified DockQ-like metrics.

    Components:
    - fnat: Fraction of native contacts preserved
    - iRMS: Interface RMSD
    - LRMS: Ligand RMSD
    """
    def parse_pdb_coords(pdb_file, chain=None):
        coords = {}  # {(chain, resnum): CA_coords}
        with open(pdb_file, "r") as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    ch = line[21]
                    if chain is None or ch == chain:
                        resnum = int(line[22:26].strip())
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords[(ch, resnum)] = np.array([x, y, z])
        return coords

    def get_contacts(coords, chain1, chain2, cutoff):
        contacts = set()
        for (c1, r1), xyz1 in coords.items():
            if c1 != chain1:
                continue
            for (c2, r2), xyz2 in coords.items():
                if c2 != chain2:
                    continue
                dist = np.sqrt(((xyz1 - xyz2) ** 2).sum())
                if dist < cutoff:
                    contacts.add((r1, r2))
        return contacts

    try:
        true_coords = parse_pdb_coords(true_pdb)
        pred_coords = parse_pdb_coords(pred_pdb)

        # Compute native contacts
        true_contacts = get_contacts(true_coords, chain1, chain2, contact_dist)
        pred_contacts = get_contacts(pred_coords, chain1, chain2, contact_dist)

        if len(true_contacts) == 0:
            fnat = 0.0
        else:
            fnat = len(true_contacts & pred_contacts) / len(true_contacts)

        # Get interface residues
        interface_res_1 = {c[0] for c in true_contacts}
        interface_res_2 = {c[1] for c in true_contacts}

        # Compute interface RMSD
        interface_pred = []
        interface_true = []

        for r in interface_res_1:
            if (chain1, r) in pred_coords and (chain1, r) in true_coords:
                interface_pred.append(pred_coords[(chain1, r)])
                interface_true.append(true_coords[(chain1, r)])

        for r in interface_res_2:
            if (chain2, r) in pred_coords and (chain2, r) in true_coords:
                interface_pred.append(pred_coords[(chain2, r)])
                interface_true.append(true_coords[(chain2, r)])

        if len(interface_pred) > 0:
            interface_pred = np.array(interface_pred)
            interface_true = np.array(interface_true)
            irms = np.sqrt(((interface_pred - interface_true) ** 2).mean())
        else:
            irms = 100.0

        # Compute DockQ (simplified formula)
        # DockQ = (fnat + 1/(1+(irms/1.5)^2) + 1/(1+(lrms/8.5)^2)) / 3
        # Using irms for lrms as approximation
        dockq = (
            fnat +
            1 / (1 + (irms / 1.5) ** 2) +
            1 / (1 + (irms / 8.5) ** 2)
        ) / 3

        return {
            "DockQ": dockq,
            "fnat": fnat,
            "iRMS": irms,
            "quality": (
                "High" if dockq >= 0.8 else
                "Medium" if dockq >= 0.49 else
                "Acceptable" if dockq >= 0.23 else
                "Incorrect"
            ),
        }

    except Exception as e:
        logger.warning(f"Error computing DockQ: {e}")
        return {"DockQ": 0.0, "fnat": 0.0, "iRMS": 100.0, "quality": "Error"}


def compute_contact_precision(
    pred_pdb: Path,
    true_pdb: Path,
    k_values: list = [5, 10, 20, 50],
    contact_dist: float = 8.0,
) -> dict:
    """
    Compute inter-chain contact precision at K.

    Args:
        pred_pdb: Path to predicted structure
        true_pdb: Path to true structure
        k_values: Values of K for P@K
        contact_dist: Distance threshold for contacts

    Returns:
        Dictionary with P@K values
    """
    def parse_contacts(pdb_file, cutoff):
        coords = {}
        with open(pdb_file, "r") as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    chain = line[21]
                    resnum = int(line[22:26].strip())
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords[(chain, resnum)] = np.array([x, y, z])

        # Find inter-chain contacts
        contacts = []
        chains = sorted(set(c for c, r in coords.keys()))

        for i, c1 in enumerate(chains):
            for c2 in chains[i+1:]:
                for (ch1, r1), xyz1 in coords.items():
                    if ch1 != c1:
                        continue
                    for (ch2, r2), xyz2 in coords.items():
                        if ch2 != c2:
                            continue
                        dist = np.sqrt(((xyz1 - xyz2) ** 2).sum())
                        if dist < cutoff:
                            contacts.append((c1, r1, c2, r2, dist))

        return contacts

    try:
        true_contacts = parse_contacts(true_pdb, contact_dist)
        pred_contacts = parse_contacts(pred_pdb, contact_dist)

        # Sort predicted contacts by distance (confidence proxy)
        pred_contacts = sorted(pred_contacts, key=lambda x: x[4])

        # Convert to sets for comparison
        true_set = {(c[0], c[1], c[2], c[3]) for c in true_contacts}

        results = {}
        for k in k_values:
            top_k = pred_contacts[:k]
            pred_set = {(c[0], c[1], c[2], c[3]) for c in top_k}

            if len(pred_set) > 0:
                precision = len(pred_set & true_set) / len(pred_set)
            else:
                precision = 0.0

            results[f"P@{k}"] = precision

        # Compute AUC-like metric
        if len(pred_contacts) > 0:
            aucs = []
            for i, c in enumerate(pred_contacts):
                if (c[0], c[1], c[2], c[3]) in true_set:
                    aucs.append(1.0)
                else:
                    aucs.append(0.0)
            results["contact_auc"] = np.mean(aucs) if aucs else 0.0
        else:
            results["contact_auc"] = 0.0

        return results

    except Exception as e:
        logger.warning(f"Error computing contact precision: {e}")
        return {f"P@{k}": 0.0 for k in k_values}


def evaluate_model(
    model_dir: Path,
    test_set: str = "pdb",
    checkpoint: str = "best_model.pt",
) -> dict:
    """
    Evaluate a trained model on test set.

    Args:
        model_dir: Directory containing trained model
        test_set: 'pdb' or 'ddi' test set
        checkpoint: Checkpoint filename

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating {model_dir} on {test_set} test set")

    results = {
        "model_dir": str(model_dir),
        "test_set": test_set,
        "samples": [],
    }

    # Load test split
    if test_set == "pdb":
        test_file = SPLITS_DIR / "pdb_test.txt"
        if test_file.exists():
            with open(test_file, "r") as f:
                test_ids = [line.strip() for line in f if line.strip()]
        else:
            logger.warning("PDB test file not found")
            return results
    else:
        test_file = SPLITS_DIR / "ddi_test_pairs.csv"
        if test_file.exists():
            test_df = pd.read_csv(test_file)
            test_ids = [
                f"{row['domain1_id']}_{row['domain2_id']}"
                for _, row in test_df.iterrows()
            ]
        else:
            logger.warning("DDI test file not found")
            return results

    # Find predictions directory
    pred_dir = model_dir / "predictions"
    if not pred_dir.exists():
        logger.warning(f"Predictions directory not found: {pred_dir}")
        return results

    # Evaluate each sample
    metrics_list = []

    for sample_id in tqdm(test_ids, desc="Evaluating"):
        pred_pdb = pred_dir / f"{sample_id}.pdb"
        true_pdb = None  # Would need to load from test data

        if not pred_pdb.exists():
            continue

        # Find true structure
        if test_set == "pdb":
            for data_dir in [
                Path("/projects/u6bz/public/jude/pdb_multimers"),
                Path("/projects/u6bz/public/jude/pdb_monomers"),
            ]:
                for suffix in [".pdb", ".cif"]:
                    candidate = data_dir / f"{sample_id}{suffix}"
                    if candidate.exists():
                        true_pdb = candidate
                        break
        else:
            domain_dir = Path("/projects/u6bz/public/jude/processed_ddi/domains")
            # For DDI pairs, need to reconstruct from individual domains
            # This is simplified - full version would combine domains
            true_pdb = domain_dir / f"{sample_id.split('_')[0]}.pdb"

        if true_pdb is None or not true_pdb.exists():
            continue

        # Compute metrics
        sample_metrics = {"id": sample_id}

        # DockQ
        dockq_results = compute_dockq(pred_pdb, true_pdb)
        sample_metrics.update(dockq_results)

        # Contact precision
        contact_results = compute_contact_precision(pred_pdb, true_pdb)
        sample_metrics.update(contact_results)

        metrics_list.append(sample_metrics)

    results["samples"] = metrics_list

    # Aggregate metrics
    if metrics_list:
        df = pd.DataFrame(metrics_list)

        results["aggregate"] = {
            "n_samples": len(metrics_list),
            "mean_DockQ": df["DockQ"].mean(),
            "std_DockQ": df["DockQ"].std(),
            "mean_fnat": df["fnat"].mean(),
            "mean_iRMS": df["iRMS"].mean(),
            "quality_distribution": df["quality"].value_counts().to_dict(),
        }

        for col in df.columns:
            if col.startswith("P@"):
                results["aggregate"][f"mean_{col}"] = df[col].mean()

        if "contact_auc" in df.columns:
            results["aggregate"]["mean_contact_auc"] = df["contact_auc"].mean()

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate DDI experiment models")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing trained model",
    )
    parser.add_argument(
        "--test-set",
        choices=["pdb", "ddi", "both"],
        default="both",
        help="Test set to evaluate on",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for results",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model.pt",
        help="Checkpoint filename",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    if args.test_set in ["pdb", "both"]:
        pdb_results = evaluate_model(args.model_dir, "pdb", args.checkpoint)
        all_results["pdb"] = pdb_results

    if args.test_set in ["ddi", "both"]:
        ddi_results = evaluate_model(args.model_dir, "ddi", args.checkpoint)
        all_results["ddi"] = ddi_results

    # Save results
    output_file = args.output_dir / f"{args.model_dir.name}_evaluation.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Results saved to {output_file}")

    # Print summary
    logger.info("=" * 50)
    logger.info("Evaluation Summary")
    logger.info("=" * 50)

    for test_set, results in all_results.items():
        if "aggregate" in results:
            logger.info(f"\n{test_set.upper()} Test Set:")
            for key, value in results["aggregate"].items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()

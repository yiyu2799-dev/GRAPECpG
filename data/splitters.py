"""Dataset split helpers for GRAPE-CpG.

This module deliberately contains no torch/torch_geometric dependency so that
split semantics can be tested independently from model code.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


SPLIT_NAMES = ("train", "val", "test")


def validate_split_fractions(fractions: Iterable[float]) -> Tuple[float, float, float]:
    """Validate and normalize a train/val/test fraction triple.

    All three fractions must be finite and strictly positive, and their sum
    must equal 1 (up to floating-point tolerance).
    """
    if fractions is None:
        raise ValueError("split_fractions must be provided for within_chromosome mode.")

    values = tuple(float(x) for x in fractions)
    if len(values) != 3:
        raise ValueError(
            "split_fractions must contain exactly three values: train val test."
        )
    if not all(np.isfinite(x) for x in values):
        raise ValueError("split_fractions must be finite numbers.")
    if not all(x > 0.0 for x in values):
        raise ValueError("All split_fractions must be strictly positive.")

    total = float(sum(values))
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(
            f"split_fractions must sum to 1.0, got {values} (sum={total:.12g})."
        )
    return values


def split_indices_by_position(
    indices: np.ndarray,
    positions: np.ndarray,
    fractions: Iterable[float],
) -> Dict[str, np.ndarray]:
    """Split genomic-site indices into contiguous position-ordered regions.

    The split is by *number of CpG sites*, not by physical chromosome length.
    Positions are sorted in ascending genomic order before slicing.  Duplicate
    positions are rejected because they would make the contiguous split
    boundary ambiguous and generally indicate an upstream preprocessing issue.
    """
    fractions = validate_split_fractions(fractions)
    indices = np.asarray(indices, dtype=np.int64)
    positions = np.asarray(positions)

    if indices.ndim != 1:
        raise ValueError(f"indices must be 1D, got shape={indices.shape}.")
    if positions.ndim != 1:
        raise ValueError(f"positions must be 1D, got shape={positions.shape}.")
    if len(indices) == 0:
        raise ValueError("Cannot split an empty chromosome.")
    if np.any(indices < 0) or np.any(indices >= len(positions)):
        raise ValueError("indices contain values outside the positions array.")

    pos = np.asarray(positions[indices], dtype=np.float64)
    if not np.all(np.isfinite(pos)):
        raise ValueError("Selected chromosome positions contain NaN/inf values.")

    order = np.argsort(pos, kind="mergesort")
    sorted_indices = indices[order]
    sorted_pos = pos[order]
    if len(sorted_pos) > 1 and np.any(np.diff(sorted_pos) <= 0):
        raise ValueError(
            "within_chromosome split requires unique genomic positions; "
            "duplicate/non-increasing positions were detected after sorting."
        )

    n = len(sorted_indices)
    train_end = int(np.floor(n * fractions[0]))
    val_end = int(np.floor(n * (fractions[0] + fractions[1])))

    boundaries = (0, train_end, val_end, n)
    counts = tuple(boundaries[i + 1] - boundaries[i] for i in range(3))
    if any(count <= 0 for count in counts):
        raise ValueError(
            "The chromosome is too small for the requested split_fractions: "
            f"n_sites={n}, fractions={fractions}, counts={counts}."
        )

    split_map = {
        "train": sorted_indices[:train_end].copy(),
        "val": sorted_indices[train_end:val_end].copy(),
        "test": sorted_indices[val_end:].copy(),
    }

    # Defensive invariants: disjoint, exhaustive, and contiguous in genomic order.
    concatenated = np.concatenate([split_map[name] for name in SPLIT_NAMES])
    if not np.array_equal(concatenated, sorted_indices):
        raise RuntimeError("Internal split invariant failed: regions are not exhaustive/ordered.")
    if len(np.unique(concatenated)) != n:
        raise RuntimeError("Internal split invariant failed: split regions overlap.")

    return split_map

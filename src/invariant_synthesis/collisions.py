"""Consequential-collision mining in the engine's current representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class RobustScaler:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, matrix: np.ndarray) -> RobustScaler:
        matrix = _as_matrix(matrix)
        center = np.median(matrix, axis=0)
        q75 = np.percentile(matrix, 75, axis=0)
        q25 = np.percentile(matrix, 25, axis=0)
        scale = q75 - q25
        fallback = np.std(matrix, axis=0)
        scale = np.where(scale > 1e-12, scale, fallback)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return cls(center=center, scale=scale)

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return (_as_matrix(matrix) - self.center) / self.scale


@dataclass(frozen=True, slots=True)
class CollisionSet:
    left: np.ndarray
    right: np.ndarray
    distances: np.ndarray
    pressure: float
    unresolved_fraction: float

    def __len__(self) -> int:
        return len(self.left)

    @property
    def pairs(self) -> np.ndarray:
        return np.column_stack([self.left, self.right])

    def to_dict(self) -> dict[str, object]:
        return {
            "pair_count": len(self),
            "pressure": self.pressure,
            "unresolved_fraction": self.unresolved_fraction,
            "median_distance": float(np.median(self.distances)) if len(self) else None,
            "pairs": self.pairs.tolist(),
            "distances": self.distances.tolist(),
        }


def _as_matrix(matrix: np.ndarray) -> np.ndarray:
    result = np.asarray(matrix, dtype=float)
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2:
        raise ValueError("representation must be a 2D matrix")
    return np.clip(np.nan_to_num(result), -1e12, 1e12)


def _outcome_eligibility(
    source: np.ndarray,
    targets: np.ndarray,
    outcome_kind: str,
    continuous_delta: float,
    continuous_spread: float | None,
) -> np.ndarray:
    if outcome_kind == "categorical":
        return source[:, None] != targets[None, :]
    assert continuous_spread is not None
    return (
        np.abs(source.astype(float)[:, None] - targets.astype(float)[None, :])
        >= continuous_delta * continuous_spread
    )


def find_collisions(
    representation: np.ndarray,
    outcomes: np.ndarray,
    *,
    feature_weights: np.ndarray | None = None,
    outcome_kind: str = "categorical",
    continuous_delta: float = 0.75,
    neighbors_per_sample: int = 2,
    max_pairs: int = 512,
    unresolved_distance: float = 0.75,
    block_size: int = 256,
) -> CollisionSet:
    """Find close trajectories whose outcomes differ materially.

    Distances are computed after robust feature scaling. Work is blocked so the
    implementation does not allocate a full distance matrix for large datasets.
    """

    matrix = _as_matrix(representation)
    y = np.asarray(outcomes).reshape(-1)
    if len(y) != len(matrix):
        raise ValueError("representation and outcomes have different sample counts")
    if len(matrix) < 2:
        return CollisionSet(
            left=np.asarray([], dtype=int),
            right=np.asarray([], dtype=int),
            distances=np.asarray([], dtype=float),
            pressure=0.0,
            unresolved_fraction=0.0,
        )
    scaled = RobustScaler.fit(matrix).transform(matrix)
    if feature_weights is None:
        weights = np.ones(scaled.shape[1], dtype=float)
    else:
        weights = np.asarray(feature_weights, dtype=float).reshape(-1)
        if len(weights) != scaled.shape[1] or np.any(weights <= 0):
            raise ValueError("feature_weights must be positive and match representation columns")
    weights = weights / np.sum(weights)
    continuous_spread = None
    if outcome_kind == "continuous":
        numeric = y.astype(float)
        q75, q25 = np.percentile(numeric, [75, 25])
        continuous_spread = max(float(q75 - q25), float(np.std(numeric)), 1e-12)
    candidates: dict[tuple[int, int], float] = {}
    k = max(1, neighbors_per_sample)
    for start in range(0, len(scaled), block_size):
        stop = min(len(scaled), start + block_size)
        eligible = _outcome_eligibility(
            y[start:stop],
            y,
            outcome_kind,
            continuous_delta,
            continuous_spread,
        )
        eligible[np.arange(stop - start), np.arange(start, stop)] = False
        diff = scaled[start:stop, None, :] - scaled[None, :, :]
        distances = np.sqrt(np.sum(diff * diff * weights, axis=2))
        distances[~eligible] = np.inf
        for local, row in enumerate(distances):
            finite = np.flatnonzero(np.isfinite(row))
            if finite.size == 0:
                continue
            count = min(k, finite.size)
            selected_local = np.argpartition(row[finite], count - 1)[:count]
            source = start + local
            for target in finite[selected_local]:
                pair = (source, int(target)) if source < target else (int(target), source)
                distance = float(row[target])
                if pair not in candidates or distance < candidates[pair]:
                    candidates[pair] = distance
    ordered = sorted(candidates.items(), key=lambda item: item[1])[:max_pairs]
    if not ordered:
        return CollisionSet(
            left=np.asarray([], dtype=int),
            right=np.asarray([], dtype=int),
            distances=np.asarray([], dtype=float),
            pressure=0.0,
            unresolved_fraction=0.0,
        )
    left = np.asarray([pair[0][0] for pair in ordered], dtype=int)
    right = np.asarray([pair[0][1] for pair in ordered], dtype=int)
    distances = np.asarray([pair[1] for pair in ordered], dtype=float)
    pressure = float(np.mean(np.exp(-distances)))
    unresolved_fraction = float(np.mean(distances <= unresolved_distance))
    return CollisionSet(
        left=left,
        right=right,
        distances=distances,
        pressure=pressure,
        unresolved_fraction=unresolved_fraction,
    )

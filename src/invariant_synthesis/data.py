"""Validated trajectory container and canonicalization utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _as_strings(values: np.ndarray | list[Any], expected: int, label: str) -> np.ndarray:
    array = np.asarray(values).reshape(-1)
    if len(array) != expected:
        raise ValueError(f"{label} must contain {expected} entries, got {len(array)}")
    return array.astype(str)


@dataclass(slots=True)
class TrajectoryDataset:
    """A collection of fixed-window multivariate trajectories.

    ``values`` has shape ``(samples, time, signals)``. Outcomes may be categorical
    strings or numeric values. NaNs are allowed so real telemetry can be loaded, but
    synthesis requires calling :meth:`imputed` first (the engine does this by default).
    """

    values: np.ndarray
    outcomes: np.ndarray
    signal_names: tuple[str, ...] | list[str]
    units: tuple[str, ...] | list[str] | None = None
    groups: np.ndarray | list[Any] | None = None
    sample_ids: np.ndarray | list[Any] | None = None
    time_step: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim != 3:
            raise ValueError(
                f"values must have shape (samples, time, signals); received {self.values.shape}"
            )
        n_samples, n_time, n_signals = self.values.shape
        if n_samples < 4:
            raise ValueError("at least four trajectories are required")
        if n_time < 3:
            raise ValueError("each trajectory must contain at least three time points")
        if n_signals < 1:
            raise ValueError("at least one signal is required")
        if np.isinf(self.values).any():
            raise ValueError("values may contain NaN but not +/- infinity")

        self.outcomes = np.asarray(self.outcomes).reshape(-1)
        if self.outcomes.dtype.kind == "O":
            self.outcomes = self.outcomes.astype(str)
        if len(self.outcomes) != n_samples:
            raise ValueError(f"outcomes must contain {n_samples} entries, got {len(self.outcomes)}")

        self.signal_names = tuple(str(name) for name in self.signal_names)
        if len(self.signal_names) != n_signals:
            raise ValueError(
                f"signal_names must contain {n_signals} entries, got {len(self.signal_names)}"
            )
        if len(set(self.signal_names)) != len(self.signal_names):
            raise ValueError("signal_names must be unique")

        if self.units is None:
            self.units = tuple("1" for _ in range(n_signals))
        else:
            self.units = tuple(str(unit) for unit in self.units)
            if len(self.units) != n_signals:
                raise ValueError(f"units must contain {n_signals} entries")

        self.groups = None if self.groups is None else _as_strings(self.groups, n_samples, "groups")
        self.sample_ids = (
            np.asarray([f"trajectory_{i:05d}" for i in range(n_samples)])
            if self.sample_ids is None
            else _as_strings(self.sample_ids, n_samples, "sample_ids")
        )
        if not np.isfinite(self.time_step) or self.time_step <= 0:
            raise ValueError("time_step must be a positive finite number")

    @property
    def n_samples(self) -> int:
        return self.values.shape[0]

    @property
    def n_time(self) -> int:
        return self.values.shape[1]

    @property
    def n_signals(self) -> int:
        return self.values.shape[2]

    @property
    def missing_fraction(self) -> float:
        return float(np.isnan(self.values).mean())

    @property
    def outcome_kind(self) -> str:
        if self.outcomes.dtype.kind in "biufc" and len(np.unique(self.outcomes)) > 10:
            return "continuous"
        return "categorical"

    def subset(self, indices: np.ndarray | list[int]) -> TrajectoryDataset:
        idx = np.asarray(indices, dtype=int)
        return TrajectoryDataset(
            values=self.values[idx],
            outcomes=self.outcomes[idx],
            signal_names=self.signal_names,
            units=self.units,
            groups=None if self.groups is None else self.groups[idx],
            sample_ids=self.sample_ids[idx],
            time_step=self.time_step,
            metadata=dict(self.metadata),
        )

    def imputed(self) -> TrajectoryDataset:
        """Linearly interpolate missing points within each trajectory and signal.

        Edge gaps use the closest observed value. A completely absent signal in one
        trajectory is filled with the across-trajectory median for that signal.
        """

        if not np.isnan(self.values).any():
            return self.subset(np.arange(self.n_samples))
        filled = self.values.copy()
        signal_defaults = np.nanmedian(filled, axis=(0, 1))
        signal_defaults = np.where(np.isfinite(signal_defaults), signal_defaults, 0.0)
        for sample in range(self.n_samples):
            for signal in range(self.n_signals):
                series = filled[sample, :, signal]
                observed = np.flatnonzero(np.isfinite(series))
                if observed.size == 0:
                    series[:] = signal_defaults[signal]
                elif observed.size < self.n_time:
                    missing = np.flatnonzero(~np.isfinite(series))
                    series[missing] = np.interp(missing, observed, series[observed])
                filled[sample, :, signal] = series
        result = self.subset(np.arange(self.n_samples))
        result.values = filled
        result.metadata["imputed_missing_fraction"] = self.missing_fraction
        return result

    def canonicalized(
        self,
        *,
        center: str = "initial",
        scale: str = "none",
    ) -> TrajectoryDataset:
        """Remove nuisance operating points and optionally trajectory-local scale.

        ``center`` is one of ``initial``, ``mean``, ``median`` or ``none``.
        ``scale`` is one of ``robust``, ``std`` or ``none``. Scaling is per trajectory
        and signal, so use it only when absolute amplitude is known to be irrelevant.
        """

        result = self.imputed()
        values = result.values.copy()
        center = center.lower()
        if center == "initial":
            values -= values[:, :1, :]
        elif center == "mean":
            values -= np.mean(values, axis=1, keepdims=True)
        elif center == "median":
            values -= np.median(values, axis=1, keepdims=True)
        elif center != "none":
            raise ValueError("center must be initial, mean, median, or none")

        scale = scale.lower()
        if scale == "robust":
            q75 = np.percentile(values, 75, axis=1, keepdims=True)
            q25 = np.percentile(values, 25, axis=1, keepdims=True)
            denom = q75 - q25
            denom = np.where(denom > 1e-12, denom, 1.0)
            values /= denom
        elif scale == "std":
            denom = np.std(values, axis=1, keepdims=True)
            denom = np.where(denom > 1e-12, denom, 1.0)
            values /= denom
        elif scale != "none":
            raise ValueError("scale must be robust, std, or none")

        result.values = values
        result.metadata["canonicalization"] = {"center": center, "scale": scale}
        return result

    def split_indices(
        self,
        validation_fraction: float = 0.25,
        seed: int = 1729,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create a deterministic holdout, respecting groups when supplied."""

        if not 0.1 <= validation_fraction <= 0.5:
            raise ValueError("validation_fraction must be between 0.1 and 0.5")
        rng = np.random.default_rng(seed)
        if self.groups is not None:
            unique = np.unique(self.groups)
            if len(unique) < 2:
                raise ValueError("group-aware validation requires at least two groups")
            target = max(1, int(round(len(unique) * validation_fraction)))
            required_labels = (
                set(np.unique(self.outcomes)) if self.outcome_kind == "categorical" else set()
            )
            valid = None
            for _ in range(100):
                shuffled = rng.permutation(unique)
                validation_groups = set(shuffled[:target])
                proposal = np.asarray([g in validation_groups for g in self.groups])
                if not required_labels or (
                    set(np.unique(self.outcomes[proposal])) == required_labels
                    and set(np.unique(self.outcomes[~proposal])) == required_labels
                ):
                    valid = proposal
                    break
            if valid is None:
                raise ValueError(
                    "could not create a group holdout containing every outcome class; "
                    "provide more groups or revise group labels"
                )
            train_idx = np.flatnonzero(~valid)
            valid_idx = np.flatnonzero(valid)
        elif self.outcome_kind == "categorical":
            train_parts: list[np.ndarray] = []
            valid_parts: list[np.ndarray] = []
            for label in np.unique(self.outcomes):
                indices = rng.permutation(np.flatnonzero(self.outcomes == label))
                count = max(1, int(round(len(indices) * validation_fraction)))
                valid_parts.append(indices[:count])
                train_parts.append(indices[count:])
            train_idx = np.sort(np.concatenate(train_parts))
            valid_idx = np.sort(np.concatenate(valid_parts))
        else:
            indices = rng.permutation(self.n_samples)
            count = max(1, int(round(self.n_samples * validation_fraction)))
            valid_idx = np.sort(indices[:count])
            train_idx = np.sort(indices[count:])
        if len(train_idx) < 2 or len(valid_idx) < 2:
            raise ValueError("the requested split leaves too few trajectories")
        return train_idx, valid_idx

    def to_npz(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            values=self.values,
            outcomes=self.outcomes,
            signal_names=np.asarray(self.signal_names),
            units=np.asarray(self.units),
            groups=np.asarray([]) if self.groups is None else self.groups,
            sample_ids=self.sample_ids,
            time_step=np.asarray(self.time_step),
        )
        return target

    @classmethod
    def from_npz(cls, path: str | Path) -> TrajectoryDataset:
        with np.load(Path(path), allow_pickle=False) as loaded:
            required = {"values", "outcomes", "signal_names"}
            missing = required.difference(loaded.files)
            if missing:
                raise ValueError(f"dataset is missing required arrays: {sorted(missing)}")
            groups = loaded["groups"] if "groups" in loaded.files else None
            if groups is not None and groups.size == 0:
                groups = None
            return cls(
                values=loaded["values"],
                outcomes=loaded["outcomes"],
                signal_names=tuple(loaded["signal_names"].astype(str)),
                units=(tuple(loaded["units"].astype(str)) if "units" in loaded.files else None),
                groups=groups,
                sample_ids=loaded["sample_ids"] if "sample_ids" in loaded.files else None,
                time_step=(float(loaded["time_step"]) if "time_step" in loaded.files else 1.0),
            )

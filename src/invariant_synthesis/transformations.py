"""User-declared nuisance transformations for empirical invariance testing."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Transformation:
    """A named transformation under which accepted expressions should remain stable."""

    name: str
    apply_fn: Callable[[np.ndarray], np.ndarray]
    description: str = ""

    def apply(self, values: np.ndarray) -> np.ndarray:
        transformed = np.asarray(self.apply_fn(values.copy()), dtype=float)
        if transformed.shape != values.shape:
            raise ValueError(
                f"transformation {self.name!r} changed shape from {values.shape} "
                f"to {transformed.shape}"
            )
        if not np.isfinite(transformed).all():
            raise ValueError(f"transformation {self.name!r} produced non-finite values")
        return transformed


def _indices(signals: Iterable[int] | None, n_signals: int) -> np.ndarray:
    if signals is None:
        return np.arange(n_signals)
    result = np.asarray(tuple(signals), dtype=int)
    if result.size == 0 or np.any(result < 0) or np.any(result >= n_signals):
        raise ValueError("transformation signal indices are empty or out of range")
    return result


def global_scale(factor: float, signals: Iterable[int] | None = None) -> Transformation:
    """Scale selected channels by a common factor."""

    if not np.isfinite(factor) or factor == 0:
        raise ValueError("factor must be finite and non-zero")
    selected = None if signals is None else tuple(signals)

    def apply(values: np.ndarray) -> np.ndarray:
        values[:, :, _indices(selected, values.shape[2])] *= factor
        return values

    return Transformation(
        name=(
            f"global_scale_{factor:g}"
            if selected is None
            else f"channel_scale_{'_'.join(map(str, selected))}_{factor:g}"
        ),
        apply_fn=apply,
        description=f"Common amplitude scaling by {factor:g}",
    )


def sensor_offset(offset: float, signals: Iterable[int] | None = None) -> Transformation:
    """Add a constant calibration/operating-point offset to selected channels."""

    if not np.isfinite(offset):
        raise ValueError("offset must be finite")
    selected = None if signals is None else tuple(signals)

    def apply(values: np.ndarray) -> np.ndarray:
        values[:, :, _indices(selected, values.shape[2])] += offset
        return values

    return Transformation(
        name=f"sensor_offset_{offset:g}",
        apply_fn=apply,
        description=f"Constant sensor offset of {offset:g}",
    )


def time_shift(steps: int) -> Transformation:
    """Circularly shift each observation window to test time-origin invariance."""

    if steps == 0:
        raise ValueError("steps must be non-zero")

    def apply(values: np.ndarray) -> np.ndarray:
        return np.roll(values, shift=steps, axis=1)

    return Transformation(
        name=f"time_shift_{steps:+d}",
        apply_fn=apply,
        description=f"Circular time-origin shift by {steps} samples",
    )


def sign_flip(signals: Iterable[int] | None = None) -> Transformation:
    """Flip selected perturbation coordinates around their zero-centered reference."""

    def apply(values: np.ndarray) -> np.ndarray:
        values[:, :, _indices(signals, values.shape[2])] *= -1.0
        return values

    return Transformation(
        name="sign_flip",
        apply_fn=apply,
        description="Sign symmetry around a zero-centered operating point",
    )


def channel_permutation(permutation: Iterable[int]) -> Transformation:
    """Relabel equivalent channels while preserving tensor shape."""

    order = np.asarray(tuple(permutation), dtype=int)

    def apply(values: np.ndarray) -> np.ndarray:
        if len(order) != values.shape[2] or set(order.tolist()) != set(range(values.shape[2])):
            raise ValueError("permutation must contain every channel index exactly once")
        return values[:, :, order]

    return Transformation(
        name="channel_permutation_" + "_".join(map(str, order.tolist())),
        apply_fn=apply,
        description="Topology/device relabeling supplied by the user",
    )

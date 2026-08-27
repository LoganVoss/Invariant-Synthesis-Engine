"""Domain-neutral trajectory descriptors that form leaves of the search grammar."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .data import TrajectoryDataset
from .dimensions import Dimension

DEFAULT_STATISTICS = (
    "mean",
    "std",
    "rms",
    "delta",
    "slope",
    "diff_rms",
    "peak_abs",
    "early_rms",
    "late_rms",
    "spectral_entropy",
    "spectral_centroid",
    "lag1_autocorr",
)


@dataclass(frozen=True, slots=True)
class PrimitiveSpec:
    name: str
    statistic: str
    signal_a: int
    signal_b: int | None
    dimension: Dimension

    def evaluate(self, values: np.ndarray, time_step: float) -> np.ndarray:
        x = values[:, :, self.signal_a]
        stat = self.statistic
        if stat == "mean":
            result = np.mean(x, axis=1)
        elif stat == "std":
            result = np.std(x, axis=1)
        elif stat == "rms":
            result = np.sqrt(np.mean(np.square(x), axis=1))
        elif stat == "delta":
            result = x[:, -1] - x[:, 0]
        elif stat == "slope":
            time = np.arange(x.shape[1], dtype=float) * time_step
            centered_time = time - np.mean(time)
            denom = float(np.sum(np.square(centered_time))) or 1.0
            result = np.sum((x - np.mean(x, axis=1, keepdims=True)) * centered_time, axis=1)
            result /= denom
        elif stat == "diff_rms":
            result = np.sqrt(np.mean(np.square(np.diff(x, axis=1) / time_step), axis=1))
        elif stat == "peak_abs":
            result = np.max(np.abs(x), axis=1)
        elif stat in {"early_rms", "late_rms"}:
            width = max(2, x.shape[1] // 3)
            window = x[:, :width] if stat == "early_rms" else x[:, -width:]
            result = np.sqrt(np.mean(np.square(window), axis=1))
        elif stat == "spectral_entropy":
            result = _spectral_entropy(x)
        elif stat == "spectral_centroid":
            result = _spectral_centroid(x, time_step)
        elif stat == "lag1_autocorr":
            result = _lagged_correlation(x[:, 1:], x[:, :-1])
        elif stat == "correlation":
            assert self.signal_b is not None
            y = values[:, :, self.signal_b]
            result = _lagged_correlation(x, y)
        elif stat == "lag1_crosscorr":
            assert self.signal_b is not None
            y = values[:, :, self.signal_b]
            forward = _lagged_correlation(x[:, 1:], y[:, :-1])
            backward = _lagged_correlation(y[:, 1:], x[:, :-1])
            result = forward - backward
        elif stat == "covariance":
            assert self.signal_b is not None
            y = values[:, :, self.signal_b]
            result = np.mean(
                (x - np.mean(x, axis=1, keepdims=True)) * (y - np.mean(y, axis=1, keepdims=True)),
                axis=1,
            )
        elif stat == "difference_rms":
            assert self.signal_b is not None
            y = values[:, :, self.signal_b]
            result = np.sqrt(np.mean(np.square(x - y), axis=1))
        else:
            raise ValueError(f"unknown primitive statistic: {stat}")
        return np.nan_to_num(result, nan=0.0, posinf=1e12, neginf=-1e12)


def _lagged_correlation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0 = x - np.mean(x, axis=1, keepdims=True)
    y0 = y - np.mean(y, axis=1, keepdims=True)
    numerator = np.sum(x0 * y0, axis=1)
    denominator = np.sqrt(np.sum(x0 * x0, axis=1) * np.sum(y0 * y0, axis=1))
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-12,
    )


def _power_spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = x - np.mean(x, axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(centered, axis=1)) ** 2
    if power.shape[1] > 1:
        power[:, 0] = 0.0
    total = np.sum(power, axis=1, keepdims=True)
    probabilities = np.divide(power, total, out=np.zeros_like(power), where=total > 1e-20)
    return power, probabilities


def _spectral_entropy(x: np.ndarray) -> np.ndarray:
    _, probabilities = _power_spectrum(x)
    bins = max(2, probabilities.shape[1] - 1)
    entropy = -np.sum(
        np.where(probabilities > 0, probabilities * np.log(probabilities + 1e-20), 0.0),
        axis=1,
    )
    return entropy / np.log(bins)


def _spectral_centroid(x: np.ndarray, time_step: float) -> np.ndarray:
    _, probabilities = _power_spectrum(x)
    frequencies = np.fft.rfftfreq(x.shape[1], d=time_step)
    return probabilities @ frequencies


class PrimitiveLibrary:
    """Build and evaluate a bounded, explainable set of primitive descriptors."""

    def __init__(
        self,
        dataset: TrajectoryDataset,
        *,
        statistics: tuple[str, ...] = DEFAULT_STATISTICS,
        pairwise: bool = True,
        max_pairwise_signals: int = 16,
    ) -> None:
        self.signal_names = dataset.signal_names
        self.units = dataset.units
        self.time_step = dataset.time_step
        self.specs = self._build_specs(
            statistics,
            pairwise=pairwise,
            max_pairwise_signals=max_pairwise_signals,
        )

    def _build_specs(
        self,
        statistics: tuple[str, ...],
        *,
        pairwise: bool,
        max_pairwise_signals: int,
    ) -> tuple[PrimitiveSpec, ...]:
        allowed = set(DEFAULT_STATISTICS)
        unknown = set(statistics).difference(allowed)
        if unknown:
            raise ValueError(f"unknown primitive statistics: {sorted(unknown)}")
        specs: list[PrimitiveSpec] = []
        time_dimension = Dimension.from_unit("time")
        for index, (name, unit) in enumerate(zip(self.signal_names, self.units, strict=True)):
            base = Dimension.from_unit(unit)
            for statistic in statistics:
                if statistic in {"spectral_entropy", "lag1_autocorr"}:
                    dimension = Dimension()
                elif statistic == "spectral_centroid":
                    dimension = Dimension().divide(time_dimension)
                elif statistic in {"slope", "diff_rms"}:
                    dimension = base.divide(time_dimension)
                else:
                    dimension = base
                specs.append(
                    PrimitiveSpec(
                        name=f"{statistic}[{name}]",
                        statistic=statistic,
                        signal_a=index,
                        signal_b=None,
                        dimension=dimension,
                    )
                )
        if pairwise and len(self.signal_names) <= max_pairwise_signals:
            for left, right in combinations(range(len(self.signal_names)), 2):
                left_name = self.signal_names[left]
                right_name = self.signal_names[right]
                left_dim = Dimension.from_unit(self.units[left])
                right_dim = Dimension.from_unit(self.units[right])
                specs.extend(
                    [
                        PrimitiveSpec(
                            name=f"correlation[{left_name},{right_name}]",
                            statistic="correlation",
                            signal_a=left,
                            signal_b=right,
                            dimension=Dimension(),
                        ),
                        PrimitiveSpec(
                            name=f"lag_asymmetry[{left_name},{right_name}]",
                            statistic="lag1_crosscorr",
                            signal_a=left,
                            signal_b=right,
                            dimension=Dimension(),
                        ),
                        PrimitiveSpec(
                            name=f"covariance[{left_name},{right_name}]",
                            statistic="covariance",
                            signal_a=left,
                            signal_b=right,
                            dimension=left_dim.multiply(right_dim),
                        ),
                    ]
                )
                if left_dim == right_dim:
                    specs.append(
                        PrimitiveSpec(
                            name=f"difference_rms[{left_name},{right_name}]",
                            statistic="difference_rms",
                            signal_a=left,
                            signal_b=right,
                            dimension=left_dim,
                        )
                    )
        return tuple(specs)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.specs)

    def evaluate_values(self, values: np.ndarray) -> np.ndarray:
        columns = [spec.evaluate(values, self.time_step) for spec in self.specs]
        matrix = np.column_stack(columns)
        return np.clip(np.nan_to_num(matrix), -1e12, 1e12)

    def evaluate(self, dataset: TrajectoryDataset) -> np.ndarray:
        if dataset.signal_names != self.signal_names:
            raise ValueError("dataset signals do not match the primitive library")
        return self.evaluate_values(dataset.values)

"""Deterministic synthetic systems used for examples and regression tests."""

from __future__ import annotations

import numpy as np

from .data import TrajectoryDataset


def make_energy_grid_dataset(
    *,
    samples: int = 240,
    time_points: int = 96,
    regions: int = 12,
    seed: int = 2401,
    missing_fraction: float = 0.005,
) -> TrajectoryDataset:
    """Create event windows with matched bulk statistics but opposite damping futures.

    The intentionally hidden coordinate is a late/early oscillatory-energy ratio.
    Each channel is normalized to nearly the same whole-window RMS distribution, making
    ordinary mean/std/delta dashboards collide while the shape of energy through time
    remains consequential.
    """

    if samples < max(24, regions * 2):
        raise ValueError("samples is too small for the requested number of regions")
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, time_points)
    values = np.zeros((samples, time_points, 6), dtype=float)
    outcomes = np.empty(samples, dtype="U16")
    groups = np.asarray([f"region_{index % regions:02d}" for index in range(samples)])

    for sample in range(samples):
        unstable = (sample // regions + sample % regions) % 2 == 0
        outcomes[sample] = "unstable" if unstable else "recovering"
        region = sample % regions
        mode_frequency = 3.0 + 0.06 * region + rng.normal(0.0, 0.08)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        growth = rng.normal(1.85 if unstable else -1.85, 0.22)
        envelope = np.exp(growth * (t - 0.5))
        carrier = np.sin(2.0 * np.pi * mode_frequency * t + phase)
        base = envelope * carrier
        stationary_a = np.sin(2.0 * np.pi * (mode_frequency + 0.35) * t + phase / 2)
        stationary_b = np.cos(2.0 * np.pi * (mode_frequency - 0.28) * t - phase / 3)
        slow_mode = np.sin(2.0 * np.pi * (0.65 + 0.01 * region) * t + 1.7 * phase)

        raw = np.column_stack(
            [
                base + rng.normal(0.0, 0.055, time_points),
                0.82 * stationary_a + rng.normal(0.0, 0.060, time_points),
                1.10 * stationary_b + 0.18 * stationary_a + rng.normal(0.0, 0.070, time_points),
                0.55 * stationary_a - 0.12 * stationary_b + rng.normal(0.0, 0.070, time_points),
                slow_mode + rng.normal(0.0, 0.035, time_points),
                np.gradient(stationary_b, t) + rng.normal(0.0, 0.08, time_points),
            ]
        )
        # Match whole-window amplitudes across outcomes; preserve where energy occurs.
        target_rms = rng.lognormal(mean=-0.05, sigma=0.18, size=6)
        current_rms = np.sqrt(np.mean(raw * raw, axis=0))
        raw *= target_rms / np.maximum(current_rms, 1e-9)
        region_gain = 1.0 + 0.015 * region
        values[sample] = raw * region_gain

    if missing_fraction > 0:
        missing = rng.random(values.shape) < missing_fraction
        # Keep enough endpoints to make event alignment unambiguous.
        missing[:, 0, :] = False
        missing[:, -1, :] = False
        values[missing] = np.nan

    return TrajectoryDataset(
        values=values,
        outcomes=outcomes,
        signal_names=(
            "frequency_deviation",
            "voltage_deviation",
            "active_power_boundary",
            "reactive_power_boundary",
            "interface_angle",
            "power_ramp",
        ),
        units=("Hz", "p.u.", "MW", "MVAr", "rad", "MW/s"),
        groups=groups,
        sample_ids=np.asarray([f"grid_event_{index:04d}" for index in range(samples)]),
        time_step=1.0 / (time_points - 1),
        metadata={
            "generator": "synthetic matched-statistics grid events",
            "hidden_coordinate": "late_rms / early_rms (oscillatory energy growth)",
            "warning": "demonstration data only; not a validated grid model",
        },
    )

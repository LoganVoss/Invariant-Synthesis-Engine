"""Minimal template for adapting ISE to another dynamical system."""

import numpy as np

from invariant_synthesis import EngineConfig, InvariantSynthesisEngine, TrajectoryDataset
from invariant_synthesis.transformations import sensor_offset

# Replace these arrays with aligned event windows and consequential outcomes.
rng = np.random.default_rng(7)
trajectories = rng.normal(size=(100, 64, 3))
outcomes = np.asarray(["nominal", "critical"] * 50)
asset_ids = np.asarray([f"asset_{index % 10}" for index in range(100)])

dataset = TrajectoryDataset(
    values=trajectories,
    outcomes=outcomes,
    signal_names=("flow", "pressure", "temperature"),
    units=("kg/s", "bar", "degC"),
    groups=asset_ids,
    time_step=0.1,
)

engine = InvariantSynthesisEngine(EngineConfig(center="initial"))
result = engine.fit(
    dataset,
    # Include only transformations that domain knowledge says are true nuisances.
    transformations=(sensor_offset(0.1, signals=(1,)),),
)
result.save("artifacts/custom-system")

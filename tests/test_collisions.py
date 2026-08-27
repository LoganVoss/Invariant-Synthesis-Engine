import numpy as np

from invariant_synthesis.collisions import find_collisions


def test_consequential_collisions_respond_to_missing_coordinate() -> None:
    outcomes = np.asarray(["safe", "unsafe"] * 10)
    dashboard = np.zeros((20, 3))
    before = find_collisions(dashboard, outcomes, neighbors_per_sample=1)
    hidden = (outcomes == "unsafe").astype(float)
    repaired = np.column_stack([dashboard, hidden])
    after = find_collisions(
        repaired,
        outcomes,
        feature_weights=np.asarray([1 / 3, 1 / 3, 1 / 3, 3.0]),
        neighbors_per_sample=1,
    )
    assert before.pressure == 1.0
    assert after.pressure < before.pressure
    assert after.unresolved_fraction < before.unresolved_fraction


def test_continuous_outcome_collisions() -> None:
    x = np.linspace(0, 1, 20)[:, None]
    y = np.linspace(-2, 2, 20)
    collisions = find_collisions(x, y, outcome_kind="continuous", continuous_delta=0.5)
    assert len(collisions) > 0
    assert np.all(np.abs(y[collisions.left] - y[collisions.right]) > 0)

import numpy as np

from invariant_synthesis.data import TrajectoryDataset


def make_dataset() -> TrajectoryDataset:
    values = np.arange(8 * 6 * 2, dtype=float).reshape(8, 6, 2)
    values[0, 2:4, 0] = np.nan
    return TrajectoryDataset(
        values=values,
        outcomes=np.asarray(["a", "b"] * 4),
        signal_names=("x", "y"),
        units=("MW", "Hz"),
        groups=np.asarray(["g0", "g0", "g1", "g1", "g2", "g2", "g3", "g3"]),
    )


def test_imputation_and_canonicalization() -> None:
    dataset = make_dataset()
    imputed = dataset.imputed()
    assert np.isfinite(imputed.values).all()
    centered = imputed.canonicalized(center="initial")
    assert np.allclose(centered.values[:, 0, :], 0.0)
    assert centered.metadata["canonicalization"]["center"] == "initial"


def test_group_split_has_no_group_leakage() -> None:
    dataset = make_dataset()
    train, validation = dataset.split_indices(validation_fraction=0.25, seed=10)
    assert set(dataset.groups[train]).isdisjoint(set(dataset.groups[validation]))


def test_npz_roundtrip(tmp_path) -> None:
    dataset = make_dataset().imputed()
    path = dataset.to_npz(tmp_path / "dataset.npz")
    loaded = TrajectoryDataset.from_npz(path)
    assert loaded.signal_names == dataset.signal_names
    assert loaded.units == dataset.units
    assert np.allclose(loaded.values, dataset.values)


def test_npz_preserves_continuous_outcomes(tmp_path) -> None:
    dataset = make_dataset().imputed()
    dataset.outcomes = np.linspace(0.0, 1.0, dataset.n_samples)
    loaded = TrajectoryDataset.from_npz(dataset.to_npz(tmp_path / "continuous.npz"))
    assert loaded.outcomes.dtype.kind == "f"
    assert np.allclose(loaded.outcomes, dataset.outcomes)

from pathlib import Path

from invariant_synthesis.demo import make_energy_grid_dataset
from invariant_synthesis.engine import EngineConfig, InvariantSynthesisEngine
from invariant_synthesis.model import InvariantModel
from invariant_synthesis.search import SearchConfig
from invariant_synthesis.transformations import global_scale


def test_engine_synthesizes_scale_invariant_growth_coordinate(tmp_path: Path) -> None:
    dataset = make_energy_grid_dataset(samples=120, regions=10, seed=2401)
    config = EngineConfig(
        center="none",
        max_rounds=1,
        initial_statistics=("mean", "std", "delta"),
        primitive_statistics=("mean", "std", "delta", "early_rms", "late_rms"),
        pairwise_primitives=False,
        search=SearchConfig(
            beam_width=18,
            primitive_pool_size=64,
            max_depth=1,
            max_candidates_per_depth=5_000,
        ),
    )
    result = InvariantSynthesisEngine(config).fit(
        dataset,
        transformations=(
            global_scale(1.35, signals=(0,)),
            global_scale(0.72, signals=(0,)),
        ),
    )
    assert result.discoveries
    discovery = result.discoveries[0]
    rendered = discovery.expression.render()
    assert "frequency_deviation" in rendered
    assert "early_rms" in rendered and "late_rms" in rendered
    assert discovery.score.invariance > 0.99
    assert discovery.score.validation_outcome > 0.95
    assert discovery.validation_pressure_after < discovery.validation_pressure_before
    json_path, markdown_path = result.save(tmp_path / "report")
    assert json_path.exists() and markdown_path.exists()
    assert "Invariant Synthesis Report" in markdown_path.read_text()
    model = InvariantModel.load(json_path.parent / "synthesis_model.json")
    assert model.transform(dataset).shape == result.feature_matrix(dataset).shape
    assert (model.transform(dataset) == result.feature_matrix(dataset)).all()


def test_feature_matrix_adds_discoveries() -> None:
    dataset = make_energy_grid_dataset(samples=80, regions=8, seed=99, missing_fraction=0)
    config = EngineConfig(
        center="none",
        max_rounds=1,
        primitive_statistics=("mean", "std", "delta", "early_rms", "late_rms"),
        pairwise_primitives=False,
        search=SearchConfig(beam_width=12, primitive_pool_size=40, max_depth=1),
    )
    result = InvariantSynthesisEngine(config).fit(
        dataset,
        transformations=(global_scale(1.2, signals=(0,)),),
    )
    matrix = result.feature_matrix(dataset)
    assert matrix.shape == (
        dataset.n_samples,
        len(result.initial_features) + len(result.discoveries),
    )

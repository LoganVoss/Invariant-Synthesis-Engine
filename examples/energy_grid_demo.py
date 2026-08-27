"""Run the deterministic matched-statistics energy-grid demonstration."""

from invariant_synthesis import EngineConfig, InvariantSynthesisEngine
from invariant_synthesis.demo import make_energy_grid_dataset
from invariant_synthesis.search import SearchConfig
from invariant_synthesis.transformations import global_scale

dataset = make_energy_grid_dataset(samples=240)
config = EngineConfig(
    center="none",
    max_rounds=2,
    initial_statistics=("mean", "std", "delta"),
    primitive_statistics=("mean", "std", "delta", "early_rms", "late_rms"),
    pairwise_primitives=False,
    search=SearchConfig(max_depth=1),
)

result = InvariantSynthesisEngine(config).fit(
    dataset,
    transformations=(
        global_scale(1.35, signals=(0,)),
        global_scale(0.72, signals=(0,)),
    ),
)

for discovery in result.discoveries:
    print(discovery.expression.render())
    print(discovery.score)

result.save("artifacts/energy-grid-demo")

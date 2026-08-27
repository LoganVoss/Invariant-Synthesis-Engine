# Invariant Synthesis Engine

> Find where your measurements say **same** but the system's futures disagree—then synthesize the missing coordinate.

Invariant Synthesis Engine (ISE) is an open research framework for discovering compact,
testable observables in complex dynamical systems. It starts from *consequential
collisions*: trajectories that look alike in the current representation but lead to
materially different outcomes. It then composes candidate expressions, subjects them to
invariance and transfer tests, and adds only those that repair the ambiguity on held-out
data.

This repository is the domain-neutral foundation. The included energy-grid example is a
demonstration, not the boundary of the framework. The same loop can be applied to energy,
industrial processes, computing infrastructure, transportation, aerospace, climate,
robotics, biological systems, and experimental science.

```text
trajectories → current observables → consequential collisions
             → synthesize a discriminator
             → test invariance + transfer + simplicity
             → augment the representation → repeat
```

## Why this matters

Complex systems are only partially observed. Two physically different states can collapse
onto the same dashboard, latent vector, or engineering summary:

```text
state A → current representation Φ → "normal" → recovery
state B → current representation Φ → "normal" → cascade
```

The useful question is not merely *“Can a classifier tell these apart?”* It is:

> **What observable is missing from the current description of reality?**

ISE turns that question into a falsifiable search objective. Its output is an inspectable
coordinate such as a ratio, difference, lag relation, spectral statistic, coupling term,
or composition—not just an anomaly score. Engineers can test it, reject it, connect it to
physics, deploy it as a read-only monitor, or use the remaining collisions to justify new
measurements.

## The foundational advance

The original prototype proved the pressure loop using hand-authored hypotheses. This
version removes that central limitation. Candidate expressions are generated from a
bounded grammar and admitted through a counterexample-guided validation pipeline:

1. **Collision pressure** finds nearby trajectories with materially different outcomes.
2. **Automatic composition** searches primitives and `+`, `-`, `×`, safe division,
   absolute value, square, and log-magnitude transformations.
3. **Empirical invariance contracts** reject formulas that break under user-declared
   nuisance transformations such as gain changes, offsets, time-origin shifts, sign
   symmetries, or valid channel relabelings.
4. **Group-aware holdouts** test transfer to unseen assets, sites, regions, devices, or
   operating regimes.
5. **Block-weighted repair** gives each accepted coordinate enough representational weight
   to repair a wide dashboard instead of being numerically diluted by it.
6. **Dimensional safeguards** reject invalid additions and penalize gratuitously complex
   units while allowing physically meaningful products and ratios.
7. **Observability-gap reporting** treats failed synthesis as information. It records which
   consequential pairs remain ambiguous and clearly distinguishes bounded search failure
   from proof of impossibility.

The objective is approximately:

```text
candidate value =
    held-out collision separation
  + outcome association
  + invariance under declared transformations
  + transfer across groups
  + novelty relative to Φ
  + train/validation agreement
  - expression complexity
  - unit complexity
  - numerical instability
```

Then `Φ ← Φ ⊕ candidate`, collisions are mined again, and the description repairs itself.

## What makes it different

| Method | Primary question | Typical output |
|---|---|---|
| Anomaly detection | Is this event unusual? | anomaly score |
| Classification | Which known class fits? | label/probability |
| PCA / autoencoders | What compresses the data? | latent coordinates |
| Symbolic regression | What predicts a supplied target globally? | fitted expression |
| System identification | What dynamical model fits? | model parameters/equations |
| **Invariant synthesis** | What coordinate repairs consequential ambiguity and survives valid transformations? | validated candidate observable + unresolved collision map |

ISE can complement every method in that table. A model embedding can be the starting
representation; a simulator can supply counterfactual trajectories; a system-identification
residual can become a primitive; a discovered invariant can become a feature for a downstream
predictor.

## Industry impact

| System | Consequential collision | Potential synthesized object |
|---|---|---|
| Energy grids | recovering and unstable disturbances look alike | damping, interface, or modal coordinate |
| Data centers | benign and grid-interacting load patterns overlap | workload-to-power coupling fingerprint |
| Industrial plants | safe drift resembles runaway precursor | balance, lag, or boundary-flow relation |
| Aerospace | recoverable perturbation resembles control degradation | frame-invariant stability indicator |
| Manufacturing | normal variation resembles failure precursor | process-consistency or wear coordinate |
| Cloud / telecom | transient congestion resembles cascade onset | topology-aware load-flow invariant |
| Batteries | healthy transients resemble latent degradation | scale-stable electrothermal relation |
| Transportation | recoverable congestion resembles network collapse | flow/capacity or propagation coordinate |
| Science | different mechanisms project to the same observables | symmetry, conserved residue, or missing measurement |

The near-term product posture is intentionally **read-only**: discovery, retrospective
validation, prospective monitoring, and sensor-design evidence. The framework does not
autonomously control safety-critical infrastructure.

## Quick start

ISE requires Python 3.10 or newer.

```bash
git clone https://github.com/loganvoss/invariant-synthesis-engine.git
cd invariant-synthesis-engine
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
ise demo --output artifacts/energy-grid-demo
```

The deterministic demo deliberately makes ordinary whole-window statistics collide. The
engine recovers an amplitude-scale-invariant growth coordinate:

```text
(early_rms[frequency_deviation] / late_rms[frequency_deviation])
```

The run writes:

- `synthesis_report.md` — a human-readable validation report;
- `synthesis_result.json` — expressions, scores, collision histories, configuration, and
  unresolved pairs;
- `synthesis_model.json` — portable canonicalization, primitive, and expression AST state;
- `synthetic_grid_events.npz` — the reproducible demonstration dataset.

The synthetic data is designed to test the algorithm. It is not a validated power-system
model and should never be presented as field evidence.

## Python API

```python
from invariant_synthesis import (
    EngineConfig,
    InvariantModel,
    InvariantSynthesisEngine,
    TrajectoryDataset,
)
from invariant_synthesis.transformations import global_scale, time_shift

dataset = TrajectoryDataset(
    values=trajectories,  # shape: samples × time × signals
    outcomes=outcomes,  # categorical or continuous consequence
    signal_names=("flow", "pressure", "temperature"),
    units=("kg/s", "bar", "degC"),
    groups=asset_ids,  # optional; held out as intact groups
    sample_ids=event_ids,
    time_step=0.02,
)

config = EngineConfig(
    center="initial",  # operating-point canonicalization
    max_rounds=3,
)

engine = InvariantSynthesisEngine(config)
result = engine.fit(
    dataset,
    transformations=(
        global_scale(1.2, signals=(0,)),
        time_shift(2),
    ),
)

for discovery in result.discoveries:
    print(discovery.expression)
    print(discovery.score.validation_collision)
    print(discovery.score.invariance)

result.save("artifacts/my-run")
repaired_coordinates = result.feature_matrix(dataset)

# Apply the frozen representation later without re-running synthesis.
model = InvariantModel.load("artifacts/my-run/synthesis_model.json")
repaired_coordinates = model.transform(new_compatible_dataset)
```

Only declare a transformation if the target coordinate *should* survive it. Treating a real
physical change as a nuisance transformation can erase the signal you are trying to find.

## Dataset contract

The CLI accepts compressed NumPy `.npz` files:

| Array | Shape | Required | Meaning |
|---|---:|:---:|---|
| `values` | `N × T × D` | yes | trajectory windows |
| `outcomes` | `N` | yes | consequential labels or numeric outcomes |
| `signal_names` | `D` | yes | unique channel names |
| `units` | `D` | no | opaque unit labels; defaults to dimensionless |
| `groups` | `N` | no | asset/site/regime IDs for leakage-resistant holdout |
| `sample_ids` | `N` | no | event IDs used in unresolved-pair reports |
| `time_step` | scalar | no | sample interval; defaults to `1.0` |

Missing values are linearly interpolated within each event window. Fully absent
trajectory/channels use the across-trajectory median and are recorded in run metadata.
For high-stakes work, perform domain-appropriate quality control before synthesis.

```bash
ise discover telemetry.npz \
  --output artifacts/plant-run \
  --center initial \
  --rounds 3 \
  --global-scale 1.1
```

## Repository map

```text
src/invariant_synthesis/
  data.py             validated trajectory contract, imputation, canonicalization
  primitives.py       temporal, spectral, coupling, and pairwise descriptor leaves
  expressions.py      unit-aware symbolic grammar and safe evaluation
  collisions.py       blocked consequential-neighbor mining
  scoring.py          held-out, invariance, transfer, novelty, and complexity pressure
  search.py           bounded beam search over generated expressions
  engine.py           iterative representation-repair loop
  report.py           machine and engineering reports
  transformations.py  explicit empirical invariance contracts
```

See [Architecture](docs/architecture.md), [Methodology](docs/methodology.md),
[Industry applications](docs/industry-applications.md), and
[Extending the grammar](docs/extending.md).

## Validation philosophy

ISE is designed to make seductive formulas harder to accept:

- synthesis and acceptance use separate collision families;
- group IDs prevent the same asset or site leaking across the split;
- invariance is measured against explicit transformed trajectories;
- expression and unit complexity are penalized;
- numerical saturation is penalized;
- every run records alternatives, pressure history, configuration, and unresolved examples;
- reports say **candidate invariant**, never discovered law.

For real deployment, add chronological holdouts, prospective replay, bootstrap confidence
intervals, comparison against strong baselines, domain-specific simulation, and independent
physical review. See [Methodology](docs/methodology.md) for a staged evidence ladder.

## Current scope and honest limitations

This is a serious research foundation, not a finished autonomous scientist.

- The search is bounded beam search, so grammar exhaustion is not mathematical impossibility.
- Collision mining is blocked but still quadratic in the number of trajectories.
- The built-in unit system treats labels such as `MW` as opaque atoms; it is not a full SI
  dimensional-analysis package.
- Fixed-length aligned windows are currently required.
- The initial release focuses on scalar trajectory descriptors; graph-native and
  differential-equation grammars are roadmap items.
- Statistical validation cannot establish causality or physical meaning.
- Human review remains mandatory for safety-critical applications.

These limits are surfaced because trustworthy scientific discovery depends as much on
knowing what was *not* established as on ranking what looked promising.

## Roadmap

- approximate-neighbor collision mining for million-event archives;
- graph/topology primitives and subsystem-boundary synthesis;
- typed SI units and conservation-law templates;
- equivariant outputs, not only invariant ones;
- Pareto-front search over separation, invariance, complexity, and transfer;
- bootstrap stability and nested cross-validation;
- streaming/prospective invariant watchtower;
- sensor-value-of-information recommendations from unresolved collision fibres;
- simulator-in-the-loop interventions and causal falsification;
- pluggable primitive and operator registry.

## Contributing

Contributions are welcome, especially benchmark datasets, physically grounded
transformations, validation methods, graph primitives, and adversarial counterexamples.
Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## License and safety

MIT licensed. See [LICENSE](LICENSE).

The software is provided for research and engineering decision support. It is not certified
for autonomous operation of grids, plants, vehicles, medical devices, or other
safety-critical systems. Report security issues through [SECURITY.md](SECURITY.md).

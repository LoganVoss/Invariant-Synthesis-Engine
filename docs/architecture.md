# Architecture

Invariant Synthesis Engine is organized around one recursive operation: repair a
representation with the smallest candidate coordinate that resolves consequential
counterexamples without breaking declared invariances.

## Data flow

```text
TrajectoryDataset (N × T × D)
  │
  ├─ quality checks and interpolation
  ├─ operating-point canonicalization
  └─ group-aware train/validation split
       │
       ▼
PrimitiveLibrary
  temporal + spectral + coupling descriptors
       │
       ├──────── declared nuisance transformations ────────┐
       │                                                    │
       ▼                                                    ▼
current representation Φ                         transformed primitive views
       │
       ▼
consequential collision mining
  close under Φ + materially different outcome
       │
       ▼
unit-aware expression beam search
       │
       ▼
train collision pressure + held-out collision separation
+ invariance + group transfer + novelty - complexity
       │
       ▼
accept I* only if held-out pressure falls
       │
       ▼
Φ ← Φ ⊕ I*  ─────────────────────────────────────── repeat
```

## Core objects

### `TrajectoryDataset`

The data object owns shape validation, names, units, outcome type, asset/regime groups,
event IDs, interpolation, operating-point centering, and deterministic holdout creation.
Groups are kept intact so the same site or asset cannot appear on both sides of validation.

### `PrimitiveLibrary`

Primitives are scalar, explainable summaries of one event window. The built-in library
contains location, amplitude, change, derivative energy, early/late energy, spectral
entropy/centroid, autocorrelation, pairwise correlation, lag asymmetry, covariance, and
difference energy.

Each primitive carries a symbolic dimension. Spectral entropy and correlations are
dimensionless; derivatives divide by a symbolic time dimension; covariance multiplies
the two signal dimensions.

### `Expression`

An expression is an immutable abstract syntax tree. This makes candidates hashable,
deduplicated, serializable in reports, and inspectable. Safe division suppresses near-zero
denominators and all evaluations are clipped before scoring. Addition and subtraction are
permitted only for matching dimensions when unit enforcement is enabled.

### `CollisionSet`

Collision mining robustly scales the current coordinates, then finds close neighbors with
different categorical outcomes or sufficiently separated continuous consequences. The
search is blocked to bound peak memory, though runtime remains quadratic in sample count.

The initial dashboard receives total weight one. Each accepted synthesized coordinate also
receives block weight one. Without this convention, a powerful new scalar could be diluted
by hundreds of mediocre dashboard columns and barely alter nearest-neighbor geometry.

### `CandidateEvaluator`

Candidates are scored on separate train and validation collision sets. The evaluator also
measures global outcome association, invariance under transformed training views, lower-tail
transfer across validation groups, linear novelty relative to the current representation,
train/validation agreement, expression complexity, unit complexity, and numerical
saturation.

The score ranks search candidates. Acceptance has an additional hard gate: the coordinate
must reduce held-out collision pressure by a configured minimum.

### `ExpressionSearch`

The search begins with all primitive leaves, retains a bounded primitive pool and beam, then
composes unary and binary operators to a configured depth and complexity. It is deterministic
for a fixed dataset and configuration. This bounded search is deliberately auditable; future
backends can implement genetic programming, e-graphs, SMT, differentiable search, or Pareto
optimization behind the same evaluator.

### `SynthesisResult`

The result retains accepted expressions, alternatives, scores, collision histories,
train/validation indices, declared transformations, unresolved sample pairs, and the exact
configuration. It can transform compatible datasets into the repaired coordinate space and
write JSON and Markdown reports.

## Extension seams

- Add primitive statistics through `PrimitiveSpec` and `PrimitiveLibrary`.
- Add operators through `Expression.evaluate` plus unit rules.
- Add nuisance symmetries with `Transformation`.
- Replace collision search while preserving `CollisionSet`.
- Replace beam search while preserving `CandidateEvaluator`.
- Add domain packages that produce typed primitives without modifying the core loop.

See [Extending the grammar](extending.md) for the review contract.

# Methodology and evidence

Invariant synthesis is vulnerable to the same failure modes as symbolic regression,
multiple-hypothesis testing, and representation learning. The framework therefore treats
validation as part of synthesis, not a report added afterward.

## Formal objective

Given trajectories `X_i`, outcomes `Y_i`, and a current representation `Φ`, define
consequential collisions:

```text
C = {(i, j): distance(Φ(X_i), Φ(X_j)) is small
             and consequence(Y_i, Y_j) is large}
```

Search an expression grammar `G` for a coordinate `I` that maximizes held-out collision
separation, outcome association, invariance, group transfer, novelty, and generalization
while minimizing description length and numerical pathologies.

```text
I* = arg max over I in G:
     separation_valid(I, C_valid)
   + invariance(I, transformations)
   + transfer(I, held-out groups)
   + novelty(I, Φ)
   - complexity(I)
```

Acceptance then requires the augmented representation `Φ ⊕ I*` to lower collision pressure
on the held-out trajectories. Ranking and acceptance are intentionally separate.

## Validation boundaries

### Group holdout

When groups are supplied, complete assets/sites/regions are assigned to validation. This is
stronger than randomly splitting windows from the same machine, which can leak calibration,
topology, and operating history.

### Transformation holdout

Transformations express domain assumptions about nuisance variation. For a scale-invariant
coordinate, for example, evaluate `I(X)` and `I(scale(X))` and penalize normalized error.
The transformation set should be reviewed like model requirements; an invalid invariance
assumption can suppress real physics.

### Collision holdout

The candidate is searched against training collisions but scored and hard-gated against a
separately mined validation collision family. A formula that explains only the counterexamples
that generated it should not be accepted.

## Evidence ladder

Use progressively stronger claims:

1. **Synthetic recovery** — the engine recovers a planted relationship under known controls.
2. **Retrospective internal validation** — the coordinate transfers to held-out events and
   assets from the same archive.
3. **External validation** — it transfers to independent sites, devices, operators, or
   datasets.
4. **Prospective replay** — the frozen expression succeeds on data acquired after selection.
5. **Mechanistic falsification** — simulation or intervention changes the coordinate as its
   proposed interpretation predicts.
6. **Operational validation** — human factors, false-alarm burden, drift, cyber resilience,
   and fail-safe integration are tested.

Only the later stages justify operational or physical-law claims.

## Recommended real-world protocol

1. Define outcomes before searching and document their operational importance.
2. Align event windows using a physically defensible reference.
3. Reserve assets, sites, topology classes, and future time periods before feature search.
4. Define valid nuisance transformations with domain experts.
5. Compare against strong simple baselines and known engineering indicators.
6. Run multiple seeds/folds and report selection frequency, not one winning formula.
7. Bootstrap score and threshold uncertainty.
8. Stress missingness, sensor bias, timing error, saturation, and topology changes.
9. Freeze the expression and run prospective monitoring without updating it.
10. Ask what mechanism could produce the relation and design a falsification experiment.

## Reading an observability gap

If no candidate is accepted, the result means:

> No expression explored by this configured grammar passed these validation gates on this
> dataset.

It does **not** prove that no invariant exists or that current sensors are mathematically
insufficient. Repeated failure across richer grammars and stronger data can motivate a
value-of-information study: which new measurement would most separate the remaining pairs?

## Multiple testing

Beam search can evaluate thousands of correlated candidates. A held-out set reduces but does
not eliminate selection bias, especially when repeatedly inspected. For publishable claims,
use nested validation: search and tune inside an inner split, then evaluate the frozen engine
once on an untouched outer split. Report the number of expressions evaluated and preserve
the full run artifact.

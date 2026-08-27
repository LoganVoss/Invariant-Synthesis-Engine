# Extending the grammar

The built-in grammar is deliberately compact. Domain power should enter through reviewed,
typed extensions rather than one unbounded collection of feature tricks.

## Adding a primitive

A primitive maps an `N × T × D` trajectory tensor to one scalar per trajectory. Add a
`PrimitiveSpec`, implement its vectorized evaluation in `primitives.py`, and assign a
symbolic `Dimension`.

Every primitive should define:

- mathematical formula and window alignment assumptions;
- valid signal types and dimensions;
- behavior with constants, missingness, short windows, and near-zero values;
- computational complexity;
- transformations it should and should not survive;
- positive recovery tests and negative controls.

Useful extension families include graph cuts, modal eigenvalues, wavelets, recurrence
statistics, transfer entropy, conservation residuals, persistence summaries, and simulator
residuals.

## Adding an operator

Operators live in `Expression.evaluate`, while constructors in `expressions.py` enforce unit
rules and canonical ordering. An operator needs a safe numerical domain. Division, logs,
inverse powers, and eigenvalue ratios require explicit denominator or conditioning guards.

Avoid operators that merely hide complexity. The rendered expression should remain an object
an engineer can reason about.

## Adding a transformation

A `Transformation` receives and returns the same trajectory shape. It states a scientific
requirement: accepted coordinates should be stable under this change. Examples include:

- calibration gain or offset;
- time-origin shift;
- sign reversal around a centered operating point;
- permutation of physically equivalent devices;
- reference-frame rotation;
- topology-preserving relabeling.

Do not add a transformation simply to improve a score. Document why it is a nuisance for the
target task and test a negative case where a meaningful coordinate should change.

## Domain packages

Keep the core independent of any one industry. A future `invariant-synthesis-grid` package,
for example, can provide PMU descriptors, graph partitions, phasor-frame transformations,
and power-balance templates while depending on the same engine and report contract.

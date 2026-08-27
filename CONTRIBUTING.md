# Contributing

Thank you for helping make invariant synthesis more rigorous, useful, and falsifiable.

## Development setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
```

## Pull requests

Keep changes focused and include tests. New primitives or operators should document:

1. their mathematical definition;
2. expected input/output dimensions;
3. numerical guards and undefined cases;
4. the physical or analytical situations where they are meaningful;
5. a synthetic test where the primitive should be recovered;
6. a negative control where it should not be selected.

Do not describe a statistically selected expression as a physical law. Reports and examples
must distinguish synthetic demonstrations, retrospective evidence, and prospective field
validation.

## Design principles

- Prefer falsifiable coordinates over opaque scores.
- Preserve train/validation and group boundaries.
- Keep nuisance transformations explicit and reviewable.
- Penalize unnecessary expression and unit complexity.
- Treat unresolved collisions as first-class output.
- Keep the core domain-neutral; place specialized physics in optional extensions.

import numpy as np

from invariant_synthesis.dimensions import Dimension
from invariant_synthesis.expressions import Expression, binary_expression
from invariant_synthesis.primitives import PrimitiveSpec


def primitive(index: int, name: str, unit: str) -> Expression:
    return Expression.primitive(
        index,
        PrimitiveSpec(name, "mean", index, None, Dimension.from_unit(unit)),
    )


def test_unit_guard_rejects_invalid_addition() -> None:
    power = primitive(0, "mean[power]", "MW")
    frequency = primitive(1, "mean[frequency]", "Hz")
    assert binary_expression("add", power, frequency, enforce_units=True) is None
    ratio = binary_expression("divide", power, frequency, enforce_units=True)
    assert ratio is not None
    assert str(ratio.dimension) == "MW/Hz"


def test_safe_division_is_finite() -> None:
    left = primitive(0, "x", "1")
    right = primitive(1, "y", "1")
    ratio = binary_expression("divide", left, right)
    assert ratio is not None
    values = ratio.evaluate(np.asarray([[1.0, 0.0], [2.0, 2.0]]))
    assert np.isfinite(values).all()
    assert values.tolist() == [0.0, 1.0]


def test_commutative_expression_is_canonical() -> None:
    x = primitive(0, "x", "1")
    y = primitive(1, "y", "1")
    assert binary_expression("multiply", x, y) == binary_expression("multiply", y, x)

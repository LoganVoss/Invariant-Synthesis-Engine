"""Serializable symbolic expressions used by the synthesis search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dimensions import Dimension
from .primitives import PrimitiveSpec

COMMUTATIVE_OPERATORS = {"add", "multiply"}


@dataclass(frozen=True, slots=True)
class Expression:
    op: str
    args: tuple[Expression, ...] = ()
    primitive_index: int | None = None
    primitive_name: str | None = None
    dimension: Dimension = Dimension()

    @classmethod
    def primitive(cls, index: int, spec: PrimitiveSpec) -> Expression:
        return cls(
            op="primitive",
            primitive_index=index,
            primitive_name=spec.name,
            dimension=spec.dimension,
        )

    @property
    def complexity(self) -> int:
        return 1 if self.op == "primitive" else 1 + sum(arg.complexity for arg in self.args)

    @property
    def depth(self) -> int:
        return 0 if self.op == "primitive" else 1 + max(arg.depth for arg in self.args)

    @property
    def primitive_indices(self) -> frozenset[int]:
        if self.op == "primitive":
            assert self.primitive_index is not None
            return frozenset({self.primitive_index})
        return frozenset().union(*(arg.primitive_indices for arg in self.args))

    def evaluate(self, primitive_matrix: np.ndarray) -> np.ndarray:
        if self.op == "primitive":
            assert self.primitive_index is not None
            return primitive_matrix[:, self.primitive_index]
        values = [arg.evaluate(primitive_matrix) for arg in self.args]
        if self.op == "abs":
            result = np.abs(values[0])
        elif self.op == "square":
            result = np.square(values[0])
        elif self.op == "log1p_abs":
            result = np.log1p(np.abs(values[0]))
        elif self.op == "add":
            result = values[0] + values[1]
        elif self.op == "subtract":
            result = values[0] - values[1]
        elif self.op == "multiply":
            result = values[0] * values[1]
        elif self.op == "divide":
            denominator = values[1]
            scale = np.median(np.abs(denominator))
            floor = max(1e-12, 1e-8 * float(scale))
            result = np.divide(
                values[0],
                denominator,
                out=np.zeros_like(values[0]),
                where=np.abs(denominator) > floor,
            )
        else:
            raise ValueError(f"unknown expression operator: {self.op}")
        return np.clip(np.nan_to_num(result), -1e12, 1e12)

    def render(self) -> str:
        if self.op == "primitive":
            return self.primitive_name or f"primitive_{self.primitive_index}"
        if self.op == "abs":
            return f"abs({self.args[0].render()})"
        if self.op == "square":
            return f"square({self.args[0].render()})"
        if self.op == "log1p_abs":
            return f"log1p_abs({self.args[0].render()})"
        symbol = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/"}[self.op]
        return f"({self.args[0].render()} {symbol} {self.args[1].render()})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "args": [arg.to_dict() for arg in self.args],
            "primitive_index": self.primitive_index,
            "primitive_name": self.primitive_name,
            "dimension": str(self.dimension),
            "rendered": self.render(),
            "complexity": self.complexity,
        }

    def __str__(self) -> str:
        return self.render()


def unary_expression(
    op: str,
    child: Expression,
    *,
    enforce_units: bool = True,
) -> Expression | None:
    if op == "abs":
        dimension = child.dimension
    elif op == "square":
        dimension = child.dimension.power(2)
    elif op == "log1p_abs":
        if enforce_units and not child.dimension.is_dimensionless:
            return None
        dimension = Dimension()
    else:
        raise ValueError(f"unsupported unary operator: {op}")
    return Expression(op=op, args=(child,), dimension=dimension)


def binary_expression(
    op: str,
    left: Expression,
    right: Expression,
    *,
    enforce_units: bool = True,
) -> Expression | None:
    if left == right and op in {"subtract", "divide"}:
        return None
    if op in COMMUTATIVE_OPERATORS and left.render() > right.render():
        left, right = right, left
    if op in {"add", "subtract"}:
        if enforce_units and left.dimension != right.dimension:
            return None
        dimension = left.dimension if left.dimension == right.dimension else Dimension()
    elif op == "multiply":
        dimension = left.dimension.multiply(right.dimension)
    elif op == "divide":
        dimension = left.dimension.divide(right.dimension)
    else:
        raise ValueError(f"unsupported binary operator: {op}")
    return Expression(op=op, args=(left, right), dimension=dimension)

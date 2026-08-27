"""Portable fitted representation model for applying discovered coordinates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .data import TrajectoryDataset
from .expressions import Expression, binary_expression, unary_expression
from .primitives import PrimitiveLibrary

if TYPE_CHECKING:
    from .engine import SynthesisResult


def _restore_expression(payload: dict[str, Any], library: PrimitiveLibrary) -> Expression:
    operation = payload["op"]
    if operation == "primitive":
        name = payload["primitive_name"]
        try:
            index = library.names.index(name)
        except ValueError as error:
            raise ValueError(f"model primitive {name!r} is unavailable") from error
        return Expression.primitive(index, library.specs[index])
    args = tuple(_restore_expression(item, library) for item in payload["args"])
    if operation in {"abs", "square", "log1p_abs"}:
        restored = unary_expression(operation, args[0], enforce_units=True)
    else:
        restored = binary_expression(operation, args[0], args[1], enforce_units=True)
    if restored is None:
        raise ValueError(f"stored expression {payload.get('rendered', operation)!r} is invalid")
    return restored


@dataclass(frozen=True, slots=True)
class InvariantModel:
    """The canonicalization, primitive grammar, and accepted expression ASTs."""

    signal_names: tuple[str, ...]
    units: tuple[str, ...]
    center: str
    scale: str
    primitive_statistics: tuple[str, ...]
    pairwise_primitives: bool
    max_pairwise_signals: int
    initial_features: tuple[str, ...]
    expression_payloads: tuple[dict[str, Any], ...]
    version: str = "0.1.0"

    @classmethod
    def from_result(cls, result: SynthesisResult) -> InvariantModel:
        return cls(
            signal_names=result._library.signal_names,
            units=result._library.units,
            center=result.config.center,
            scale=result.config.scale,
            primitive_statistics=result.config.primitive_statistics,
            pairwise_primitives=result.config.pairwise_primitives,
            max_pairwise_signals=result.config.max_pairwise_signals,
            initial_features=result.initial_features,
            expression_payloads=tuple(item.to_dict() for item in result.expressions),
        )

    def transform(self, dataset: TrajectoryDataset) -> np.ndarray:
        if dataset.signal_names != self.signal_names:
            raise ValueError(
                "dataset signal names/order differ from the fitted model: "
                f"expected {self.signal_names}, got {dataset.signal_names}"
            )
        if dataset.units != self.units:
            raise ValueError(
                f"dataset units differ from the fitted model: expected {self.units}, "
                f"got {dataset.units}"
            )
        canonical = dataset.canonicalized(center=self.center, scale=self.scale)
        library = PrimitiveLibrary(
            canonical,
            statistics=self.primitive_statistics,
            pairwise=self.pairwise_primitives,
            max_pairwise_signals=self.max_pairwise_signals,
        )
        primitives = library.evaluate(canonical)
        names = {name: index for index, name in enumerate(library.names)}
        try:
            initial_indices = [names[name] for name in self.initial_features]
        except KeyError as error:
            raise ValueError(f"model initial feature {error.args[0]!r} is unavailable") from error
        expressions = [
            _restore_expression(payload, library) for payload in self.expression_payloads
        ]
        columns = [primitives[:, index] for index in initial_indices]
        columns.extend(expression.evaluate(primitives) for expression in expressions)
        return np.column_stack(columns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "InvariantModel",
            "version": self.version,
            "signal_names": list(self.signal_names),
            "units": list(self.units),
            "canonicalization": {"center": self.center, "scale": self.scale},
            "primitive_statistics": list(self.primitive_statistics),
            "pairwise_primitives": self.pairwise_primitives,
            "max_pairwise_signals": self.max_pairwise_signals,
            "initial_features": list(self.initial_features),
            "expressions": list(self.expression_payloads),
        }

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> InvariantModel:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model_type") != "InvariantModel":
            raise ValueError("file is not an InvariantModel")
        canonicalization = payload["canonicalization"]
        return cls(
            signal_names=tuple(payload["signal_names"]),
            units=tuple(payload["units"]),
            center=canonicalization["center"],
            scale=canonicalization["scale"],
            primitive_statistics=tuple(payload["primitive_statistics"]),
            pairwise_primitives=bool(payload["pairwise_primitives"]),
            max_pairwise_signals=int(payload["max_pairwise_signals"]),
            initial_features=tuple(payload["initial_features"]),
            expression_payloads=tuple(payload["expressions"]),
            version=payload["version"],
        )

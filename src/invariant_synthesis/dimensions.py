"""Small symbolic unit system used to keep generated expressions dimensionally sane."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Dimension:
    """A product of opaque unit symbols raised to integer powers.

    The engine intentionally does not attempt a full SI parser. ``MW`` and ``Hz`` are
    treated as atomic dimensions, which is enough to reject additions such as MW + Hz
    while still permitting products and ratios.
    """

    powers: tuple[tuple[str, int], ...] = ()

    @classmethod
    def from_unit(cls, unit: str | None) -> Dimension:
        clean = (unit or "1").strip()
        if clean.lower() in {"", "1", "unitless", "dimensionless", "p.u.", "pu"}:
            return cls()
        return cls(((clean, 1),))

    @classmethod
    def from_mapping(cls, values: dict[str, int]) -> Dimension:
        return cls(tuple(sorted((name, power) for name, power in values.items() if power)))

    @property
    def is_dimensionless(self) -> bool:
        return not self.powers

    def multiply(self, other: Dimension) -> Dimension:
        result = dict(self.powers)
        for name, power in other.powers:
            result[name] = result.get(name, 0) + power
        return Dimension.from_mapping(result)

    def divide(self, other: Dimension) -> Dimension:
        result = dict(self.powers)
        for name, power in other.powers:
            result[name] = result.get(name, 0) - power
        return Dimension.from_mapping(result)

    def power(self, exponent: int) -> Dimension:
        return Dimension.from_mapping({name: power * exponent for name, power in self.powers})

    def __str__(self) -> str:
        if self.is_dimensionless:
            return "1"
        numerator: list[str] = []
        denominator: list[str] = []
        for name, power in self.powers:
            target = numerator if power > 0 else denominator
            magnitude = abs(power)
            target.append(name if magnitude == 1 else f"{name}^{magnitude}")
        top = "*".join(numerator) or "1"
        return top if not denominator else f"{top}/{'*'.join(denominator)}"

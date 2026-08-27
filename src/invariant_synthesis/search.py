"""Bounded beam search over the compositional invariant grammar."""

from __future__ import annotations

from dataclasses import dataclass

from .expressions import Expression, binary_expression, unary_expression
from .primitives import PrimitiveLibrary
from .scoring import CandidateEvaluator, CandidateScore


@dataclass(frozen=True, slots=True)
class SearchConfig:
    beam_width: int = 36
    primitive_pool_size: int = 96
    max_depth: int = 2
    max_complexity: int = 9
    max_candidates_per_depth: int = 18_000
    unary_operators: tuple[str, ...] = ("abs", "square", "log1p_abs")
    binary_operators: tuple[str, ...] = ("add", "subtract", "multiply", "divide")
    enforce_units: bool = True

    def __post_init__(self) -> None:
        if self.beam_width < 1 or self.primitive_pool_size < 1:
            raise ValueError("beam_width and primitive_pool_size must be positive")
        if self.max_depth < 1 or self.max_complexity < 1:
            raise ValueError("max_depth and max_complexity must be positive")
        if self.max_candidates_per_depth < 1:
            raise ValueError("max_candidates_per_depth must be positive")


@dataclass(frozen=True, slots=True)
class ScoredExpression:
    expression: Expression
    score: CandidateScore

    def to_dict(self) -> dict[str, object]:
        return {
            "expression": self.expression.to_dict(),
            "score": self.score.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SearchResult:
    best: ScoredExpression | None
    ranked: tuple[ScoredExpression, ...]
    evaluated_candidates: int


class ExpressionSearch:
    def __init__(self, library: PrimitiveLibrary, config: SearchConfig) -> None:
        self.library = library
        self.config = config
        self.primitives = tuple(
            Expression.primitive(index, spec) for index, spec in enumerate(library.specs)
        )

    def run(
        self,
        evaluator: CandidateEvaluator,
        *,
        excluded: set[Expression] | None = None,
        keep: int = 20,
    ) -> SearchResult:
        excluded = excluded or set()
        seen: set[Expression] = set(excluded)
        scored: list[ScoredExpression] = []
        evaluated = 0

        primitive_scored = []
        for expression in self.primitives:
            if expression in seen:
                continue
            score = evaluator.score(expression)
            evaluated += 1
            item = ScoredExpression(expression, score)
            primitive_scored.append(item)
            scored.append(item)
            seen.add(expression)
        primitive_scored.sort(key=lambda item: item.score.search_total, reverse=True)
        pool_count = min(self.config.primitive_pool_size, len(primitive_scored))
        primitive_pool = [item.expression for item in primitive_scored[:pool_count]]

        initial: list[ScoredExpression] = list(primitive_scored)
        for primitive in primitive_pool:
            for operator in self.config.unary_operators:
                expression = unary_expression(
                    operator,
                    primitive,
                    enforce_units=self.config.enforce_units,
                )
                if expression is None or expression in seen:
                    continue
                if (
                    expression.complexity > self.config.max_complexity
                    or expression.depth > self.config.max_depth
                ):
                    continue
                score = evaluator.score(expression)
                evaluated += 1
                item = ScoredExpression(expression, score)
                initial.append(item)
                scored.append(item)
                seen.add(expression)
        initial.sort(key=lambda item: item.score.search_total, reverse=True)
        beam = initial[: self.config.beam_width]

        for _depth in range(1, self.config.max_depth + 1):
            generated: list[ScoredExpression] = []
            generation_count = 0
            for beam_item in beam:
                if generation_count >= self.config.max_candidates_per_depth:
                    break
                parent = beam_item.expression
                for primitive in primitive_pool:
                    if generation_count >= self.config.max_candidates_per_depth:
                        break
                    for operator in self.config.binary_operators:
                        expression = binary_expression(
                            operator,
                            parent,
                            primitive,
                            enforce_units=self.config.enforce_units,
                        )
                        if expression is None or expression in seen:
                            continue
                        if (
                            expression.complexity > self.config.max_complexity
                            or expression.depth > self.config.max_depth
                        ):
                            continue
                        seen.add(expression)
                        generation_count += 1
                        score = evaluator.score(expression)
                        evaluated += 1
                        item = ScoredExpression(expression, score)
                        generated.append(item)
                        scored.append(item)
                for operator in self.config.unary_operators:
                    expression = unary_expression(
                        operator,
                        parent,
                        enforce_units=self.config.enforce_units,
                    )
                    if expression is None or expression in seen:
                        continue
                    if (
                        expression.complexity > self.config.max_complexity
                        or expression.depth > self.config.max_depth
                    ):
                        continue
                    seen.add(expression)
                    generation_count += 1
                    score = evaluator.score(expression)
                    evaluated += 1
                    item = ScoredExpression(expression, score)
                    generated.append(item)
                    scored.append(item)
            if not generated:
                break
            generated.sort(key=lambda item: item.score.search_total, reverse=True)
            beam = generated[: self.config.beam_width]

        # The validation metrics never steer grammar expansion. Only the small shortlist
        # chosen by the training objective is handed to the engine's held-out gate.
        scored.sort(key=lambda item: item.score.search_total, reverse=True)
        ranked = tuple(scored[:keep])
        return SearchResult(
            best=ranked[0] if ranked else None,
            ranked=ranked,
            evaluated_candidates=evaluated,
        )

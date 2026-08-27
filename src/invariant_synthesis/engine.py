"""Counterexample-guided representation repair for dynamical systems."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .collisions import CollisionSet, find_collisions
from .data import TrajectoryDataset
from .expressions import Expression
from .primitives import DEFAULT_STATISTICS, PrimitiveLibrary
from .scoring import CandidateEvaluator, CandidateScore, ScoreWeights
from .search import ExpressionSearch, ScoredExpression, SearchConfig
from .transformations import Transformation


@dataclass(slots=True)
class EngineConfig:
    """Controls data canonicalization, search pressure, and acceptance safeguards."""

    center: str = "initial"
    scale: str = "none"
    validation_fraction: float = 0.25
    seed: int = 1729
    max_rounds: int = 3
    initial_statistics: tuple[str, ...] = ("mean", "std", "delta")
    primitive_statistics: tuple[str, ...] = DEFAULT_STATISTICS
    pairwise_primitives: bool = True
    max_pairwise_signals: int = 16
    neighbors_per_sample: int = 2
    max_collision_pairs: int = 512
    unresolved_distance: float = 0.75
    continuous_outcome_delta: float = 0.75
    minimum_score: float = 0.52
    minimum_validation_collision: float = 0.12
    minimum_novelty: float = 0.05
    minimum_pressure_reduction: float = 0.005
    target_collision_pressure: float = 0.20
    target_unresolved_fraction: float = 0.15
    complexity_cost: float = 0.012
    unit_complexity_cost: float = 0.008
    score_weights: ScoreWeights = field(default_factory=ScoreWeights)
    search: SearchConfig = field(default_factory=SearchConfig)

    def __post_init__(self) -> None:
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least one")
        if not 0 <= self.minimum_score <= 1:
            raise ValueError("minimum_score must be between zero and one")


@dataclass(frozen=True, slots=True)
class Discovery:
    round_number: int
    expression: Expression
    score: CandidateScore
    train_pressure_before: float
    train_pressure_after: float
    validation_pressure_before: float
    validation_pressure_after: float
    evaluated_candidates: int
    alternatives: tuple[ScoredExpression, ...]

    @property
    def validation_pressure_reduction(self) -> float:
        return self.validation_pressure_before - self.validation_pressure_after

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_number,
            "expression": self.expression.to_dict(),
            "score": self.score.to_dict(),
            "pressure": {
                "train_before": self.train_pressure_before,
                "train_after": self.train_pressure_after,
                "validation_before": self.validation_pressure_before,
                "validation_after": self.validation_pressure_after,
                "validation_reduction": self.validation_pressure_reduction,
            },
            "evaluated_candidates": self.evaluated_candidates,
            "alternatives": [item.to_dict() for item in self.alternatives],
        }


@dataclass(slots=True)
class SynthesisResult:
    config: EngineConfig
    initial_features: tuple[str, ...]
    discoveries: tuple[Discovery, ...]
    stop_reason: str
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_collision_history: tuple[CollisionSet, ...]
    validation_collision_history: tuple[CollisionSet, ...]
    unresolved_sample_pairs: tuple[tuple[str, str], ...]
    outcome_kind: str
    transformation_names: tuple[str, ...]
    dataset_summary: dict[str, Any]
    _library: PrimitiveLibrary = field(repr=False)

    @property
    def expressions(self) -> tuple[Expression, ...]:
        return tuple(item.expression for item in self.discoveries)

    @property
    def resolved(self) -> bool:
        if not self.validation_collision_history:
            return True
        final = self.validation_collision_history[-1]
        return (
            final.pressure <= self.config.target_collision_pressure
            or final.unresolved_fraction <= self.config.target_unresolved_fraction
        )

    def feature_matrix(self, dataset: TrajectoryDataset) -> np.ndarray:
        canonical = dataset.canonicalized(center=self.config.center, scale=self.config.scale)
        primitives = self._library.evaluate(canonical)
        name_to_index = {name: index for index, name in enumerate(self._library.names)}
        initial = [name_to_index[name] for name in self.initial_features]
        columns = [primitives[:, index] for index in initial]
        columns.extend(expression.evaluate(primitives) for expression in self.expressions)
        return np.column_stack(columns)

    def to_dict(self) -> dict[str, Any]:
        final_train = self.train_collision_history[-1] if self.train_collision_history else None
        final_validation = (
            self.validation_collision_history[-1] if self.validation_collision_history else None
        )
        return {
            "engine": "Invariant Synthesis Engine",
            "version": "0.1.0",
            "status": "resolved" if self.resolved else "partially_resolved",
            "stop_reason": self.stop_reason,
            "outcome_kind": self.outcome_kind,
            "dataset": self.dataset_summary,
            "config": asdict(self.config),
            "declared_transformations": list(self.transformation_names),
            "initial_features": list(self.initial_features),
            "discoveries": [item.to_dict() for item in self.discoveries],
            "collision_history": {
                "train": [item.to_dict() for item in self.train_collision_history],
                "validation": [item.to_dict() for item in self.validation_collision_history],
            },
            "final_pressure": {
                "train": None if final_train is None else final_train.pressure,
                "validation": None if final_validation is None else final_validation.pressure,
            },
            "observability_gap": {
                "remaining_pair_count": len(self.unresolved_sample_pairs),
                "sample_pairs": [list(pair) for pair in self.unresolved_sample_pairs],
                "interpretation": (
                    "The configured collision-resolution target was reached on held-out data."
                    if self.resolved
                    else (
                        "Accepted coordinates reduced held-out collisions, but the configured "
                        "resolution target was not reached. Remaining ambiguity may require more "
                        "data, a richer grammar, or new sensors."
                        if self.discoveries
                        else "No expression in the configured grammar met the held-out acceptance "
                        "criteria. This bounded result is not proof that no invariant exists."
                    )
                ),
            },
        }

    def save(self, directory: str | Path) -> tuple[Path, Path]:
        from .report import write_result

        return write_result(self, directory)


class InvariantSynthesisEngine:
    """Discover compact coordinates that repair consequential representation collisions."""

    def __init__(self, config: EngineConfig | None = None) -> None:
        self.config = config or EngineConfig()

    def fit(
        self,
        dataset: TrajectoryDataset,
        *,
        transformations: Iterable[Transformation] = (),
        initial_features: Iterable[str] | None = None,
    ) -> SynthesisResult:
        config = self.config
        canonical = dataset.canonicalized(center=config.center, scale=config.scale)
        train_idx, validation_idx = canonical.split_indices(
            validation_fraction=config.validation_fraction,
            seed=config.seed,
        )
        library = PrimitiveLibrary(
            canonical,
            statistics=config.primitive_statistics,
            pairwise=config.pairwise_primitives,
            max_pairwise_signals=config.max_pairwise_signals,
        )
        primitive_all = library.evaluate(canonical)
        selected_initial = self._resolve_initial_features(
            library,
            initial_features=initial_features,
        )
        name_to_index = {name: index for index, name in enumerate(library.names)}
        initial_indices = [name_to_index[name] for name in selected_initial]
        current_all = primitive_all[:, initial_indices]
        # The starting dashboard is one representation block. Each accepted coordinate
        # becomes another block, so a wide dashboard cannot dilute a repaired coordinate.
        current_weights = np.full(len(initial_indices), 1.0 / len(initial_indices))
        transformations = tuple(transformations)
        transformed_all = tuple(
            library.evaluate_values(transformation.apply(canonical.values))
            for transformation in transformations
        )
        transformed_train = tuple(matrix[train_idx] for matrix in transformed_all)
        transformed_validation = tuple(matrix[validation_idx] for matrix in transformed_all)

        discoveries: list[Discovery] = []
        train_history: list[CollisionSet] = []
        validation_history: list[CollisionSet] = []
        excluded = {Expression.primitive(index, library.specs[index]) for index in initial_indices}
        stop_reason = "max_rounds_reached"
        search = ExpressionSearch(library, config.search)

        for round_number in range(1, config.max_rounds + 1):
            if train_history:
                train_collisions = train_history[-1]
                validation_collisions = validation_history[-1]
            else:
                train_collisions = self._collisions(
                    current_all[train_idx],
                    canonical.outcomes[train_idx],
                    canonical.outcome_kind,
                    current_weights,
                )
                validation_collisions = self._collisions(
                    current_all[validation_idx],
                    canonical.outcomes[validation_idx],
                    canonical.outcome_kind,
                    current_weights,
                )
                train_history.append(train_collisions)
                validation_history.append(validation_collisions)
            if len(train_collisions) == 0 or len(validation_collisions) == 0:
                stop_reason = "no_consequential_collisions"
                break
            if self._target_reached(validation_collisions):
                stop_reason = "target_collision_resolution_reached"
                break

            evaluator = CandidateEvaluator(
                primitive_train=primitive_all[train_idx],
                primitive_validation=primitive_all[validation_idx],
                transformed_train=transformed_train,
                transformed_validation=transformed_validation,
                outcomes_train=canonical.outcomes[train_idx],
                outcomes_validation=canonical.outcomes[validation_idx],
                groups_validation=(
                    None if canonical.groups is None else canonical.groups[validation_idx]
                ),
                train_collisions=train_collisions,
                validation_collisions=validation_collisions,
                current_train=current_all[train_idx],
                outcome_kind=canonical.outcome_kind,
                weights=config.score_weights,
                complexity_cost=config.complexity_cost,
                unit_complexity_cost=config.unit_complexity_cost,
            )
            search_result = search.run(evaluator, excluded=excluded)
            accepted = self._choose_accepted(
                search_result.ranked,
                primitive_all,
                current_all,
                canonical,
                train_idx,
                validation_idx,
                train_collisions,
                validation_collisions,
                current_weights,
            )
            if accepted is None:
                stop_reason = (
                    "search_exhausted"
                    if search_result.best is None
                    else "held_out_acceptance_threshold_not_met"
                )
                break
            chosen, train_after, validation_after = accepted
            expression_values = chosen.expression.evaluate(primitive_all)
            current_all = np.column_stack([current_all, expression_values])
            current_weights = np.append(current_weights, 1.0)
            excluded.add(chosen.expression)
            discoveries.append(
                Discovery(
                    round_number=round_number,
                    expression=chosen.expression,
                    score=chosen.score,
                    train_pressure_before=train_collisions.pressure,
                    train_pressure_after=train_after.pressure,
                    validation_pressure_before=validation_collisions.pressure,
                    validation_pressure_after=validation_after.pressure,
                    evaluated_candidates=search_result.evaluated_candidates,
                    alternatives=tuple(item for item in search_result.ranked if item != chosen)[:5],
                )
            )
            train_history.append(train_after)
            validation_history.append(validation_after)
            if self._target_reached(validation_after):
                stop_reason = "target_collision_resolution_reached"
                break

        final_validation = validation_history[-1] if validation_history else None
        unresolved_pairs: tuple[tuple[str, str], ...] = ()
        if final_validation is not None:
            unresolved_pairs = tuple(
                (
                    str(canonical.sample_ids[validation_idx[left]]),
                    str(canonical.sample_ids[validation_idx[right]]),
                )
                for left, right, distance in zip(
                    final_validation.left,
                    final_validation.right,
                    final_validation.distances,
                    strict=True,
                )
                if distance <= config.unresolved_distance
            )[:50]
        return SynthesisResult(
            config=config,
            initial_features=selected_initial,
            discoveries=tuple(discoveries),
            stop_reason=stop_reason,
            train_indices=train_idx,
            validation_indices=validation_idx,
            train_collision_history=tuple(train_history),
            validation_collision_history=tuple(validation_history),
            unresolved_sample_pairs=unresolved_pairs,
            outcome_kind=canonical.outcome_kind,
            transformation_names=tuple(item.name for item in transformations),
            dataset_summary={
                "samples": canonical.n_samples,
                "time_points": canonical.n_time,
                "signals": canonical.n_signals,
                "signal_names": list(canonical.signal_names),
                "groups": None if canonical.groups is None else len(np.unique(canonical.groups)),
                "missing_fraction_before_imputation": dataset.missing_fraction,
                "train_samples": len(train_idx),
                "validation_samples": len(validation_idx),
            },
            _library=library,
        )

    def _resolve_initial_features(
        self,
        library: PrimitiveLibrary,
        *,
        initial_features: Iterable[str] | None,
    ) -> tuple[str, ...]:
        if initial_features is not None:
            selected = tuple(initial_features)
            unknown = set(selected).difference(library.names)
            if unknown:
                raise ValueError(f"unknown initial features: {sorted(unknown)}")
        else:
            selected = tuple(
                spec.name
                for spec in library.specs
                if spec.statistic in self.config.initial_statistics and spec.signal_b is None
            )
        if not selected:
            raise ValueError("the initial representation contains no features")
        return selected

    def _collisions(
        self,
        representation: np.ndarray,
        outcomes: np.ndarray,
        outcome_kind: str,
        feature_weights: np.ndarray | None = None,
    ) -> CollisionSet:
        config = self.config
        return find_collisions(
            representation,
            outcomes,
            feature_weights=feature_weights,
            outcome_kind=outcome_kind,
            continuous_delta=config.continuous_outcome_delta,
            neighbors_per_sample=config.neighbors_per_sample,
            max_pairs=config.max_collision_pairs,
            unresolved_distance=config.unresolved_distance,
        )

    def _target_reached(self, collisions: CollisionSet) -> bool:
        return (
            collisions.pressure <= self.config.target_collision_pressure
            or collisions.unresolved_fraction <= self.config.target_unresolved_fraction
        )

    def _choose_accepted(
        self,
        ranked: tuple[ScoredExpression, ...],
        primitive_all: np.ndarray,
        current_all: np.ndarray,
        dataset: TrajectoryDataset,
        train_idx: np.ndarray,
        validation_idx: np.ndarray,
        train_before: CollisionSet,
        validation_before: CollisionSet,
        current_weights: np.ndarray,
    ) -> tuple[ScoredExpression, CollisionSet, CollisionSet] | None:
        options = []
        for candidate in ranked:
            if candidate.score.total < self.config.minimum_score:
                continue
            if candidate.score.validation_collision < self.config.minimum_validation_collision:
                continue
            if candidate.score.novelty < self.config.minimum_novelty:
                continue
            values = candidate.expression.evaluate(primitive_all)
            augmented = np.column_stack([current_all, values])
            augmented_weights = np.append(current_weights, 1.0)
            train_after = self._collisions(
                augmented[train_idx],
                dataset.outcomes[train_idx],
                dataset.outcome_kind,
                augmented_weights,
            )
            validation_after = self._collisions(
                augmented[validation_idx],
                dataset.outcomes[validation_idx],
                dataset.outcome_kind,
                augmented_weights,
            )
            reduction = validation_before.pressure - validation_after.pressure
            if reduction < self.config.minimum_pressure_reduction:
                continue
            if validation_after.unresolved_fraction > validation_before.unresolved_fraction + 1e-12:
                continue
            merit = candidate.score.total + 0.25 * reduction
            options.append((merit, candidate, train_after, validation_after))
        if not options:
            return None
        _, candidate, train_after, validation_after = max(options, key=lambda item: item[0])
        return candidate, train_after, validation_after

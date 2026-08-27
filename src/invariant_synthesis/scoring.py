"""Leakage-resistant scoring for candidate invariant expressions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .collisions import CollisionSet
from .expressions import Expression


@dataclass(frozen=True, slots=True)
class ScoreWeights:
    collision: float = 0.34
    outcome: float = 0.18
    invariance: float = 0.20
    transfer: float = 0.10
    novelty: float = 0.08
    generalization: float = 0.10


@dataclass(frozen=True, slots=True)
class CandidateScore:
    search_total: float
    total: float
    train_collision: float
    validation_collision: float
    train_outcome: float
    validation_outcome: float
    invariance: float
    transfer: float
    novelty: float
    generalization: float
    complexity_penalty: float
    unit_penalty: float
    saturation_penalty: float

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    position = 0
    while position < len(values):
        end = position + 1
        while end < len(values) and values[order[end]] == values[order[position]]:
            end += 1
        ranks[order[position:end]] = 0.5 * (position + end - 1) + 1.0
        position = end
    return ranks


def _binary_auc_separation(values: np.ndarray, positive: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=bool)
    n_positive = int(np.sum(positive))
    n_negative = len(positive) - n_positive
    if n_positive == 0 or n_negative == 0:
        return 0.0
    ranks = _rankdata(values)
    rank_sum = float(np.sum(ranks[positive]))
    auc = (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)
    return float(np.clip(2.0 * abs(auc - 0.5), 0.0, 1.0))


def outcome_separation(values: np.ndarray, outcomes: np.ndarray, kind: str) -> float:
    values = np.asarray(values, dtype=float)
    y = np.asarray(outcomes).reshape(-1)
    if len(values) < 3 or np.std(values) < 1e-12:
        return 0.0
    if kind == "continuous":
        numeric = y.astype(float)
        value_ranks = _rankdata(values)
        outcome_ranks = _rankdata(numeric)
        corr = np.corrcoef(value_ranks, outcome_ranks)[0, 1]
        return float(abs(corr)) if np.isfinite(corr) else 0.0
    labels = np.unique(y)
    if len(labels) < 2:
        return 0.0
    scores = []
    weights = []
    for label in labels:
        positive = y == label
        scores.append(_binary_auc_separation(values, positive))
        weights.append(float(np.mean(positive)))
    return float(np.average(scores, weights=weights))


def robust_spread(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    q75, q25 = np.percentile(values, [75, 25])
    return max(float(q75 - q25), float(np.std(values)), 1e-12)


def collision_separation(
    values: np.ndarray,
    collisions: CollisionSet,
    scale: float,
) -> float:
    if len(collisions) == 0:
        return 0.0
    differences = np.abs(values[collisions.left] - values[collisions.right]) / scale
    useful = float(np.median(np.clip(differences, 0.0, 20.0)))
    return float(1.0 - np.exp(-useful))


def invariance_score(
    reference: np.ndarray,
    transformed: list[np.ndarray],
    scale: float,
) -> float:
    if not transformed:
        return 1.0
    scores = []
    for values in transformed:
        normalized_error = np.median(np.abs(reference - values)) / scale
        scores.append(float(np.exp(-normalized_error)))
    return float(np.mean(scores))


def novelty_score(values: np.ndarray, current_representation: np.ndarray) -> float:
    matrix = np.asarray(current_representation, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.shape[1] == 0 or np.std(values) < 1e-12:
        return 1.0
    correlations = []
    value_ranks = _rankdata(values)
    for column in matrix.T:
        if np.std(column) < 1e-12:
            continue
        linear = np.corrcoef(values, column)[0, 1]
        monotonic = np.corrcoef(value_ranks, _rankdata(column))[0, 1]
        for correlation in (linear, monotonic):
            if np.isfinite(correlation):
                correlations.append(abs(float(correlation)))
    return 1.0 - max(correlations, default=0.0)


def group_transfer_score(
    values: np.ndarray,
    outcomes: np.ndarray,
    groups: np.ndarray | None,
    outcome_kind: str,
    fallback: float,
) -> float:
    if groups is None:
        return fallback
    scores = []
    for group in np.unique(groups):
        mask = groups == group
        if np.sum(mask) < 4:
            continue
        if outcome_kind == "categorical" and len(np.unique(outcomes[mask])) < 2:
            continue
        if outcome_kind == "continuous" and np.std(outcomes[mask].astype(float)) < 1e-12:
            continue
        score = outcome_separation(values[mask], outcomes[mask], outcome_kind)
        scores.append(score)
    if not scores:
        return fallback
    return float(np.quantile(scores, 0.25))


class CandidateEvaluator:
    """Scores expressions on independent train/validation collision families."""

    def __init__(
        self,
        *,
        primitive_train: np.ndarray,
        primitive_validation: np.ndarray,
        transformed_train: tuple[np.ndarray, ...],
        transformed_validation: tuple[np.ndarray, ...],
        outcomes_train: np.ndarray,
        outcomes_validation: np.ndarray,
        groups_validation: np.ndarray | None,
        train_collisions: CollisionSet,
        validation_collisions: CollisionSet,
        current_train: np.ndarray,
        outcome_kind: str,
        weights: ScoreWeights | None = None,
        complexity_cost: float = 0.012,
        unit_complexity_cost: float = 0.008,
    ) -> None:
        self.primitive_train = primitive_train
        self.primitive_validation = primitive_validation
        self.transformed_train = transformed_train
        self.transformed_validation = transformed_validation
        self.outcomes_train = outcomes_train
        self.outcomes_validation = outcomes_validation
        self.groups_validation = groups_validation
        self.train_collisions = train_collisions
        self.validation_collisions = validation_collisions
        self.current_train = current_train
        self.outcome_kind = outcome_kind
        self.weights = weights or ScoreWeights()
        self.complexity_cost = complexity_cost
        self.unit_complexity_cost = unit_complexity_cost

    def score(self, expression: Expression) -> CandidateScore:
        train = expression.evaluate(self.primitive_train)
        validation = expression.evaluate(self.primitive_validation)
        scale = robust_spread(train)
        train_collision = collision_separation(train, self.train_collisions, scale)
        validation_collision = collision_separation(validation, self.validation_collisions, scale)
        train_outcome = outcome_separation(train, self.outcomes_train, self.outcome_kind)
        validation_outcome = outcome_separation(
            validation, self.outcomes_validation, self.outcome_kind
        )
        transformed = [expression.evaluate(matrix) for matrix in self.transformed_train]
        transformed_valid = [expression.evaluate(matrix) for matrix in self.transformed_validation]
        train_invariance = invariance_score(train, transformed, scale)
        validation_invariance = invariance_score(validation, transformed_valid, scale)
        invariance = min(train_invariance, validation_invariance)
        novelty = novelty_score(train, self.current_train)
        generalization = min(train_collision, validation_collision)
        transfer = group_transfer_score(
            validation,
            self.outcomes_validation,
            self.groups_validation,
            self.outcome_kind,
            fallback=min(train_outcome, validation_outcome),
        )
        complexity_penalty = self.complexity_cost * max(0, expression.complexity - 1)
        unit_penalty = self.unit_complexity_cost * sum(
            abs(power) for _, power in expression.dimension.powers
        )
        saturated = np.mean((np.abs(train) >= 1e11) | ~np.isfinite(train))
        saturation_penalty = float(saturated) * 0.25
        w = self.weights
        search_total = (
            0.50 * train_collision
            + 0.25 * train_outcome
            + 0.20 * train_invariance
            + 0.05 * novelty
            - complexity_penalty
            - unit_penalty
            - saturation_penalty
        )
        total = (
            w.collision * validation_collision
            + w.outcome * validation_outcome
            + w.invariance * invariance
            + w.transfer * transfer
            + w.novelty * novelty
            + w.generalization * generalization
            - complexity_penalty
            - unit_penalty
            - saturation_penalty
        )
        return CandidateScore(
            search_total=float(search_total),
            total=float(total),
            train_collision=train_collision,
            validation_collision=validation_collision,
            train_outcome=train_outcome,
            validation_outcome=validation_outcome,
            invariance=invariance,
            transfer=transfer,
            novelty=novelty,
            generalization=generalization,
            complexity_penalty=complexity_penalty,
            unit_penalty=unit_penalty,
            saturation_penalty=saturation_penalty,
        )

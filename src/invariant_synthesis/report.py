"""Human-readable and machine-readable synthesis reports."""

from __future__ import annotations

import json
from pathlib import Path

from .engine import SynthesisResult
from .model import InvariantModel


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_markdown(result: SynthesisResult) -> str:
    payload = result.to_dict()
    dataset = payload["dataset"]
    lines = [
        "# Invariant Synthesis Report",
        "",
        f"**Status:** {payload['status'].replace('_', ' ').title()}  ",
        f"**Stop reason:** `{payload['stop_reason']}`  ",
        f"**Held-out collision pressure:** {_fmt(payload['final_pressure']['validation'])}",
        "",
        "## System and validation boundary",
        "",
        f"- Trajectories: {dataset['samples']} ({dataset['train_samples']} train, "
        f"{dataset['validation_samples']} held out)",
        f"- Window: {dataset['time_points']} time points × {dataset['signals']} signals",
        f"- Signals: {', '.join(dataset['signal_names'])}",
        f"- Outcome type: {payload['outcome_kind']}",
        f"- Held-out groups: "
        f"{dataset['groups'] if dataset['groups'] is not None else 'not supplied'}",
        f"- Declared nuisance transformations: "
        f"{', '.join(payload['declared_transformations']) or 'none'}",
        "",
        "The reported expressions were selected on training collisions and had to "
        "separate an independent held-out collision family. Transformations are user-declared "
        "assumptions, not facts inferred by the engine.",
        "",
        "## Repaired representation",
        "",
        f"Starting coordinates ({len(payload['initial_features'])}): "
        f"{', '.join(payload['initial_features'])}",
        "",
    ]
    if not result.discoveries:
        lines.extend(
            [
                "No candidate met all configured acceptance criteria.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "| Round | Synthesized coordinate | Unit | Score | Held-out separation | "
                "Invariance | Pressure reduction |",
                "|---:|---|---|---:|---:|---:|---:|",
            ]
        )
        for discovery in result.discoveries:
            lines.append(
                f"| {discovery.round_number} | `{discovery.expression.render()}` | "
                f"`{discovery.expression.dimension}` | {discovery.score.total:.3f} | "
                f"{discovery.score.validation_collision:.3f} | "
                f"{discovery.score.invariance:.3f} | "
                f"{discovery.validation_pressure_reduction:.3f} |"
            )
        lines.append("")
        for discovery in result.discoveries:
            lines.extend(
                [
                    f"### Round {discovery.round_number}",
                    "",
                    f"Expression: `{discovery.expression.render()}`",
                    "",
                    f"This expression was chosen from {discovery.evaluated_candidates:,} "
                    "evaluated candidates. Its held-out outcome association was "
                    f"{discovery.score.validation_outcome:.3f}, group-transfer score was "
                    f"{discovery.score.transfer:.3f}, novelty was "
                    f"{discovery.score.novelty:.3f}, and complexity penalty was "
                    f"{discovery.score.complexity_penalty:.3f} (unit penalty "
                    f"{discovery.score.unit_penalty:.3f}).",
                    "",
                ]
            )
    train_history = payload["collision_history"]["train"]
    validation_history = payload["collision_history"]["validation"]
    lines.extend(
        [
            "## Collision-pressure trajectory",
            "",
            "| Checkpoint | Train pressure | Held-out pressure | Held-out unresolved |",
            "|---:|---:|---:|---:|",
        ]
    )
    checkpoints = max(len(train_history), len(validation_history))
    for index in range(checkpoints):
        train = train_history[min(index, len(train_history) - 1)]
        valid = validation_history[min(index, len(validation_history) - 1)]
        lines.append(
            f"| {index} | {train['pressure']:.3f} | {valid['pressure']:.3f} | "
            f"{valid['unresolved_fraction']:.3f} |"
        )
    gap = payload["observability_gap"]
    lines.extend(
        [
            "",
            "## Observability boundary",
            "",
            gap["interpretation"],
            "",
            f"Remaining close held-out pairs recorded: {gap['remaining_pair_count']}.",
            "",
            "## Interpretation guardrails",
            "",
            "- A discovered expression is a falsifiable candidate coordinate, not a physical law.",
            "- Validation supports the sampled regimes and declared transformations only.",
            "- Remaining collisions may require a richer grammar, better event alignment, "
            "or new sensors.",
            "- Safety-critical use requires independent physical review and prospective "
            "validation.",
            "",
        ]
    )
    return "\n".join(lines)


def write_result(result: SynthesisResult, directory: str | Path) -> tuple[Path, Path]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    json_path = target / "synthesis_result.json"
    markdown_path = target / "synthesis_report.md"
    json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(result), encoding="utf-8")
    InvariantModel.from_result(result).save(target / "synthesis_model.json")
    return json_path, markdown_path

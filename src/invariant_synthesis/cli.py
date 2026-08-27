"""Command line entry point for reproducible synthesis runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import TrajectoryDataset
from .demo import make_energy_grid_dataset
from .engine import EngineConfig, InvariantSynthesisEngine
from .search import SearchConfig
from .transformations import global_scale


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ise",
        description="Synthesize robust coordinates from consequential trajectory collisions.",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    demo = subcommands.add_parser("demo", help="run the synthetic energy-grid demonstration")
    demo.add_argument("--output", type=Path, default=Path("artifacts/energy-grid-demo"))
    demo.add_argument("--samples", type=int, default=240)
    demo.add_argument("--seed", type=int, default=2401)
    demo.add_argument("--fast", action="store_true", help="use a smaller search for a smoke test")

    discover = subcommands.add_parser("discover", help="run synthesis on a trajectory NPZ")
    discover.add_argument("dataset", type=Path)
    discover.add_argument("--output", type=Path, default=Path("artifacts/discovery"))
    discover.add_argument(
        "--center",
        choices=("initial", "mean", "median", "none"),
        default="initial",
    )
    discover.add_argument("--scale", choices=("none", "robust", "std"), default="none")
    discover.add_argument("--rounds", type=int, default=3)
    discover.add_argument("--beam-width", type=int, default=36)
    discover.add_argument("--max-depth", type=int, default=2)
    discover.add_argument("--no-pairwise", action="store_true")
    discover.add_argument(
        "--global-scale",
        type=float,
        action="append",
        default=[],
        help="declare a common amplitude scaling that valid coordinates should survive",
    )
    return parser


def _print_result(result, json_path: Path, markdown_path: Path) -> None:
    print(f"status: {'resolved' if result.resolved else 'partially resolved'}")
    print(f"stop reason: {result.stop_reason}")
    for discovery in result.discoveries:
        print(
            f"round {discovery.round_number}: {discovery.expression.render()} "
            f"(score={discovery.score.total:.3f}, "
            f"held-out pressure reduction={discovery.validation_pressure_reduction:.3f})"
        )
    print(f"machine report: {json_path}")
    print(f"human report: {markdown_path}")
    print(f"portable model: {json_path.parent / 'synthesis_model.json'}")


def _run_demo(args: argparse.Namespace) -> int:
    dataset = make_energy_grid_dataset(samples=args.samples, seed=args.seed)
    if args.fast:
        search = SearchConfig(
            beam_width=18,
            primitive_pool_size=64,
            max_depth=1,
            max_candidates_per_depth=5_000,
        )
        rounds = 1
    else:
        search = SearchConfig()
        rounds = 2
    config = EngineConfig(
        center="none",
        max_rounds=rounds,
        initial_statistics=("mean", "std", "delta"),
        primitive_statistics=("mean", "std", "delta", "early_rms", "late_rms"),
        pairwise_primitives=False,
        search=search,
        target_collision_pressure=0.25,
        target_unresolved_fraction=0.16,
    )
    result = InvariantSynthesisEngine(config).fit(
        dataset,
        transformations=(
            global_scale(1.35, signals=(0,)),
            global_scale(0.72, signals=(0,)),
        ),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    dataset.to_npz(args.output / "synthetic_grid_events.npz")
    json_path, markdown_path = result.save(args.output)
    _print_result(result, json_path, markdown_path)
    return 0


def _run_discover(args: argparse.Namespace) -> int:
    dataset = TrajectoryDataset.from_npz(args.dataset)
    config = EngineConfig(
        center=args.center,
        scale=args.scale,
        max_rounds=args.rounds,
        pairwise_primitives=not args.no_pairwise,
        search=SearchConfig(beam_width=args.beam_width, max_depth=args.max_depth),
    )
    transformations = tuple(global_scale(factor) for factor in args.global_scale)
    result = InvariantSynthesisEngine(config).fit(dataset, transformations=transformations)
    json_path, markdown_path = result.save(args.output)
    _print_result(result, json_path, markdown_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "demo":
        return _run_demo(args)
    if args.command == "discover":
        return _run_discover(args)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())

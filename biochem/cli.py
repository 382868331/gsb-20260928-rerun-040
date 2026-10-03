"""Command-line interface for offline synthetic experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis import conserved_quantities, run_short_experiments, validate_trajectory_conservation
from .deterministic import rk4_scan
from .models import build_conservation_network
from .persistence import load_model, load_state, save_model, save_state
from .stochastic import (
    FATE_ALLOW_RELEASES,
    FATE_BLOCK_RELEASES,
    FATE_INACTIVATE,
    FATE_RELEASE_SEQUESTERED,
    FATE_SEQUESTER,
    StochasticRunner,
)


def print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def parse_condition(spec: str) -> tuple[float, dict[str, float]]:
    """Parse TIME:REACTION=RATE[,REACTION=RATE...] into a scheduled change."""
    try:
        time_text, assignments_text = spec.split(":", 1)
        time_value = float(time_text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid condition {spec!r}; expected TIME:REACTION=RATE"
        ) from None
    rate_constants: dict[str, float] = {}
    for assignment in assignments_text.split(","):
        try:
            reaction_id, rate_text = assignment.split("=", 1)
            rate_constants[reaction_id.strip()] = float(rate_text)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"invalid assignment {assignment!r}; expected REACTION=RATE"
            ) from None
    if not rate_constants:
        raise argparse.ArgumentTypeError(f"condition {spec!r} changes nothing")
    return time_value, rate_constants


def parse_fate(spec: str) -> tuple[float, str, dict[str, object]]:
    """Parse TIME:KIND[:KEY=VALUE,...] into a scheduled waiting fate.

    Examples:
      2.0:inactivate:release=release-000001:amount=1:species=P
      2.0:sequester:species=P
      3.0:release_sequestered:sequester=sequester-000001
      2.5:block_releases
      3.5:allow_releases
    """
    parts = spec.split(":")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            f"invalid fate {spec!r}; expected TIME:KIND[:KEY=VALUE...]"
        )
    try:
        time_value = float(parts[0])
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid fate time in {spec!r}"
        ) from None
    kind = parts[1].strip()
    valid = {
        FATE_INACTIVATE,
        FATE_SEQUESTER,
        FATE_RELEASE_SEQUESTERED,
        FATE_BLOCK_RELEASES,
        FATE_ALLOW_RELEASES,
    }
    if kind not in valid:
        raise argparse.ArgumentTypeError(
            f"invalid fate kind {kind!r}; expected one of {sorted(valid)}"
        )
    options: dict[str, object] = {}
    for assignment in parts[2:]:
        if not assignment:
            continue
        try:
            key, value = assignment.split("=", 1)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"invalid fate option {assignment!r}; expected KEY=VALUE"
            ) from None
        key = key.strip()
        value = value.strip()
        if key in ("release", "sequester"):
            options[f"{key}_id"] = value
        elif key == "amount":
            try:
                options["amount"] = int(value)
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"fate amount must be an integer: {value!r}"
                ) from None
        elif key == "species":
            options.setdefault("products", {})[value] = 1
        elif key == "label":
            options["label"] = value
        else:
            raise argparse.ArgumentTypeError(f"unknown fate option {key!r}")
    return time_value, kind, options


def cmd_init(args: argparse.Namespace) -> None:
    save_model(build_conservation_network(), args.model)
    print(f"saved built-in conservation model to {args.model}")


def cmd_run(args: argparse.Namespace) -> None:
    if args.state and Path(args.state).exists():
        state = load_state(args.state)
        runner = StochasticRunner.from_state(state)
    else:
        model = load_model(args.model)
        runner = StochasticRunner(model, seed=args.seed)
    for time_value, rate_constants in args.condition or []:
        runner.schedule_condition_change(time_value, rate_constants, label="cli")
    for time_value, kind, options in args.fate or []:
        runner.schedule_waiting_fate(time_value, kind, label="cli", **options)
    failure = None
    try:
        runner.run(end_time=args.end_time, max_events=args.max_events)
        validate_trajectory_conservation(runner.state)
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
    if args.state:
        save_state(runner.state, args.state)
    print_json(
        {
            "time": runner.state.time,
            "status": runner.state.status,
            "events": len(runner.state.events),
            "counts": runner.state.counts,
            "waiting_counts": runner.state.waiting_counts,
            "deactivated_counts": runner.state.deactivated_counts,
            "sequestered_counts": runner.state.sequestered_counts,
            "blocked_release_ids": sorted(runner.state.blocked_release_ids),
            "releases_blocked": runner.state.releases_blocked,
            "pending_releases": [
                release.to_dict() for release in runner.state.pending_releases
            ],
            "conserved": conserved_quantities(
                runner.state.counts,
                waiting_counts=runner.state.waiting_counts,
            ),
            "condition_events": [
                change.to_dict() for change in runner.state.condition_events
            ],
            "condition_version": runner.state.condition_version,
            "fate_events": [event.to_dict() for event in runner.state.fate_events],
            "failure_reason": failure or runner.state.failure_reason,
            "state_path": args.state,
        }
    )
    if failure is not None:
        sys.exit(1)


def cmd_experiments(args: argparse.Namespace) -> None:
    model = load_model(args.model)
    _, summary = run_short_experiments(
        model=model,
        runs=args.runs,
        duration=args.duration,
        first_seed=args.first_seed,
    )
    print_json(summary.to_dict())


def cmd_deterministic(args: argparse.Namespace) -> None:
    model = load_model(args.model)
    points = rk4_scan(model, args.end_time, args.step_size)
    print_json(
        {
            "rule": "continuous-mass-action-rk4",
            "points": [
                {"time": point.time, "values": point.values} for point in points
            ],
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline synthetic biochemical reaction experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="write the built-in conservation network")
    init_parser.add_argument("--model", default="artifacts/model.json")
    init_parser.set_defaults(func=cmd_init)

    run_parser = subparsers.add_parser("run", help="run or resume one stochastic trajectory")
    run_parser.add_argument("--model", default="artifacts/model.json")
    run_parser.add_argument("--state", default="artifacts/trajectory.json")
    run_parser.add_argument("--seed", type=int, default=1)
    run_parser.add_argument("--end-time", type=float, required=True)
    run_parser.add_argument("--max-events", type=int)
    run_parser.add_argument(
        "--condition",
        action="append",
        type=parse_condition,
        metavar="TIME:REACTION=RATE",
        help="schedule a replayable rate change, e.g. 2.5:catalyze=0.5",
    )
    run_parser.add_argument(
        "--fate",
        action="append",
        type=parse_fate,
        metavar="TIME:KIND[:KEY=VALUE]",
        help=(
            "schedule a waiting fate, e.g. 2.0:inactivate:release=release-000001:amount=1:species=P; "
            "kinds: inactivate, sequester, release_sequestered, block_releases, allow_releases"
        ),
    )
    run_parser.set_defaults(func=cmd_run)

    batch_parser = subparsers.add_parser("experiments", help="compare many short random runs")
    batch_parser.add_argument("--model", default="artifacts/model.json")
    batch_parser.add_argument("--runs", type=int, default=30)
    batch_parser.add_argument("--duration", type=float, default=5.0)
    batch_parser.add_argument("--first-seed", type=int, default=100)
    batch_parser.set_defaults(func=cmd_experiments)

    deterministic_parser = subparsers.add_parser("deterministic", help="evaluate same model with RK4")
    deterministic_parser.add_argument("--model", default="artifacts/model.json")
    deterministic_parser.add_argument("--end-time", type=float, required=True)
    deterministic_parser.add_argument("--step-size", type=float, default=0.05)
    deterministic_parser.set_defaults(func=cmd_deterministic)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""Command-line interface for offline synthetic experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import conserved_quantities, run_short_experiments, validate_trajectory_conservation
from .deterministic import rk4_scan
from .models import build_conservation_network
from .persistence import load_model, load_state, save_model, save_state
from .stochastic import StochasticRunner


def print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


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
    try:
        for time, encoded_rates in args.rate_change:
            change_time = float(time)
            try:
                rate_constants = json.loads(encoded_rates)
            except json.JSONDecodeError as exc:
                runner.state.status = "failed"
                runner.state.failed_at = change_time
                runner.state.failure_reason = f"invalid rate-change JSON: {exc.msg}"
                raise ValueError(runner.state.failure_reason) from exc
            runner.schedule_rate_change(change_time, rate_constants)
        runner.run(end_time=args.end_time, max_events=args.max_events)
        validate_trajectory_conservation(runner.state)
    except Exception:
        if args.state:
            save_state(runner.state, args.state)
            print_json(
                {
                    "time": runner.state.time,
                    "status": runner.state.status,
                    "events": len(runner.state.events),
                    "counts": runner.state.counts,
                    "failed_at": runner.state.failed_at,
                    "failure_reason": runner.state.failure_reason,
                    "state_path": args.state,
                }
            )
        raise SystemExit(1)
    if args.state:
        save_state(runner.state, args.state)
    print_json(
        {
            "time": runner.state.time,
            "status": runner.state.status,
            "events": len(runner.state.events),
            "counts": runner.state.counts,
            "condition_changes": [change.to_dict() for change in runner.state.condition_changes],
            "failed_at": runner.state.failed_at,
            "failure_reason": runner.state.failure_reason,
            "conserved": conserved_quantities(runner.state.counts),
            "state_path": args.state,
        }
    )


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
        "--rate-change",
        action="append",
        nargs=2,
        metavar=("TIME", "RATE_JSON"),
        default=[],
        help='schedule rate changes, for example: 2.5 \'{"bind": 0.3}\'',
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

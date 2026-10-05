"""Conservation checks and short stochastic experiment summaries."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable

from .models import ExperimentModel
from .stochastic import StochasticRunner, TrajectoryState


CONSERVED_GROUPS = (
    ("enzyme", ("E", "ES")),
    ("substrate_atoms", ("S", "ES", "P")),
)


@dataclass(frozen=True)
class ExperimentSummary:
    runs: int
    duration: float
    start_mean: dict[str, float]
    end_mean: dict[str, float]
    mean_change: dict[str, float]
    mean_events: float

    def to_dict(self) -> dict[str, object]:
        return {
            "runs": self.runs,
            "duration": self.duration,
            "start_mean": self.start_mean,
            "end_mean": self.end_mean,
            "mean_change": self.mean_change,
            "mean_events": self.mean_events,
        }


def conserved_quantities(
    counts: dict[str, int],
    groups: Iterable[tuple[str, tuple[str, ...]]] = CONSERVED_GROUPS,
    waiting_counts: dict[str, int] | None = None,
    deactivated_counts: dict[str, int] | None = None,
    sequestered_counts: dict[str, int] | None = None,
    converted_counts: dict[str, int] | None = None,
    cancelled_counts: dict[str, int] | None = None,
) -> dict[str, int]:
    waiting_counts = waiting_counts or {}
    deactivated_counts = deactivated_counts or {}
    sequestered_counts = sequestered_counts or {}
    converted_counts = converted_counts or {}
    cancelled_counts = cancelled_counts or {}
    values: dict[str, int] = {}
    for name, species_ids in groups:
        total = 0
        for species_id in species_ids:
            total += counts.get(species_id, 0)
            total += waiting_counts.get(species_id, 0)
            total += deactivated_counts.get(species_id, 0)
            total += sequestered_counts.get(species_id, 0)
            total += converted_counts.get(species_id, 0)
            total += cancelled_counts.get(species_id, 0)
        values[name] = total
    return values


def validate_trajectory_conservation(
    state: TrajectoryState,
    groups: Iterable[tuple[str, tuple[str, ...]]] = CONSERVED_GROUPS,
) -> None:
    expected = conserved_quantities(state.model.initial_counts, groups)
    LedgerSnapshot = tuple[
        dict[str, int],
        dict[str, int],
        dict[str, int],
        dict[str, int],
        dict[str, int],
        dict[str, int],
    ]
    checkpoints: list[LedgerSnapshot] = [(state.model.initial_counts, {}, {}, {}, {}, {})]
    for event in state.events:
        checkpoints.append(
            (
                event.counts_after,
                event.waiting_after,
                event.deactivated_after,
                event.sequestered_after,
                event.converted_after,
                event.cancelled_after,
            )
        )
    for event_index, (counts, waiting, deactivated, sequestered, converted, cancelled) in enumerate(
        checkpoints
    ):
        actual = conserved_quantities(
            counts, groups, waiting, deactivated, sequestered, converted, cancelled
        )
        if actual != expected:
            raise AssertionError(
                f"conservation violated at event {event_index}: {actual} != {expected}"
            )
    # Ledger cross-check: the final snapshots must agree with the state totals.
    if state.events:
        final_event = state.events[-1]
        for ledger_name, state_ledger, snapshot in (
            ("waiting", state.waiting_counts, final_event.waiting_after),
            ("deactivated", state.deactivated_counts, final_event.deactivated_after),
            ("sequestered", state.sequestered_counts, final_event.sequestered_after),
            ("converted", state.converted_counts, final_event.converted_after),
            ("cancelled", state.cancelled_counts, final_event.cancelled_after),
        ):
            for species_id in state.model.species_ids:
                if state_ledger.get(species_id, 0) != snapshot.get(species_id, 0):
                    raise AssertionError(
                        f"{ledger_name} ledger disagrees with final event snapshot for {species_id}"
                    )


def run_short_experiments(
    model: ExperimentModel,
    runs: int,
    duration: float,
    first_seed: int = 1,
) -> tuple[list[TrajectoryState], ExperimentSummary]:
    if runs <= 0 or duration <= 0:
        raise ValueError("runs and duration must be positive")
    trajectories: list[TrajectoryState] = []
    for seed in range(first_seed, first_seed + runs):
        runner = StochasticRunner(model, seed=seed)
        state = runner.run(end_time=duration)
        validate_trajectory_conservation(state)
        trajectories.append(state)

    species_ids = [species.id for species in model.species]
    start_mean = {
        species_id: mean(float(model.initial_counts.get(species_id, 0)) for _ in trajectories)
        for species_id in species_ids
    }
    end_mean = {
        species_id: mean(float(state.counts.get(species_id, 0)) for state in trajectories)
        for species_id in species_ids
    }
    summary = ExperimentSummary(
        runs=runs,
        duration=duration,
        start_mean=start_mean,
        end_mean=end_mean,
        mean_change={
            species_id: end_mean[species_id] - start_mean[species_id]
            for species_id in species_ids
        },
        mean_events=mean(float(len(state.events)) for state in trajectories),
    )
    return trajectories, summary

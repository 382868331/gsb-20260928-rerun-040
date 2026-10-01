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
) -> dict[str, int]:
    return {
        name: sum(counts.get(species_id, 0) for species_id in species_ids)
        for name, species_ids in groups
    }


def waiting_quantities(state: TrajectoryState) -> dict[str, int]:
    """Species amounts currently withheld in the waiting pool."""

    totals = {species.id: 0 for species in state.model.species}
    for item in state.waiting:
        totals[item.species_id] = totals.get(item.species_id, 0) + item.amount
    return totals


def material_balance(
    state: TrajectoryState,
    groups: Iterable[tuple[str, tuple[str, ...]]] = CONSERVED_GROUPS,
) -> dict[str, int]:
    """Conserved totals counting both free molecules and waiting batches."""

    waiting = waiting_quantities(state)
    combined = {
        species_id: state.counts.get(species_id, 0) + waiting.get(species_id, 0)
        for species_id in {*state.counts, *waiting}
    }
    return conserved_quantities(combined, groups)


def validate_material_balance(
    state: TrajectoryState,
    groups: Iterable[tuple[str, tuple[str, ...]]] = CONSERVED_GROUPS,
) -> None:
    """Replay every initiation and release, asserting conservation throughout.

    Molecules are counted whether they are free or held in an identified
    waiting batch, so a delayed product can neither vanish early nor be
    released twice.
    """

    expected = conserved_quantities(state.model.initial_counts, groups)

    free = dict(state.model.initial_counts)
    waiting = {species_id: 0 for species_id in free}
    timeline: list[tuple[float, int, str, Any]] = [
        (event.time, 1, "reaction", event) for event in state.events
    ] + [
        (release.time, 0, "release", release)
        for release in state.release_events
    ]

    def check(label: str) -> None:
        combined = {
            species_id: free.get(species_id, 0) + waiting.get(species_id, 0)
            for species_id in {*free, *waiting}
        }
        actual = conserved_quantities(combined, groups)
        if actual != expected:
            raise AssertionError(
                f"material balance violated at {label}: {actual} != {expected}"
            )

    check("start")
    for index, (_, _, kind, entry) in enumerate(sorted(timeline, key=lambda item: (item[0], item[1]))):
        if kind == "reaction":
            for species_id, change in entry.changes.items():
                free[species_id] = free.get(species_id, 0) + change
            for descriptor in entry.waiting_created:
                waiting[str(descriptor["species_id"])] = (
                    waiting.get(str(descriptor["species_id"]), 0)
                    + int(descriptor["amount"])
                )
        else:
            waiting[entry.species_id] -= entry.amount
            free[entry.species_id] = free.get(entry.species_id, 0) + entry.amount
        check(f"event {index} ({kind})")


def validate_trajectory_conservation(
    state: TrajectoryState,
    groups: Iterable[tuple[str, tuple[str, ...]]] = CONSERVED_GROUPS,
) -> None:
    expected = conserved_quantities(state.model.initial_counts, groups)
    model_holds_waiting = any(
        reaction.delayed_products for reaction in state.model.reactions
    )
    if model_holds_waiting:
        validate_material_balance(state, groups)
        return
    checkpoints = [state.model.initial_counts] + [event.counts_after for event in state.events]
    for event_index, counts in enumerate(checkpoints):
        actual = conserved_quantities(counts, groups)
        if actual != expected:
            raise AssertionError(
                f"conservation violated at event {event_index}: {actual} != {expected}"
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

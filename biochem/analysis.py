"""Conservation checks and short stochastic experiment summaries."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable

from .models import ExperimentModel
from .stochastic import StochasticRunner, TrajectoryState


def compare_scenarios(
    base_runner: StochasticRunner,
    scenarios: dict[str, dict[str, float]],
    end_time: float,
) -> dict[str, TrajectoryState]:
    """Fork independent condition scenarios from one checkpoint.

    Each fork deep-copies the RNG state, so scenarios never share subsequent
    random draws and the checkpoint runner itself is left untouched.
    """

    results: dict[str, TrajectoryState] = {}
    for label, rate_constants in scenarios.items():
        forked = base_runner.branch(rate_constants=dict(rate_constants), label=label)
        state = forked.run(end_time=end_time)
        validate_trajectory_conservation(state)
        results[label] = state
    return results


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


def validate_trajectory_conservation(
    state: TrajectoryState,
    groups: Iterable[tuple[str, tuple[str, ...]]] = CONSERVED_GROUPS,
) -> None:
    expected = conserved_quantities(state.model.initial_counts, groups)
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

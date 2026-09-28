"""多次独立短实验与平均变化统计。"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from .conservation import assert_record_conserved
from .model import Model
from .runner import GillespieRunner


@dataclass
class ShortExperimentResult:
    seed: int
    duration: float
    initial: dict[str, int]
    final: dict[str, int]
    changes: dict[str, int] = field(default_factory=dict)
    event_count: int = 0


@dataclass
class ReplicateSummary:
    duration: float
    replicate_count: int
    initial_mean: dict[str, float]
    final_mean: dict[str, float]
    mean_changes: dict[str, float]
    rate_per_time: dict[str, float]
    results: list[ShortExperimentResult]


def run_short_experiments(
    model: Model,
    initial_counts: dict[str, int],
    duration: float,
    replicate_count: int,
    base_seed: int = 1,
    check_conservation: bool = True,
) -> ReplicateSummary:
    if replicate_count <= 0:
        raise ValueError("replicate_count 必须为正")
    results = []
    for replicate in range(replicate_count):
        seed = base_seed + replicate
        runner = GillespieRunner(model=model, seed=seed, counts=dict(initial_counts))
        record = runner.run(duration=duration)
        if check_conservation:
            assert_record_conserved(model, record)
        final = dict(runner.counts)
        changes = {
            species_id: final.get(species_id, 0) - initial_counts.get(species_id, 0)
            for species_id in model.species_ids
        }
        results.append(
            ShortExperimentResult(
                seed=seed,
                duration=duration,
                initial=dict(initial_counts),
                final=final,
                changes=changes,
                event_count=record.event_count(),
            )
        )
    species_ids = model.species_ids
    initial_mean = {
        species_id: sum(item.initial.get(species_id, 0) for item in results) / replicate_count
        for species_id in species_ids
    }
    final_mean = {
        species_id: sum(item.final.get(species_id, 0) for item in results) / replicate_count
        for species_id in species_ids
    }
    mean_changes = {
        species_id: final_mean[species_id] - initial_mean[species_id]
        for species_id in species_ids
    }
    rate_per_time = {
        species_id: mean_changes[species_id] / duration for species_id in species_ids
    }
    return ReplicateSummary(
        duration=duration,
        replicate_count=replicate_count,
        initial_mean=initial_mean,
        final_mean=final_mean,
        mean_changes=mean_changes,
        rate_per_time=rate_per_time,
        results=results,
    )


def write_summary_csv(summary: ReplicateSummary, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["species", "initial_mean", "final_mean", "mean_change", "change_per_time"])
        for species_id in summary.initial_mean:
            writer.writerow(
                [
                    species_id,
                    summary.initial_mean[species_id],
                    summary.final_mean[species_id],
                    summary.mean_changes[species_id],
                    summary.rate_per_time[species_id],
                ]
            )


def write_replicates_csv(results: list[ShortExperimentResult], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    species_ids = list(results[0].initial) if results else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "duration", "event_count", *[f"change_{s}" for s in species_ids]])
        for result in results:
            writer.writerow(
                [result.seed, result.duration, result.event_count]
                + [result.changes[species_id] for species_id in species_ids]
            )

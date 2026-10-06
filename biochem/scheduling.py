"""Release-resource scheduling across independently identified reaction chains."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .stochastic import PendingRelease, StochasticRunner, TrajectoryState


SchedulingStrategy = Literal["earliest_target", "minimum_inactivation"]


@dataclass(frozen=True)
class ReleaseChain:
    chain_id: str
    reaction_ids: frozenset[str]
    target_species: str
    target_amount: int
    window_start: float
    window_end: float

    def __post_init__(self) -> None:
        if self.target_amount <= 0:
            raise ValueError("target amount must be positive")
        if self.window_start < 0 or self.window_end < self.window_start:
            raise ValueError("invalid release window")


@dataclass(frozen=True)
class ReleaseConstraints:
    capacity_per_slot: int = 1
    slot_interval: float = 1.0
    gate_open_time: float | None = None
    inactivation_before: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.capacity_per_slot <= 0:
            raise ValueError("capacity must be positive")
        if self.slot_interval <= 0:
            raise ValueError("slot interval must be positive")


@dataclass(frozen=True)
class ReleasePlacement:
    release_id: str
    source_event_id: str
    reaction_id: str
    chain_id: str | None
    products: dict[str, int]
    release_time: float


@dataclass(frozen=True)
class ReleasePlan:
    strategy: SchedulingStrategy
    placements: tuple[ReleasePlacement, ...]
    commitments: tuple[ReleasePlacement, ...]
    target_met_times: dict[str, float | None]
    expected_deactivated: dict[str, int]
    unmet_targets: dict[str, int]

    def placement_for(self, release_id: str) -> ReleasePlacement:
        for placement in self.placements:
            if placement.release_id == release_id:
                return placement
        raise KeyError(f"release is not in this plan: {release_id}")


@dataclass(frozen=True)
class SchedulingComparison:
    earliest_target: ReleasePlan
    minimum_inactivation: ReleasePlan


def _chain_for_reaction(
    chains: tuple[ReleaseChain, ...], reaction_id: str
) -> ReleaseChain | None:
    for chain in chains:
        if reaction_id in chain.reaction_ids:
            return chain
    return None


def _historical_released(
    state: TrajectoryState, chains: tuple[ReleaseChain, ...]
) -> dict[str, int]:
    released = {chain.chain_id: 0 for chain in chains}
    for event in state.events:
        if event.event_type != "release":
            continue
        chain = _chain_for_reaction(chains, event.reaction_id)
        if chain is None:
            continue
        released[chain.chain_id] += event.changes.get(chain.target_species, 0)
    return released


def _earliest_slot(
    earliest: float,
    constraints: ReleaseConstraints,
    occupancy: dict[int, int],
) -> tuple[int, float]:
    index = max(0, int(earliest // constraints.slot_interval))
    if index * constraints.slot_interval < earliest:
        index += 1
    while occupancy.get(index, 0) >= constraints.capacity_per_slot:
        index += 1
    return index, index * constraints.slot_interval


def build_release_plan(
    state: TrajectoryState,
    chains: tuple[ReleaseChain, ...] | list[ReleaseChain],
    strategy: SchedulingStrategy = "earliest_target",
    constraints: ReleaseConstraints | None = None,
) -> ReleasePlan:
    """Schedule only batches that are really waiting at this checkpoint."""
    chains = tuple(chains)
    constraints = constraints or ReleaseConstraints()
    if strategy not in ("earliest_target", "minimum_inactivation"):
        raise ValueError(f"unknown scheduling strategy: {strategy}")
    if len({chain.chain_id for chain in chains}) != len(chains):
        raise ValueError("chain ids must be unique")

    target_chains: dict[tuple[str, str], ReleaseChain] = {}
    for chain in chains:
        for reaction_id in chain.reaction_ids:
            key = (reaction_id, chain.target_species)
            if key in target_chains:
                raise ValueError(f"reaction target {key} belongs to multiple chains")
            target_chains[key] = chain

    candidates: dict[str, tuple[ReleaseChain | None, PendingRelease]] = {}
    committed_sources: set[str] = set()
    for release in state.pending_releases:
        chain: ReleaseChain | None = None
        for species_id in release.products:
            matched = target_chains.get((release.reaction_id, species_id))
            if matched is None:
                continue
            if chain is not None and chain != matched:
                raise RuntimeError(
                    f"batch {release.release_id} matches multiple chain targets"
                )
            chain = matched
        if chain is not None and release.source_event_id in committed_sources:
            raise RuntimeError(
                "one derived batch cannot be committed to two chains: "
                f"{release.source_event_id}"
            )
        if chain is not None:
            committed_sources.add(release.source_event_id)
        candidates[release.release_id] = (chain, release)

    historical = _historical_released(state, chains)
    remaining = {
        chain.chain_id: max(0, chain.target_amount - historical[chain.chain_id])
        for chain in chains
    }
    target_met_times = {
        chain.chain_id: state.time if historical[chain.chain_id] >= chain.target_amount else None
        for chain in chains
    }
    placements: list[ReleasePlacement] = []
    commitments: list[ReleasePlacement] = []
    occupancy: dict[int, int] = {}
    last_commit_time: dict[str, float] = {}
    chain_order = {chain.chain_id: index for index, chain in enumerate(chains)}
    expected_deactivated = {species_id: 0 for species_id in state.waiting_counts}

    while candidates:
        scored: list[tuple[tuple, str, ReleaseChain | None, PendingRelease, float]] = []
        for release_id, (chain, release) in candidates.items():
            planned_chain = chain
            if chain is not None and remaining[chain.chain_id] <= 0:
                planned_chain = None
            earliest = max(state.time, release.scheduled_time)
            if planned_chain is not None:
                earliest = max(earliest, planned_chain.window_start)
            if constraints.gate_open_time is not None:
                earliest = max(earliest, constraints.gate_open_time)
            _, release_time = _earliest_slot(earliest, constraints, occupancy)
            deadline = constraints.inactivation_before.get(release_id)
            deadline_key = deadline if deadline is not None else float("inf")
            window_end = (
                planned_chain.window_end if planned_chain is not None else float("inf")
            )
            balance_key = (
                last_commit_time.get(planned_chain.chain_id, float("inf"))
                if planned_chain is not None
                else float("inf")
            )
            order_key = (
                chain_order[planned_chain.chain_id]
                if planned_chain is not None
                else len(chains)
            )
            if strategy == "minimum_inactivation":
                key = (
                    1 if planned_chain is None else 0,
                    deadline_key,
                    1 if planned_chain is None else 0,
                    window_end,
                    release_time,
                    release_id,
                )
            else:
                key = (
                    1 if planned_chain is None else 0,
                    window_end,
                    balance_key,
                    order_key,
                    deadline_key,
                    release_time,
                )
            scored.append((key, release_id, planned_chain, release, release_time))

        _, release_id, planned_chain, release, release_time = min(scored)
        slot_index = int(round(release_time / constraints.slot_interval))
        occupancy[slot_index] = occupancy.get(slot_index, 0) + 1
        if planned_chain is not None and release_time > planned_chain.window_end:
            raise RuntimeError(
                f"release capacity cannot satisfy {planned_chain.chain_id} within its window"
            )

        deadline = constraints.inactivation_before.get(release_id)
        if deadline is not None and release_time > deadline:
            for species_id, amount in release.products.items():
                expected_deactivated[species_id] = (
                    expected_deactivated.get(species_id, 0) + amount
                )

        placement = ReleasePlacement(
            release_id=release_id,
            source_event_id=release.source_event_id,
            reaction_id=release.reaction_id,
            chain_id=planned_chain.chain_id if planned_chain else None,
            products=dict(release.products),
            release_time=release_time,
        )
        placements.append(placement)
        del candidates[release_id]
        if planned_chain is not None:
            commitments.append(placement)
            amount = release.products.get(planned_chain.target_species, 0)
            before = remaining[planned_chain.chain_id]
            remaining[planned_chain.chain_id] = max(0, before - amount)
            last_commit_time[planned_chain.chain_id] = release_time
            if before > 0 and amount >= before:
                target_met_times[planned_chain.chain_id] = release_time

    return ReleasePlan(
        strategy=strategy,
        placements=tuple(placements),
        commitments=tuple(commitments),
        target_met_times=dict(target_met_times),
        expected_deactivated={
            species_id: amount
            for species_id, amount in expected_deactivated.items()
            if amount
        },
        unmet_targets={
            chain_id: amount for chain_id, amount in remaining.items() if amount
        },
    )


def apply_release_plan(runner: StochasticRunner, plan: ReleasePlan) -> None:
    """Move real pending batches onto their planned capacity slots."""
    for placement in plan.placements:
        runner.reschedule_pending_release(placement.release_id, placement.release_time)


def replan_after_actuals(
    runner: StochasticRunner,
    chains: tuple[ReleaseChain, ...] | list[ReleaseChain],
    strategy: SchedulingStrategy,
    constraints: ReleaseConstraints | None = None,
) -> ReleasePlan:
    """Rebuild only the future plan; existing events and releases are retained."""
    return build_release_plan(runner.state, chains, strategy, constraints)


def compare_release_strategies(
    state: TrajectoryState,
    chains: tuple[ReleaseChain, ...] | list[ReleaseChain],
    constraints: ReleaseConstraints | None = None,
) -> SchedulingComparison:
    """Compare target timing with expected irreversible inactivation loss."""
    return SchedulingComparison(
        earliest_target=build_release_plan(state, chains, "earliest_target", constraints),
        minimum_inactivation=build_release_plan(
            state, chains, "minimum_inactivation", constraints
        ),
    )

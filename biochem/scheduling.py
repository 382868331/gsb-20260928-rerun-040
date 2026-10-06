"""Release-resource scheduling study across multiple reaction chains.

A scheduling study lets the researcher declare, per reaction chain, a target
free amount, a release window and an irreversible deactivation process.  The
planner then orders the *existing* waiting batches under a release-capacity
limit, comparing two policies: reaching the targets as early as possible
versus minimizing deactivation loss.  Two invariants are enforced while
planning:

* predicted conversion products are never counted as existing resources --
  only batches already waiting (pending_releases) can be allocated, never
  material that scheduled fates or future reactions might produce later;
* one derived batch cannot fully support two chains at once -- the amounts
  promised to all chains from a single batch never exceed what the batch
  actually holds.

During execution the actual conversions, cancellations, deactivations and
gate toggles may deviate from the plan.  Only the not-yet-occurred part is
replanned: reactions that already happened and amounts already released stay
exactly as recorded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .stochastic import StochasticRunner, TrajectoryState


POLICY_EARLIEST_TARGET = "earliest_target"
POLICY_MIN_DEACTIVATION = "min_deactivation"
POLICIES = (POLICY_EARLIEST_TARGET, POLICY_MIN_DEACTIVATION)

_EPS = 1e-9


@dataclass(frozen=True)
class ChainTarget:
    """One reaction chain's release goal.

    targets maps species to the free amount the chain must reach inside
    the release window [window_start, window_end].  deactivation_rate
    describes an irreversible loss process acting on the chain's waiting
    material: the longer a batch waits, the more of it is expected to be
    lost, and lost material can never be recovered.
    """

    chain_id: str
    targets: dict[str, int]
    window_start: float
    window_end: float
    deactivation_rate: float = 0.0

    def validate(self, species_ids: set[str]) -> None:
        if not self.chain_id:
            raise ValueError("chain id must be non-empty")
        if not self.targets:
            raise ValueError(f"chain {self.chain_id} declares no target")
        for species_id, amount in self.targets.items():
            if species_id not in species_ids:
                raise ValueError(
                    f"chain {self.chain_id} targets unknown species {species_id}"
                )
            if int(amount) <= 0:
                raise ValueError(
                    f"chain {self.chain_id} target for {species_id} must be positive"
                )
        if self.window_end < self.window_start:
            raise ValueError(f"chain {self.chain_id} window ends before it starts")
        if self.deactivation_rate < 0:
            raise ValueError(
                f"chain {self.chain_id} deactivation rate cannot be negative"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "targets": dict(self.targets),
            "window_start": self.window_start,
            "window_end": self.window_end,
            "deactivation_rate": self.deactivation_rate,
        }


@dataclass(frozen=True)
class ReleaseAllocation:
    """One planned commitment of part of a waiting batch to one chain."""

    chain_id: str
    release_id: str
    source_event_id: str
    products: dict[str, int]
    release_time: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "release_id": self.release_id,
            "source_event_id": self.source_event_id,
            "products": dict(self.products),
            "release_time": self.release_time,
        }


@dataclass(frozen=True)
class SchedulePlan:
    """An ordered release plan over existing waiting batches."""

    policy: str
    created_time: float
    capacity: float
    allocations: tuple[ReleaseAllocation, ...]
    unmet: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "created_time": self.created_time,
            "capacity": self.capacity,
            "allocations": [item.to_dict() for item in self.allocations],
            "unmet": {chain: dict(need) for chain, need in self.unmet.items()},
        }


def _validate_inputs(
    state: TrajectoryState,
    chains: list[ChainTarget],
    capacity: float,
    policy: str,
) -> None:
    if policy not in POLICIES:
        raise ValueError(f"unknown scheduling policy: {policy}")
    if capacity <= 0:
        raise ValueError("release capacity must be positive")
    if not chains:
        raise ValueError("at least one reaction chain is required")
    chain_ids = [chain.chain_id for chain in chains]
    if len(chain_ids) != len(set(chain_ids)):
        raise ValueError("chain ids must be unique")
    for chain in chains:
        chain.validate(set(state.model.species_ids))


def plan_release_schedule(
    state: TrajectoryState,
    chains: list[ChainTarget],
    capacity: float,
    policy: str = POLICY_EARLIEST_TARGET,
    released_credit: dict[str, dict[str, int]] | None = None,
) -> SchedulePlan:
    """Order existing waiting batches to serve the chains' targets.

    Only batches that are waiting right now are plannable resources; nothing
    that scheduled fates or future reactions are merely predicted to produce
    is counted.  released_credit carries the amounts already released for
    each chain so a replan after a deviation only covers the remaining gap.
    """
    _validate_inputs(state, chains, capacity, policy)
    now = state.time
    credit = released_credit or {}
    needs: dict[str, dict[str, int]] = {}
    for chain in chains:
        chain_credit = credit.get(chain.chain_id, {})
        needs[chain.chain_id] = {
            species_id: max(0, int(amount) - int(chain_credit.get(species_id, 0)))
            for species_id, amount in chain.targets.items()
        }

    # Existing waiting batches only -- predicted conversion products and
    # future reaction events are not resources yet.
    batches = sorted(
        state.pending_releases,
        key=lambda release: (release.scheduled_time, release.release_id),
    )
    remaining = {
        release.release_id: dict(release.products) for release in batches
    }
    batch_by_id = {release.release_id: release for release in batches}

    if policy == POLICY_EARLIEST_TARGET:
        chain_order = sorted(chains, key=lambda chain: (chain.window_start, chain.chain_id))
    else:
        chain_order = sorted(
            chains,
            key=lambda chain: (-chain.deactivation_rate, chain.window_start, chain.chain_id),
        )

    def batch_order(chain: ChainTarget) -> list[str]:
        if policy == POLICY_EARLIEST_TARGET:
            return [release.release_id for release in batches]
        # Minimize deactivation loss: the largest, most loss-exposed batches
        # of the chain's species are committed first.
        return [
            release.release_id
            for release in sorted(
                batches,
                key=lambda release: (
                    -chain.deactivation_rate * sum(release.products.values()),
                    release.scheduled_time,
                    release.release_id,
                ),
            )
        ]

    committed: dict[tuple[str, str], dict[str, int]] = {}
    for chain in chain_order:
        need = needs[chain.chain_id]
        for release_id in batch_order(chain):
            if all(amount <= 0 for amount in need.values()):
                break
            available = remaining[release_id]
            for species_id in sorted(need):
                wanted = need[species_id]
                if wanted <= 0:
                    continue
                take = min(wanted, available.get(species_id, 0))
                if take <= 0:
                    continue
                # A batch's promises across chains never exceed its contents.
                available[species_id] -= take
                need[species_id] = wanted - take
                key = (chain.chain_id, release_id)
                committed.setdefault(key, {})
                committed[key][species_id] = committed[key].get(species_id, 0) + take

    gap = 1.0 / capacity
    chain_by_id = {chain.chain_id: chain for chain in chains}

    def allocation_priority(item: tuple[tuple[str, str], dict[str, int]]) -> tuple:
        (chain_id, release_id), products = item
        chain = chain_by_id[chain_id]
        release = batch_by_id[release_id]
        if policy == POLICY_EARLIEST_TARGET:
            return (chain.window_start, release.scheduled_time, release_id, chain_id)
        exposure = chain.deactivation_rate * sum(products.values())
        return (-exposure, release.scheduled_time, release_id, chain_id)

    slots: list[float] = []
    allocations: list[ReleaseAllocation] = []
    for (chain_id, release_id), products in sorted(
        committed.items(), key=allocation_priority
    ):
        chain = chain_by_id[chain_id]
        release = batch_by_id[release_id]
        moment = max(now, chain.window_start, release.scheduled_time)
        while any(abs(slot - moment) < gap - _EPS for slot in slots):
            moment = min(slot for slot in slots if abs(slot - moment) < gap - _EPS) + gap
        if moment > chain.window_end + _EPS:
            # No capacity left inside the window: the promise is unmet.
            need = needs[chain_id]
            for species_id, amount in products.items():
                need[species_id] = need.get(species_id, 0) + amount
            continue
        slots.append(moment)
        allocations.append(
            ReleaseAllocation(
                chain_id=chain_id,
                release_id=release_id,
                source_event_id=release.source_event_id,
                products=dict(products),
                release_time=moment,
            )
        )

    unmet = {
        chain.chain_id: {
            species_id: amount
            for species_id, amount in needs[chain.chain_id].items()
            if amount > 0
        }
        for chain in chains
    }
    return SchedulePlan(
        policy=policy,
        created_time=now,
        capacity=capacity,
        allocations=tuple(allocations),
        unmet={chain: need for chain, need in unmet.items() if need},
    )


@dataclass
class ExecutionReport:
    """Outcome of executing (and replanning) one scheduling study."""

    policy: str
    plans: list[SchedulePlan] = field(default_factory=list)
    replans: list[dict[str, Any]] = field(default_factory=list)
    target_times: dict[str, float | None] = field(default_factory=dict)
    released: dict[str, dict[str, int]] = field(default_factory=dict)
    final_time: float = 0.0
    status: str = "ready"

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "plans": [plan.to_dict() for plan in self.plans],
            "replans": [dict(item) for item in self.replans],
            "target_times": dict(self.target_times),
            "released": {chain: dict(amounts) for chain, amounts in self.released.items()},
            "final_time": self.final_time,
            "status": self.status,
        }


def _apply_plan(runner: StochasticRunner, plan: SchedulePlan) -> None:
    for allocation in plan.allocations:
        if any(
            release.release_id == allocation.release_id
            for release in runner.state.pending_releases
        ):
            runner.reschedule_pending_release(
                allocation.release_id, allocation.release_time
            )


def execute_schedule(
    runner: StochasticRunner,
    chains: list[ChainTarget],
    capacity: float,
    policy: str = POLICY_EARLIEST_TARGET,
    end_time: float = 0.0,
) -> ExecutionReport:
    """Execute a scheduling plan, replanning only what has not happened yet.

    Releases that already occurred and reactions already fired are kept
    untouched; when an actual conversion, cancellation, deactivation or gate
    toggle deviates from the plan, only the remaining gap is replanned from
    the current state.
    """
    if end_time < runner.state.time:
        raise ValueError("end_time precedes current trajectory time")
    report = ExecutionReport(policy=policy)
    credit: dict[str, dict[str, int]] = {
        chain.chain_id: {} for chain in chains
    }
    target_times: dict[str, float | None] = {chain.chain_id: None for chain in chains}

    plan = plan_release_schedule(runner.state, chains, capacity, policy, credit)
    report.plans.append(plan)
    _apply_plan(runner, plan)
    outstanding = {
        allocation.release_id: allocation for allocation in plan.allocations
    }
    gate_state = (
        runner.state.releases_blocked,
        frozenset(runner.state.blocked_release_ids),
    )
    event_cursor = len(runner.state.events)

    def record_targets(moment: float) -> None:
        for chain in chains:
            if target_times[chain.chain_id] is not None:
                continue
            chain_credit = credit[chain.chain_id]
            if all(
                chain_credit.get(species_id, 0) >= amount
                for species_id, amount in chain.targets.items()
            ):
                target_times[chain.chain_id] = moment

    while runner.state.time < end_time:
        previous_time = runner.state.time
        previous_status = runner.state.status
        # Bounded single-event slices never commit anything past end_time;
        # an over-horizon draw is kept as the resumable pending event.
        runner.run(end_time=end_time, max_events=1)
        # One step can commit several same-instant events; scan them all.
        new_events = runner.state.events[event_cursor:]
        event_cursor = len(runner.state.events)
        for event in new_events:
            if event.event_type == "release" and event.release_id in outstanding:
                allocation = outstanding.pop(event.release_id)
                chain_credit = credit[allocation.chain_id]
                for species_id, amount in event.changes.items():
                    chain_credit[species_id] = chain_credit.get(species_id, 0) + amount
                record_targets(event.time)
        if (
            not new_events
            and runner.state.time == previous_time
            and runner.state.status == previous_status
        ):
            break
        if runner.state.status in ("exhausted", "failed"):
            break

        reasons: list[str] = []
        new_gate = (
            runner.state.releases_blocked,
            frozenset(runner.state.blocked_release_ids),
        )
        if new_gate != gate_state:
            gate_state = new_gate
            reasons.append("release gate changed")
        for release_id, allocation in list(outstanding.items()):
            batch = next(
                (
                    release
                    for release in runner.state.pending_releases
                    if release.release_id == release_id
                ),
                None,
            )
            if batch is None:
                reasons.append(f"batch {release_id} left the release track")
                outstanding.pop(release_id)
                continue
            shortfall = {
                species_id: amount - batch.products.get(species_id, 0)
                for species_id, amount in allocation.products.items()
                if batch.products.get(species_id, 0) < amount
            }
            if shortfall:
                reasons.append(
                    f"batch {release_id} deviates from plan: {shortfall}"
                )
                outstanding.pop(release_id)
        if not reasons:
            continue
        report.replans.append(
            {"time": runner.state.time, "reasons": reasons}
        )
        plan = plan_release_schedule(runner.state, chains, capacity, policy, credit)
        report.plans.append(plan)
        _apply_plan(runner, plan)
        for allocation in plan.allocations:
            outstanding.setdefault(allocation.release_id, allocation)

    report.target_times = target_times
    report.released = credit
    report.final_time = runner.state.time
    report.status = runner.state.status
    return report


def compare_policies(
    runner: StochasticRunner,
    chains: list[ChainTarget],
    capacity: float,
    end_time: float,
) -> dict[str, ExecutionReport]:
    """Run both policies on independent forks of the same checkpoint."""
    reports: dict[str, ExecutionReport] = {}
    for policy in POLICIES:
        branch = runner.fork()
        reports[policy] = execute_schedule(
            branch, chains, capacity, policy=policy, end_time=end_time
        )
    return reports

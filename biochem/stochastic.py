"""Stochastic reaction runner and resumable trajectory state."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from .models import ExperimentModel, Reaction
from .rules import MassActionRule, ReactionRule


@dataclass(frozen=True)
class ReactionEvent:
    time: float
    reaction_id: str
    waiting_time: float
    propensity: float
    total_propensity: float
    changes: dict[str, int]
    reason: str
    counts_after: dict[str, int]
    condition_version: int = 0
    event_id: str | None = None
    event_type: str = "reaction"
    release_id: str | None = None
    source_event_id: str | None = None
    release_condition_version: int | None = None
    waiting_after: dict[str, int] = field(default_factory=dict)
    pending_releases: tuple[dict[str, Any], ...] = ()
    destination: str = "free"
    inactivated_after: dict[str, int] = field(default_factory=dict)
    sequestered_after: dict[str, int] = field(default_factory=dict)
    gate_version: int | None = None
    root_release_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReactionEvent":
        return cls(
            time=float(data["time"]),
            reaction_id=str(data["reaction_id"]),
            waiting_time=float(data["waiting_time"]),
            propensity=float(data["propensity"]),
            total_propensity=float(data["total_propensity"]),
            changes={str(k): int(v) for k, v in data["changes"].items()},
            reason=str(data["reason"]),
            counts_after={str(k): int(v) for k, v in data["counts_after"].items()},
            condition_version=int(data.get("condition_version", 0)),
            event_id=data.get("event_id"),
            event_type=str(data.get("event_type", "reaction")),
            release_id=data.get("release_id"),
            source_event_id=data.get("source_event_id"),
            release_condition_version=data.get("release_condition_version"),
            waiting_after={
                str(k): int(v) for k, v in data.get("waiting_after", {}).items()
            },
            pending_releases=tuple(
                dict(item) for item in data.get("pending_releases", [])
            ),
            destination=str(data.get("destination", "free")),
            inactivated_after={str(k): int(v) for k, v in data.get("inactivated_after", {}).items()},
            sequestered_after={str(k): int(v) for k, v in data.get("sequestered_after", {}).items()},
            gate_version=data.get("gate_version"),
            root_release_id=data.get("root_release_id"),
        )


@dataclass(frozen=True)
class PendingRelease:
    release_id: str
    source_event_id: str
    reaction_id: str
    products: dict[str, int]
    created_time: float
    scheduled_time: float
    condition_version: int
    status: str = "waiting"
    root_release_id: str | None = None
    parent_release_id: str | None = None
    original_scheduled_time: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "release_id": self.release_id,
            "source_event_id": self.source_event_id,
            "reaction_id": self.reaction_id,
            "products": dict(self.products),
            "created_time": self.created_time,
            "scheduled_time": self.scheduled_time,
            "condition_version": self.condition_version,
            "status": self.status,
            "root_release_id": self.root_release_id,
            "parent_release_id": self.parent_release_id,
            "original_scheduled_time": (
                self.original_scheduled_time
                if self.original_scheduled_time is not None
                else self.scheduled_time
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PendingRelease":
        return cls(
            release_id=str(data["release_id"]),
            source_event_id=str(data["source_event_id"]),
            reaction_id=str(data["reaction_id"]),
            products={str(k): int(v) for k, v in data["products"].items()},
            created_time=float(data["created_time"]),
            scheduled_time=float(data["scheduled_time"]),
            condition_version=int(data["condition_version"]),
            status=str(data.get("status", "waiting")),
            root_release_id=data.get("root_release_id"),
            parent_release_id=data.get("parent_release_id"),
            original_scheduled_time=(
                float(data["original_scheduled_time"])
                if data.get("original_scheduled_time") is not None
                else None
            ),
        )

    @property
    def root_id(self) -> str:
        return self.root_release_id or self.release_id


@dataclass(frozen=True)
class ConditionChange:
    """A scheduled, replayable change of reaction rate constants."""

    time: float
    rate_constants: dict[str, float]
    label: str = ""
    version: int | None = None
    rates_after: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "rate_constants": dict(self.rate_constants),
            "label": self.label,
            "version": self.version,
            "rates_after": (
                dict(self.rates_after) if self.rates_after is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConditionChange":
        return cls(
            time=float(data["time"]),
            rate_constants={str(k): float(v) for k, v in data["rate_constants"].items()},
            label=str(data.get("label", "")),
            version=data.get("version"),
            rates_after=(
                {str(k): float(v) for k, v in data["rates_after"].items()}
                if data.get("rates_after") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class WaitingTransfer:
    time: float
    release_id: str
    destination: str
    amount: int
    label: str = ""
    products: dict[str, int] | None = None
    scheduled_time: float | None = None
    transfer_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WaitingTransfer":
        products = data.get("products")
        scheduled_time = data.get("scheduled_time")
        return cls(
            time=float(data["time"]),
            release_id=str(data["release_id"]),
            destination=str(data["destination"]),
            amount=int(data["amount"]),
            label=str(data.get("label", "")),
            products={str(k): int(v) for k, v in products.items()} if products else None,
            scheduled_time=float(scheduled_time) if scheduled_time is not None else None,
            transfer_id=data.get("transfer_id"),
        )


@dataclass(frozen=True)
class ReleaseGate:
    time: float
    allowed: bool
    label: str = ""
    version: int | None = None
    allowed_after: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReleaseGate":
        return cls(
            time=float(data["time"]),
            allowed=bool(data["allowed"]),
            label=str(data.get("label", "")),
            version=data.get("version"),
            allowed_after=data.get("allowed_after"),
        )


@dataclass
class TrajectoryState:
    model: ExperimentModel
    time: float = 0.0
    counts: dict[str, int] = field(default_factory=dict)
    events: list[ReactionEvent] = field(default_factory=list)
    rng_state: tuple[Any, ...] = field(default_factory=random.Random().getstate)
    status: str = "ready"
    pending_event: dict[str, Any] | None = None
    condition_schedule: list[ConditionChange] = field(default_factory=list)
    condition_events: list[ConditionChange] = field(default_factory=list)
    condition_version: int = 0
    waiting_counts: dict[str, int] = field(default_factory=dict)
    pending_releases: list[PendingRelease] = field(default_factory=list)
    inactivated_counts: dict[str, int] = field(default_factory=dict)
    sequestered_counts: dict[str, int] = field(default_factory=dict)
    waiting_transfer_schedule: list[WaitingTransfer] = field(default_factory=list)
    release_gate_schedule: list[ReleaseGate] = field(default_factory=list)
    gate_events: list[ReleaseGate] = field(default_factory=list)
    release_allowed: bool = True
    gate_version: int = 0
    next_event_number: int = 1
    next_release_number: int = 1
    next_transfer_number: int = 1
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        complete_counts = {species.id: 0 for species in self.model.species}
        complete_counts.update(self.counts)
        self.counts = complete_counts
        self.waiting_counts = {
            species.id: int(self.waiting_counts.get(species.id, 0))
            for species in self.model.species
        }
        self.inactivated_counts = {
            species.id: int(self.inactivated_counts.get(species.id, 0))
            for species in self.model.species
        }
        self.sequestered_counts = {
            species.id: int(self.sequestered_counts.get(species.id, 0))
            for species in self.model.species
        }
        for species_id, count in self.counts.items():
            if count < 0:
                raise ValueError(f"count cannot be negative: {species_id}")
        for species_id, count in self.waiting_counts.items():
            if count < 0:
                raise ValueError(f"waiting count cannot be negative: {species_id}")
        for bucket_name in ("inactivated_counts", "sequestered_counts"):
            for species_id, count in getattr(self, bucket_name).items():
                if count < 0:
                    raise ValueError(f"{bucket_name.removesuffix('_counts')} count cannot be negative: {species_id}")
        ledger_waiting = {species.id: 0 for species in self.model.species}
        ledger_sequestered = {species.id: 0 for species in self.model.species}
        for release in self.pending_releases:
            if release.status not in {"waiting", "sequestered", "inactivated"}:
                raise ValueError(f"invalid pending release status: {release.status}")
            ledger = ledger_waiting if release.status == "waiting" else ledger_sequestered if release.status == "sequestered" else None
            if ledger is None:
                continue
            for species_id, amount in release.products.items():
                ledger[species_id] = ledger.get(species_id, 0) + amount
        if ledger_waiting != self.waiting_counts or ledger_sequestered != self.sequestered_counts:
            raise ValueError("pending release ledger does not match waiting and sequestered counts")
        self.waiting_transfer_schedule.sort(key=lambda item: item.time)
        self.release_gate_schedule.sort(key=lambda item: item.time)
        self.condition_schedule.sort(key=lambda change: change.time)
        self.pending_releases.sort(key=lambda release: (release.scheduled_time, release.release_id))
        reaction_numbers = [
            int(event.event_id.removeprefix("event-"))
            for event in self.events
            if event.event_type == "reaction"
            and event.event_id
            and event.event_id.startswith("event-")
        ]
        release_numbers = [
            int(item.release_id.removeprefix("release-").split("-", 1)[0])
            for item in self.pending_releases
            if item.release_id.startswith("release-")
            and item.release_id.removeprefix("release-").split("-", 1)[0].isdigit()
        ]
        release_numbers.extend(
            int(event.release_id.removeprefix("release-").split("-", 1)[0])
            for event in self.events
            if event.event_type == "release"
            and event.release_id
            and event.release_id.startswith("release-")
            and event.release_id.removeprefix("release-").split("-", 1)[0].isdigit()
        )
        transfer_numbers = [
            int(event.release_id.removeprefix("transfer-").split("-", 1)[0])
            for event in self.events
            if event.event_type == "waiting_transfer"
            and event.release_id
            and event.release_id.startswith("transfer-")
            and event.release_id.removeprefix("transfer-").split("-", 1)[0].isdigit()
        ]
        if reaction_numbers:
            self.next_event_number = max(self.next_event_number, max(reaction_numbers) + 1)
        if release_numbers:
            self.next_release_number = max(self.next_release_number, max(release_numbers) + 1)
        if transfer_numbers:
            self.next_transfer_number = max(self.next_transfer_number, max(transfer_numbers) + 1)

    def to_dict(self) -> dict[str, Any]:
        rng_version, rng_values, rng_gauss = self.rng_state
        return {
            "schema_version": 1,
            "time": self.time,
            "counts": dict(self.counts),
            "events": [event.to_dict() for event in self.events],
            "rng_state": [rng_version, list(rng_values), rng_gauss],
            "status": self.status,
            "pending_event": self.pending_event,
            "condition_schedule": [change.to_dict() for change in self.condition_schedule],
            "condition_events": [change.to_dict() for change in self.condition_events],
            "condition_version": self.condition_version,
            "waiting_counts": dict(self.waiting_counts),
            "pending_releases": [release.to_dict() for release in self.pending_releases],
            "inactivated_counts": dict(self.inactivated_counts),
            "sequestered_counts": dict(self.sequestered_counts),
            "waiting_transfer_schedule": [item.to_dict() for item in self.waiting_transfer_schedule],
            "release_gate_schedule": [item.to_dict() for item in self.release_gate_schedule],
            "gate_events": [item.to_dict() for item in self.gate_events],
            "release_allowed": self.release_allowed,
            "gate_version": self.gate_version,
            "next_event_number": self.next_event_number,
            "next_release_number": self.next_release_number,
            "next_transfer_number": self.next_transfer_number,
            "failure_reason": self.failure_reason,
            "model": self.model.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryState":
        model = ExperimentModel.from_dict(data["model"])
        rng_data = data["rng_state"]
        rng_state = (int(rng_data[0]), tuple(int(v) for v in rng_data[1]), rng_data[2])
        return cls(
            model=model,
            time=float(data.get("time", 0.0)),
            counts={str(k): int(v) for k, v in data.get("counts", {}).items()},
            events=[ReactionEvent.from_dict(item) for item in data.get("events", [])],
            rng_state=rng_state,
            status=str(data.get("status", "ready")),
            pending_event=data.get("pending_event"),
            condition_schedule=[
                ConditionChange.from_dict(item)
                for item in data.get("condition_schedule", [])
            ],
            condition_events=[
                ConditionChange.from_dict(item)
                for item in data.get("condition_events", [])
            ],
            condition_version=int(data.get("condition_version", 0)),
            waiting_counts={
                str(k): int(v) for k, v in data.get("waiting_counts", {}).items()
            },
            pending_releases=[
                PendingRelease.from_dict(item)
                for item in data.get("pending_releases", [])
            ],
            inactivated_counts={str(k): int(v) for k, v in data.get("inactivated_counts", {}).items()},
            sequestered_counts={str(k): int(v) for k, v in data.get("sequestered_counts", {}).items()},
            waiting_transfer_schedule=[
                WaitingTransfer.from_dict(item)
                for item in data.get("waiting_transfer_schedule", [])
            ],
            release_gate_schedule=[
                ReleaseGate.from_dict(item) for item in data.get("release_gate_schedule", [])
            ],
            gate_events=[ReleaseGate.from_dict(item) for item in data.get("gate_events", [])],
            release_allowed=bool(data.get("release_allowed", True)),
            gate_version=int(data.get("gate_version", 0)),
            next_event_number=int(data.get("next_event_number", 1)),
            next_release_number=int(data.get("next_release_number", 1)),
            next_transfer_number=int(data.get("next_transfer_number", 1)),
            failure_reason=data.get("failure_reason"),
        )

    def fork(self) -> "TrajectoryState":
        """Deep-copy a checkpoint so branches never share RNG or event state."""

        return TrajectoryState.from_dict(self.to_dict())


class StochasticRunner:
    def __init__(
        self,
        model: ExperimentModel,
        seed: int | None = None,
        rule: ReactionRule | None = None,
    ) -> None:
        model.validate()
        self.model = model
        self.rule = rule or MassActionRule()
        rng = random.Random(seed)
        self.state = TrajectoryState(
            model=model,
            counts=dict(model.initial_counts),
            rng_state=rng.getstate(),
        )

    @classmethod
    def from_state(cls, state: TrajectoryState, rule: ReactionRule | None = None) -> "StochasticRunner":
        runner = cls.__new__(cls)
        runner.model = state.model
        runner.rule = rule or MassActionRule()
        runner.state = state
        return runner

    def schedule_condition_change(
        self,
        time: float,
        rate_constants: dict[str, float],
        label: str = "",
    ) -> ConditionChange:
        """Record a replayable rate-constant change taking effect at the given time."""
        if time < self.state.time:
            raise ValueError("condition change time precedes current trajectory time")
        for reaction_id, value in rate_constants.items():
            self.model.reaction(reaction_id)
            if value < 0:
                raise ValueError(f"rate constant cannot be negative: {reaction_id}")
        change = ConditionChange(
            time=float(time), rate_constants=dict(rate_constants), label=label
        )
        self.state.condition_schedule.append(change)
        # Stable ordering: changes scheduled for the same instant keep their
        # declaration order so a batch is deterministic across save points.
        self.state.condition_schedule.sort(key=lambda item: item.time)
        return change

    def reschedule_pending_release(
        self,
        release_id: str,
        scheduled_time: float,
    ) -> PendingRelease:
        """Move one identified, not-yet-released product batch."""
        if scheduled_time < self.state.time:
            raise ValueError("release time precedes current trajectory time")
        for index, release in enumerate(self.state.pending_releases):
            if release.release_id == release_id:
                updated = replace(release, scheduled_time=float(scheduled_time))
                self.state.pending_releases[index] = updated
                self.state.pending_releases.sort(
                    key=lambda item: (item.scheduled_time, item.release_id)
                )
                return updated
        raise KeyError(f"pending release not found: {release_id}")

    def schedule_waiting_transfer(
        self,
        time: float,
        release_id: str,
        destination: str,
        amount: int = 1,
        *,
        products: dict[str, int] | None = None,
        scheduled_time: float | None = None,
        label: str = "",
    ) -> WaitingTransfer:
        if time < self.state.time:
            raise ValueError("waiting transfer time precedes current trajectory time")
        if amount <= 0:
            raise ValueError("waiting transfer amount must be positive")
        if destination not in {"waiting", "inactivated", "sequestered", "free"}:
            raise ValueError(f"unknown waiting destination: {destination}")
        lot = self._find_pending_release(release_id)
        available = self._lot_amount(lot, products)
        if products is not None and sum(products.values()) != amount:
            raise ValueError("waiting transfer products must total the requested amount")
        root_lots = [item for item in self.state.pending_releases if item.root_id == lot.root_id]
        reserved = sum(
            planned.amount
            for planned in self.state.waiting_transfer_schedule
            if planned.release_id == lot.root_id or planned.release_id == lot.release_id
        )
        root_available = sum(
            sum(item.products.values())
            for item in root_lots
            if item.status == lot.status
        )
        if amount + reserved > root_available:
            raise ValueError("waiting transfer amount exceeds available intermediates")
        if amount > available:
            raise ValueError("waiting transfer amount exceeds available intermediates")
        if scheduled_time is not None and scheduled_time < time:
            raise ValueError("release time precedes transfer time")
        transfer = WaitingTransfer(
            time=float(time),
            release_id=lot.root_id,
            destination=destination,
            amount=amount,
            label=label,
            products=dict(products) if products is not None else None,
            scheduled_time=float(scheduled_time) if scheduled_time is not None else None,
        )
        self.state.waiting_transfer_schedule.append(transfer)
        self.state.waiting_transfer_schedule.sort(key=lambda item: item.time)
        return transfer

    def schedule_release_gate(
        self,
        time: float,
        allowed: bool,
        *,
        label: str = "",
    ) -> ReleaseGate:
        if time < self.state.time:
            raise ValueError("release gate time precedes current trajectory time")
        gate = ReleaseGate(time=float(time), allowed=bool(allowed), label=label)
        self.state.release_gate_schedule.append(gate)
        self.state.release_gate_schedule.sort(key=lambda item: item.time)
        return gate

    def _find_pending_release(self, release_id: str) -> PendingRelease:
        exact = [item for item in self.state.pending_releases if item.release_id == release_id]
        if exact:
            return exact[0]
        matches = [
            item
            for item in self.state.pending_releases
            if item.root_id == release_id and item.status in {"waiting", "sequestered"}
        ]
        matches.sort(key=lambda item: (item.status != "waiting", item.scheduled_time, item.release_id))
        if not matches:
            raise KeyError(f"pending release not found: {release_id}")
        return matches[0]

    @staticmethod
    def _lot_amount(lot: PendingRelease, products: dict[str, int] | None) -> int:
        if products is None:
            return sum(lot.products.values())
        amount = 0
        for species_id, requested in products.items():
            amount += min(lot.products.get(species_id, 0), requested)
        return amount

    def _complete_bucket(self, values: dict[str, int]) -> dict[str, int]:
        complete = {species.id: 0 for species in self.model.species}
        complete.update(values)
        return complete

    def _next_gate_time(self) -> float | None:
        return self.state.release_gate_schedule[0].time if self.state.release_gate_schedule else None

    def _next_transfer_time(self) -> float | None:
        return self.state.waiting_transfer_schedule[0].time if self.state.waiting_transfer_schedule else None

    def _policy_time(self) -> float | None:
        values = [value for value in (self._next_gate_time(), self._next_transfer_time()) if value is not None]
        return min(values) if values else None

    def _rng(self) -> random.Random:
        rng = random.Random()
        rng.setstate(self.state.rng_state)
        return rng

    def _available(self) -> tuple[list[Reaction], dict[str, float], float]:
        propensities = self.rule.propensities(self.model, self.state.counts)
        available = [
            self.model.reaction(reaction_id)
            for reaction_id, value in propensities.items()
            if value > 0.0
        ]
        return available, propensities, sum(propensities.values())

    def _next_condition_change(self) -> ConditionChange | None:
        return self.state.condition_schedule[0] if self.state.condition_schedule else None

    def _apply_due_condition_changes(self, allow_equal: bool = True) -> None:
        """Apply scheduled changes at or before the current trajectory time.

        ``allow_equal=False`` defers changes tied exactly with the current
        time; an event at that instant is fired first under the old version,
        after which the changes take effect.
        """

        def reached(change: ConditionChange) -> bool:
            return change.time < self.state.time or (
                allow_equal and change.time == self.state.time
            )

        due = [
            change for change in self.state.condition_schedule if reached(change)
        ]
        if not due:
            return

        rate_constants = {
            reaction.id: reaction.rate_constant for reaction in self.model.reactions
        }
        version = self.state.condition_version
        recorded: list[ConditionChange] = []
        for change in due:
            if (
                not recorded
                or recorded[-1].time != change.time
            ):
                version += 1
            rate_constants.update(change.rate_constants)
            snapshot = dict(rate_constants)
            applied = replace(change, version=version, rates_after=dict(snapshot))
            rate_constants = snapshot
            recorded.append(applied)
        for change in recorded:
            self.state.condition_events.append(change)
        self.state.condition_version = version
        self.state.condition_schedule = [
            change
            for change in self.state.condition_schedule
            if not reached(change)
        ]
        reactions = tuple(
            Reaction(
                id=reaction.id,
                reactants=dict(reaction.reactants),
                products=dict(reaction.products),
                product_delays=dict(reaction.product_delays),
                rate_constant=rate_constants[reaction.id],
                name=reaction.name,
            )
            for reaction in self.model.reactions
        )
        self.model = ExperimentModel(
            species=self.model.species,
            reactions=reactions,
            initial_counts=dict(self.model.initial_counts),
            name=self.model.name,
        )
        self.state.model = self.model

    def _event_reason(
        self, propensity: float, total: float, extra: str = ""
    ) -> str:
        reason = (
            f"mass-action propensity {propensity:.12g} selected from total "
            f"{total:.12g}; all reactant counts were sufficient; "
            f"condition_version={self.state.condition_version}"
        )
        return f"{reason}; {extra}" if extra else reason

    def _discard_pending_event(self, pending: dict[str, Any]) -> None:
        """Roll back a pre-drawn event invalidated by an intervening change."""

        before = pending.get("rng_state_before")
        if before is not None:
            self.state.rng_state = tuple(
                (int(before[0]), tuple(int(v) for v in before[1]), before[2])
            )

    def _snapshot_pending_releases(self) -> tuple[dict[str, Any], ...]:
        return tuple(release.to_dict() for release in self.state.pending_releases)

    def _next_release_time(self) -> float | None:
        if not self.state.release_allowed:
            return None
        times = [item.scheduled_time for item in self.state.pending_releases if item.status == "waiting"]
        return min(times) if times else None

    def _release_precedes_change(
        self, release_time: float | None, change_time: float | None
    ) -> bool:
        if release_time is None:
            return False
        if change_time is None:
            return True
        return release_time <= change_time

    @staticmethod
    def _split_products(
        lot: PendingRelease,
        amount: int,
        selected_products: dict[str, int] | None,
    ) -> tuple[dict[str, int], dict[str, int]]:
        if selected_products is not None:
            moved = dict(selected_products)
            remaining = {
                species_id: lot.products.get(species_id, 0) - moved.get(species_id, 0)
                for species_id in set(lot.products) | set(moved)
            }
        elif len(lot.products) == 1:
            species_id = next(iter(lot.products))
            moved = {species_id: amount}
            remaining = {species_id: lot.products[species_id] - amount}
        else:
            moved: dict[str, int] = {}
            remaining = dict(lot.products)
            left = amount
            for species_id in sorted(remaining):
                take = min(left, remaining[species_id])
                moved[species_id] = take
                remaining[species_id] -= take
                left -= take
            if left:
                raise ValueError("waiting transfer amount exceeds available intermediates")
        for species_id, value in moved.items():
            if value <= 0 or value > lot.products.get(species_id, 0):
                raise ValueError("waiting transfer amount exceeds available intermediates")
        return (
            {key: value for key, value in moved.items() if value > 0},
            {key: value for key, value in remaining.items() if value > 0},
        )

    def _record_event(self, event: ReactionEvent) -> ReactionEvent:
        self.state.events.append(event)
        self.state.status = "running"
        return event

    def _apply_due_release_gates(self, allow_equal: bool = True) -> list[ReactionEvent]:
        events: list[ReactionEvent] = []
        while self.state.release_gate_schedule:
            gate = self.state.release_gate_schedule[0]
            if gate.time > self.state.time or (not allow_equal and gate.time == self.state.time):
                break
            self.state.gate_version += 1
            self.state.release_allowed = gate.allowed
            applied = replace(
                gate,
                version=self.state.gate_version,
                allowed_after=self.state.release_allowed,
            )
            self.state.gate_events.append(applied)
            self.state.release_gate_schedule = self.state.release_gate_schedule[1:]
            if gate.allowed:
                shifted_releases: list[PendingRelease] = []
                for lot in self.state.pending_releases:
                    if lot.status == "waiting" and lot.scheduled_time <= gate.time:
                        shifted_releases.append(
                            replace(
                                lot,
                                scheduled_time=gate.time,
                                original_scheduled_time=(
                                    lot.original_scheduled_time
                                    if lot.original_scheduled_time is not None
                                    else lot.scheduled_time
                                ),
                            )
                        )
                    else:
                        shifted_releases.append(lot)
                self.state.pending_releases = shifted_releases
                self.state.pending_releases.sort(key=lambda item: (item.scheduled_time, item.release_id))
            event = ReactionEvent(
                time=gate.time,
                reaction_id="",
                waiting_time=0.0,
                propensity=0.0,
                total_propensity=0.0,
                changes={},
                reason=f"release gate allowed={gate.allowed} version={self.state.gate_version} label={gate.label}".rstrip(),
                counts_after=dict(self.state.counts),
                event_id=f"gate-{self.state.gate_version:06d}",
                event_type="release_gate",
                gate_version=self.state.gate_version,
                waiting_after=dict(self.state.waiting_counts),
                inactivated_after=dict(self.state.inactivated_counts),
                sequestered_after=dict(self.state.sequestered_counts),
            )
            events.append(self._record_event(event))
        return events

    def _apply_one_waiting_transfer(self, transfer: WaitingTransfer) -> ReactionEvent:
        lot = self._find_pending_release(transfer.release_id)
        valid_statuses = {"waiting", "sequestered"}
        if lot.status not in valid_statuses:
            raise RuntimeError(f"{transfer.release_id} is not available for transfer")
        if transfer.destination == "free" and not self.state.release_allowed:
            raise RuntimeError(f"release is blocked for {transfer.release_id}")
        moved_products, remaining_products = self._split_products(
            lot, transfer.amount, transfer.products
        )
        transfer_number = self.state.next_transfer_number
        self.state.next_transfer_number += 1
        transfer_id = transfer.transfer_id or f"transfer-{transfer_number:06d}"
        moved_status = transfer.destination if transfer.destination in {"inactivated", "sequestered", "waiting"} else "waiting"
        moved_scheduled_time = transfer.scheduled_time if transfer.scheduled_time is not None else lot.scheduled_time
        moved_lot = replace(
            lot,
            release_id=(
                f"{lot.release_id}-{transfer_number:03d}"
                if remaining_products or transfer.destination == "free"
                else lot.release_id
            ),
            products=moved_products,
            scheduled_time=moved_scheduled_time,
            status=moved_status,
            root_release_id=lot.root_id,
            parent_release_id=lot.release_id,
        )
        new_pending = [
            item for item in self.state.pending_releases if item.release_id != lot.release_id
        ]
        if remaining_products:
            new_pending.append(replace(lot, products=remaining_products))
        free_changes: dict[str, int] = {}
        new_counts = dict(self.state.counts)
        new_waiting = dict(self.state.waiting_counts)
        new_inactivated = dict(self.state.inactivated_counts)
        new_sequestered = dict(self.state.sequestered_counts)
        for species_id, amount in moved_products.items():
            source_bucket = new_sequestered if lot.status == "sequestered" else new_waiting
            source_bucket[species_id] = source_bucket.get(species_id, 0) - amount
            if moved_status == "inactivated":
                new_inactivated[species_id] = new_inactivated.get(species_id, 0) + amount
            elif moved_status == "sequestered":
                new_sequestered[species_id] = new_sequestered.get(species_id, 0) + amount
            elif moved_status == "waiting":
                new_waiting[species_id] = new_waiting.get(species_id, 0) + amount
            elif transfer.destination == "free":
                new_counts[species_id] = new_counts.get(species_id, 0) + amount
                free_changes[species_id] = amount
        if moved_status in {"waiting", "sequestered"}:
            new_pending.append(moved_lot)
        new_pending.sort(key=lambda item: (item.scheduled_time, item.release_id))
        self.state.pending_releases = new_pending
        self.state.counts = self._complete_bucket(new_counts)
        self.state.waiting_counts = self._complete_bucket(new_waiting)
        self.state.inactivated_counts = self._complete_bucket(new_inactivated)
        self.state.sequestered_counts = self._complete_bucket(new_sequestered)
        event_type = "release" if transfer.destination == "free" else "waiting_transfer"
        event = ReactionEvent(
            time=transfer.time,
            reaction_id=lot.reaction_id,
            waiting_time=transfer.time - lot.created_time,
            propensity=0.0,
            total_propensity=0.0,
            changes=free_changes,
            reason=f"{event_type} {transfer_id} from {lot.release_id} to {transfer.destination}",
            counts_after=dict(self.state.counts),
            condition_version=lot.condition_version,
            event_id=transfer_id,
            event_type=event_type,
            release_id=lot.release_id if event_type == "waiting_transfer" else moved_lot.release_id,
            source_event_id=lot.source_event_id,
            release_condition_version=self.state.condition_version if event_type == "release" else None,
            waiting_after={k: v for k, v in new_waiting.items() if v != 0},
            pending_releases=tuple(item.to_dict() for item in self.state.pending_releases),
            destination=transfer.destination,
            inactivated_after={k: v for k, v in new_inactivated.items() if v != 0},
            sequestered_after={k: v for k, v in new_sequestered.items() if v != 0},
            gate_version=self.state.gate_version,
            root_release_id=lot.root_id,
        )
        return self._record_event(event)

    def _apply_due_waiting_transfers(self, allow_equal: bool = True) -> list[ReactionEvent]:
        events: list[ReactionEvent] = []
        while self.state.waiting_transfer_schedule:
            transfer = self.state.waiting_transfer_schedule[0]
            if transfer.time > self.state.time or (not allow_equal and transfer.time == self.state.time):
                break
            self.state.time = transfer.time
            events.append(self._apply_one_waiting_transfer(transfer))
            self.state.waiting_transfer_schedule = self.state.waiting_transfer_schedule[1:]
        return events

    def _apply_due_waiting_policies(self, allow_equal: bool = True) -> list[ReactionEvent]:
        events: list[ReactionEvent] = []
        events.extend(self._apply_due_release_gates(allow_equal=allow_equal))
        events.extend(self._apply_due_waiting_transfers(allow_equal=allow_equal))
        return events

    def _apply_one_release(self, release: PendingRelease) -> ReactionEvent:
        if release.status != "waiting" or not self.state.release_allowed:
            raise RuntimeError(f"release {release.release_id} is not eligible for release")
        version_before_release = self.state.condition_version
        new_counts = dict(self.state.counts)
        new_waiting = dict(self.state.waiting_counts)
        for species_id, amount in release.products.items():
            if new_waiting.get(species_id, 0) < amount:
                raise RuntimeError(f"waiting inventory is missing {species_id} for {release.release_id}")
            new_counts[species_id] = new_counts.get(species_id, 0) + amount
            new_waiting[species_id] = new_waiting.get(species_id, 0) - amount
        release_ids = {item.release_id for item in self.state.pending_releases}
        if release.release_id not in release_ids:
            raise RuntimeError(f"release {release.release_id} was already applied")
        self.state.time = release.scheduled_time
        self.state.counts = new_counts
        complete_waiting = {species.id: 0 for species in self.model.species}
        complete_waiting.update(new_waiting)
        self.state.waiting_counts = complete_waiting
        self.state.pending_releases = [
            item for item in self.state.pending_releases if item.release_id != release.release_id
        ]
        self._apply_due_condition_changes(allow_equal=True)
        event = ReactionEvent(
            time=release.scheduled_time,
            reaction_id=release.reaction_id,
            waiting_time=release.scheduled_time - release.created_time,
            propensity=0.0,
            total_propensity=0.0,
            changes=dict(release.products),
            reason=(
                f"scheduled release {release.release_id} from event {release.source_event_id}; "
                f"release used condition_version={release.condition_version}; "
                f"condition_version_before_release={version_before_release}; condition_version_after_release={self.state.condition_version}"
            ),
            counts_after=dict(new_counts),
            condition_version=release.condition_version,
            event_id=release.release_id,
            event_type="release",
            release_id=release.release_id,
            source_event_id=release.source_event_id,
            release_condition_version=self.state.condition_version,
            waiting_after={k: v for k, v in new_waiting.items() if v != 0},
            pending_releases=self._snapshot_pending_releases(),
            inactivated_after={k: v for k, v in self.state.inactivated_counts.items() if v != 0},
            sequestered_after={k: v for k, v in self.state.sequestered_counts.items() if v != 0},
            gate_version=self.state.gate_version,
            root_release_id=release.root_id,
        )
        return self._record_event(event)

    def _apply_due_releases(self, allow_equal: bool = True) -> list[ReactionEvent]:
        release_time = self._next_release_time()
        if release_time is None:
            return []
        if release_time > self.state.time or (
            not allow_equal and release_time == self.state.time
        ):
            return []
        release = next(
            item
            for item in sorted(
                self.state.pending_releases,
                key=lambda lot: (lot.scheduled_time, lot.release_id),
            )
            if item.status == "waiting" and item.scheduled_time == release_time
        )
        return [self._apply_one_release(release)]

    def step(self) -> ReactionEvent | None:
        due_gate = self._apply_due_release_gates(allow_equal=True)
        if due_gate:
            return due_gate[0]
        due_transfer = self._apply_due_waiting_transfers(allow_equal=True)
        if due_transfer:
            return due_transfer[0]
        due_releases = self._apply_due_releases(allow_equal=True)
        if due_releases:
            return due_releases[0]
        if self.state.pending_event is not None:
            pending = self.state.pending_event
            next_change = self._next_condition_change()
            if next_change is not None and next_change.time < float(pending["time"]):
                # A condition now intervenes before the stored pre-drawn event.
                # Discard that draw and its random numbers, then redraw under
                # the new version so resume matches a continuous run.
                self.state.pending_event = None
                self._discard_pending_event(pending)
                self.state.time = next_change.time
                self._apply_due_condition_changes(allow_equal=True)
                return self.step()
            next_release = self._next_release_time()
            if next_release is not None and next_release <= float(pending["time"]):
                pending_event = self.state.pending_event
                self.state.pending_event = None
                self._discard_pending_event(pending_event)
                self.state.time = next_release
                return self.step()
            selected = self.model.reaction(str(pending["reaction_id"]))
            event_time = float(pending["time"])
            if event_time < self.state.time:
                raise RuntimeError("pending event is earlier than current trajectory time")
            self.state.pending_event = None
            version = int(pending.get("condition_version", self.state.condition_version))
            return self._apply_event(
                selected,
                event_time,
                float(pending["waiting_time"]),
                float(pending["propensity"]),
                float(pending["total_propensity"]),
                version,
                str(pending.get("reason", "mass-action event continued from saved state")),
            )

        self._apply_due_condition_changes(allow_equal=False)
        rng = self._rng()
        available, propensities, total = self._available()
        if total <= 0.0:
            self.state.rng_state = rng.getstate()
            self.state.status = "exhausted"
            return None

        waiting_time = rng.expovariate(total)
        cutoff = rng.random() * total
        cumulative = 0.0
        selected: Reaction | None = None
        for reaction in available:
            cumulative += propensities[reaction.id]
            if cutoff <= cumulative:
                selected = reaction
                break
        if selected is None:
            selected = available[-1]

        if not self.rule.is_possible(selected, self.state.counts):
            raise RuntimeError(f"rule selected impossible reaction {selected.id}")

        event = self._build_event(
            selected,
            self.state.time + waiting_time,
            waiting_time,
            propensities[selected.id],
            total,
            self.state.condition_version,
            self._event_reason(propensities[selected.id], total),
        )
        self.state.rng_state = rng.getstate()
        applied = self._apply_built_event(event)
        self._apply_due_condition_changes(allow_equal=True)
        return applied

    def _build_event(
        self,
        selected: Reaction,
        event_time: float,
        waiting_time: float,
        propensity: float,
        total_propensity: float,
        condition_version: int,
        reason: str,
    ) -> ReactionEvent:
        if not self.rule.is_possible(selected, self.state.counts):
            raise RuntimeError(f"rule selected impossible reaction {selected.id}")
        new_counts = dict(self.state.counts)
        changes: dict[str, int] = {}
        for species_id, amount in selected.reactants.items():
            changes[species_id] = changes.get(species_id, 0) - amount
        for species_id, amount in selected.products.items():
            if species_id in selected.product_delays:
                continue
            changes[species_id] = changes.get(species_id, 0) + amount
        for species_id, change in changes.items():
            new_counts[species_id] = new_counts.get(species_id, 0) + change
        new_waiting = dict(self.state.waiting_counts)
        planned_releases: list[PendingRelease] = []
        release_ordinal = 0
        for species_id in sorted(selected.product_delays):
            amount = int(selected.products[species_id])
            delay = float(selected.product_delays[species_id])
            new_waiting[species_id] = new_waiting.get(species_id, 0) + amount
            release_ordinal += 1
            release_id = f"release-{self.state.next_release_number + release_ordinal - 1:06d}"
            source_event_id = f"event-{self.state.next_event_number:06d}"
            planned_releases.append(
                PendingRelease(
                    release_id=release_id,
                    source_event_id=source_event_id,
                    reaction_id=selected.id,
                    products={species_id: amount},
                    created_time=event_time,
                    scheduled_time=event_time + delay,
                    condition_version=condition_version,
                )
            )
        release_ids = [release.release_id for release in planned_releases]
        release_text = f"; scheduled delayed products: {', '.join(release_ids)}" if release_ids else ""
        return ReactionEvent(
            time=event_time,
            reaction_id=selected.id,
            waiting_time=waiting_time,
            propensity=propensity,
            total_propensity=total_propensity,
            changes={k: v for k, v in changes.items() if v != 0},
            reason=f"{reason}{release_text}",
            counts_after=new_counts,
            condition_version=condition_version,
            event_id=f"event-{self.state.next_event_number:06d}",
            waiting_after={k: v for k, v in new_waiting.items() if v != 0},
            pending_releases=tuple(planned_releases),
        )

    def _apply_built_event(self, event: ReactionEvent) -> ReactionEvent:
        self.state.time = event.time
        self.state.counts = dict(event.counts_after)
        complete_waiting = {species.id: 0 for species in self.model.species}
        complete_waiting.update(event.waiting_after)
        self.state.waiting_counts = complete_waiting
        for release_data in event.pending_releases:
            if isinstance(release_data, PendingRelease):
                release = release_data
            else:
                release = PendingRelease.from_dict(dict(release_data))
            if any(item.release_id == release.release_id for item in self.state.pending_releases):
                raise RuntimeError(f"duplicate pending release {release.release_id}")
            self.state.pending_releases.append(release)
        self.state.pending_releases.sort(key=lambda release: (release.scheduled_time, release.release_id))
        if event.event_type == "reaction":
            self.state.next_event_number += 1
            self.state.next_release_number += sum(
                1
                for release in event.pending_releases
                if (release if isinstance(release, PendingRelease) else PendingRelease.from_dict(dict(release))).products
            )
        self.state.events.append(event)
        self.state.status = "running"
        return event

    def _apply_event(
        self,
        selected: Reaction,
        event_time: float,
        waiting_time: float,
        propensity: float,
        total_propensity: float,
        condition_version: int,
        reason: str,
    ) -> ReactionEvent:
        if "; scheduled delayed products:" in reason:
            reason = reason.split("; scheduled delayed products:", 1)[0]
        event = self._build_event(
            selected,
            event_time,
            waiting_time,
            propensity,
            total_propensity,
            condition_version,
            reason,
        )
        applied = self._apply_built_event(event)
        self._apply_due_condition_changes(allow_equal=True)
        return applied

    def _draw_event(self, rng: random.Random) -> ReactionEvent | None:
        """Draw the next event from the current state without committing it."""
        available, propensities, total = self._available()
        if total <= 0.0:
            return None
        waiting_time = rng.expovariate(total)
        cutoff = rng.random() * total
        cumulative = 0.0
        selected = available[-1]
        for candidate in available:
            cumulative += propensities[candidate.id]
            if cutoff <= cumulative:
                selected = candidate
                break
        return self._build_event(
            selected,
            self.state.time + waiting_time,
            waiting_time,
            propensities[selected.id],
            total,
            self.state.condition_version,
            (
                self._event_reason(propensities[selected.id], total)
            ),
        )

    def run(
        self,
        end_time: float | None = None,
        max_events: int | None = None,
    ) -> TrajectoryState:
        if end_time is not None and end_time < self.state.time:
            raise ValueError("end_time precedes current trajectory time")
        events_used = 0
        try:
            while max_events is None or events_used < max_events:
                policy_events = self._apply_due_waiting_policies(allow_equal=True)
                if policy_events:
                    while True:
                        due_release = self._apply_due_releases(allow_equal=True)
                        if not due_release:
                            break
                        events_used += len(due_release)
                    self._apply_due_condition_changes(allow_equal=True)
                    events_used += len(policy_events)
                    continue
                release = self._apply_due_releases(allow_equal=True)
                if release:
                    events_used += 1
                    continue
                next_change = self._next_condition_change()
                next_release = self._next_release_time()
                next_policy = self._policy_time()

                if self.state.pending_event is not None:
                    pending = self.state.pending_event
                    pending_time = float(pending["time"])
                    release_first = self._release_precedes_change(
                        next_release, next_change.time if next_change else None
                    )
                    if release_first and next_release <= pending_time:
                        self.state.pending_event = None
                        self._discard_pending_event(pending)
                        if end_time is not None and end_time < next_release:
                            self.state.time = end_time
                            self.state.status = "time_limit"
                            break
                        self.state.time = next_release
                        continue
                    if next_change is not None and next_change.time < pending_time:
                        self.state.pending_event = None
                        self._discard_pending_event(pending)
                        if end_time is not None and end_time <= next_change.time:
                            self.state.time = end_time
                            self._apply_due_condition_changes(allow_equal=True)
                            self.state.status = "time_limit"
                            break
                        self.state.time = next_change.time
                        self._apply_due_condition_changes(allow_equal=True)
                        continue
                    if next_policy is not None and next_policy <= pending_time:
                        if end_time is not None and end_time <= next_policy:
                            self.state.time = end_time
                            self.state.status = "time_limit"
                            break
                        self.state.time = next_policy
                        continue
                    if next_release is not None and next_release <= pending_time:
                        self.state.pending_event = None
                        self._discard_pending_event(pending)
                        if end_time is not None and end_time < next_release:
                            self.state.time = end_time
                            self.state.status = "time_limit"
                            break
                        self.state.time = next_release
                        continue
                    if end_time is not None and pending_time > end_time:
                        self.state.time = end_time
                        self.state.status = "time_limit"
                        break
                    self.step()
                    events_used += 1
                    continue

                self._apply_due_condition_changes(allow_equal=False)
                next_change = self._next_condition_change()
                next_policy = self._policy_time()
                if end_time is not None and self.state.time >= end_time:
                    self.state.status = "time_limit"
                    break

                rng_state_before = self.state.rng_state
                rng = self._rng()
                available, propensities, total = self._available()
                if total <= 0.0:
                    if next_release is not None and self._release_precedes_change(
                        next_release, next_change.time if next_change else None
                    ):
                        if end_time is not None and end_time < next_release:
                            self.state.time = end_time
                            self.state.status = "time_limit"
                            break
                        self.state.time = next_release
                        continue
                    if next_change is not None:
                        if end_time is not None and end_time < next_change.time:
                            self.state.time = end_time
                            self.state.status = "time_limit"
                            break
                        self.state.time = next_change.time
                        self._apply_due_condition_changes(allow_equal=True)
                        continue
                    if next_policy is not None:
                        if end_time is not None and end_time < next_policy:
                            self.state.time = end_time
                            self.state.status = "time_limit"
                            break
                        self.state.time = next_policy
                        continue
                    self.state.status = "exhausted" if not self.state.pending_releases else "time_limit"
                    break

                waiting_time = rng.expovariate(total)
                cutoff = rng.random() * total
                cumulative = 0.0
                selected = available[-1]
                for candidate in available:
                    cumulative += propensities[candidate.id]
                    if cutoff <= cumulative:
                        selected = candidate
                        break
                event = self._build_event(
                    selected,
                    self.state.time + waiting_time,
                    waiting_time,
                    propensities[selected.id],
                    total,
                    self.state.condition_version,
                    self._event_reason(propensities[selected.id], total),
                )

                if next_release is not None and next_release <= event.time and self._release_precedes_change(
                    next_release, next_change.time if next_change else None
                ):
                    if end_time is not None and end_time < next_release:
                        self.state.time = end_time
                        self.state.status = "time_limit"
                        break
                    self.state.time = next_release
                    continue
                if (
                    next_change is not None
                    and next_change.time < event.time
                    and (end_time is None or next_change.time <= end_time)
                ):
                    self.state.time = next_change.time
                    self._apply_due_condition_changes(allow_equal=True)
                    continue

                if next_policy is not None and next_policy <= event.time:
                    if end_time is not None and end_time < next_policy:
                        self.state.time = end_time
                        self.state.status = "time_limit"
                        break
                    self.state.time = next_policy
                    continue

                if end_time is not None and event.time > end_time:
                    self.state.time = end_time
                    self.state.pending_event = {
                        "time": event.time,
                        "reaction_id": event.reaction_id,
                        "waiting_time": event.waiting_time,
                        "propensity": event.propensity,
                        "total_propensity": event.total_propensity,
                        "reason": event.reason,
                        "condition_version": event.condition_version,
                        "rng_state_before": self._serialize_rng_state(rng_state_before),
                    }
                    self.state.rng_state = rng.getstate()
                    self.state.status = "time_limit"
                    break

                self.state.rng_state = rng.getstate()
                self._apply_built_event(event)
                self._apply_due_condition_changes(allow_equal=True)
                events_used += 1
        except Exception as exc:
            self.state.status = "failed"
            self.state.failure_reason = f"{type(exc).__name__}: {exc}"
            raise
        return self.state

    @staticmethod
    def _serialize_rng_state(state: tuple[Any, ...]) -> list[Any]:
        version, values, gauss = state
        return [int(version), [int(value) for value in values], gauss]

    def fork(self) -> "StochasticRunner":
        """Create an independent trajectory branch from this checkpoint."""

        return StochasticRunner.from_state(self.state.fork(), rule=self.rule)

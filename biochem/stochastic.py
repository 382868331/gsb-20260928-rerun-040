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
    fate_type: str | None = None
    fate_id: str | None = None
    sequester_id: str | None = None
    origin_release_id: str | None = None
    deactivated_after: dict[str, int] = field(default_factory=dict)
    sequestered_after: dict[str, int] = field(default_factory=dict)
    blocked_after: tuple[str, ...] = ()

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
            fate_type=data.get("fate_type"),
            fate_id=data.get("fate_id"),
            sequester_id=data.get("sequester_id"),
            origin_release_id=data.get("origin_release_id"),
            deactivated_after={
                str(k): int(v) for k, v in data.get("deactivated_after", {}).items()
            },
            sequestered_after={
                str(k): int(v) for k, v in data.get("sequestered_after", {}).items()
            },
            blocked_after=tuple(str(item) for item in data.get("blocked_after", [])),
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
    origin_release_id: str | None = None
    origin_fate: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "release_id": self.release_id,
            "source_event_id": self.source_event_id,
            "reaction_id": self.reaction_id,
            "products": dict(self.products),
            "created_time": self.created_time,
            "scheduled_time": self.scheduled_time,
            "condition_version": self.condition_version,
            "origin_release_id": self.origin_release_id,
            "origin_fate": self.origin_fate,
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
            origin_release_id=data.get("origin_release_id"),
            origin_fate=data.get("origin_fate"),
        )


@dataclass(frozen=True)
class SequesteredLot:
    """A waiting batch moved away from the release track by an isolate fate."""

    sequester_id: str
    release_id: str
    source_event_id: str
    reaction_id: str
    products: dict[str, int]
    created_time: float
    scheduled_time: float
    condition_version: int
    sequestered_time: float
    fate_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequester_id": self.sequester_id,
            "release_id": self.release_id,
            "source_event_id": self.source_event_id,
            "reaction_id": self.reaction_id,
            "products": dict(self.products),
            "created_time": self.created_time,
            "scheduled_time": self.scheduled_time,
            "condition_version": self.condition_version,
            "sequestered_time": self.sequestered_time,
            "fate_id": self.fate_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SequesteredLot":
        return cls(
            sequester_id=str(data["sequester_id"]),
            release_id=str(data["release_id"]),
            source_event_id=str(data["source_event_id"]),
            reaction_id=str(data["reaction_id"]),
            products={str(k): int(v) for k, v in data["products"].items()},
            created_time=float(data["created_time"]),
            scheduled_time=float(data["scheduled_time"]),
            condition_version=int(data["condition_version"]),
            sequestered_time=float(data["sequestered_time"]),
            fate_id=str(data["fate_id"]),
        )


FATE_INACTIVATE = "inactivate"
FATE_SEQUESTER = "sequester"
FATE_RELEASE_SEQUESTERED = "release_sequestered"
FATE_BLOCK_RELEASES = "block_releases"
FATE_ALLOW_RELEASES = "allow_releases"
FATE_TYPES = (
    FATE_INACTIVATE,
    FATE_SEQUESTER,
    FATE_RELEASE_SEQUESTERED,
    FATE_BLOCK_RELEASES,
    FATE_ALLOW_RELEASES,
)


@dataclass(frozen=True)
class WaitingFate:
    """A scheduled, replayable competing destination for waiting intermediates.

    ``amount`` of None means "every currently waiting unit of each named
    species"; an integer applies only to one identified pending release.
    ``products`` constrains the fate to specific species (empty means every
    waiting species for the global fates).
    """

    time: float
    fate_type: str
    release_id: str | None = None
    sequester_id: str | None = None
    products: dict[str, int] | None = None
    amount: int | None = None
    label: str = ""
    fate_id: str | None = None
    version: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "fate_type": self.fate_type,
            "release_id": self.release_id,
            "sequester_id": self.sequester_id,
            "products": None if self.products is None else dict(self.products),
            "amount": self.amount,
            "label": self.label,
            "fate_id": self.fate_id,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WaitingFate":
        return cls(
            time=float(data["time"]),
            fate_type=str(data["fate_type"]),
            release_id=data.get("release_id"),
            sequester_id=data.get("sequester_id"),
            products=(
                None
                if data.get("products") is None
                else {str(k): int(v) for k, v in data["products"].items()}
            ),
            amount=(None if data.get("amount") is None else int(data["amount"])),
            label=str(data.get("label", "")),
            fate_id=data.get("fate_id"),
            version=data.get("version"),
        )


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
    next_event_number: int = 1
    next_release_number: int = 1
    fate_schedule: list[WaitingFate] = field(default_factory=list)
    fate_events: list[WaitingFate] = field(default_factory=list)
    blocked_release_ids: set[str] = field(default_factory=set)
    releases_blocked: bool = False
    deactivated_counts: dict[str, int] = field(default_factory=dict)
    sequestered_counts: dict[str, int] = field(default_factory=dict)
    sequestered_lots: list[SequesteredLot] = field(default_factory=list)
    next_fate_number: int = 1
    next_sequester_number: int = 1
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        complete_counts = {species.id: 0 for species in self.model.species}
        complete_counts.update(self.counts)
        self.counts = complete_counts
        self.waiting_counts = {
            species.id: int(self.waiting_counts.get(species.id, 0))
            for species in self.model.species
        }
        self.deactivated_counts = {
            species.id: int(self.deactivated_counts.get(species.id, 0))
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
        for species_id, count in self.deactivated_counts.items():
            if count < 0:
                raise ValueError(f"deactivated count cannot be negative: {species_id}")
        for species_id, count in self.sequestered_counts.items():
            if count < 0:
                raise ValueError(f"sequestered count cannot be negative: {species_id}")
        self.condition_schedule.sort(key=lambda change: change.time)
        self.fate_schedule.sort(key=lambda fate: fate.time)
        self.pending_releases.sort(key=lambda release: (release.scheduled_time, release.release_id))
        reaction_numbers = [
            int(event.event_id.removeprefix("event-"))
            for event in self.events
            if event.event_type == "reaction"
            and event.event_id
            and event.event_id.startswith("event-")
        ]
        release_numbers = [
            int(item.release_id.removeprefix("release-"))
            for item in self.pending_releases
            if item.release_id.startswith("release-")
        ]
        release_numbers.extend(
            int(event.release_id.removeprefix("release-"))
            for event in self.events
            if event.event_type == "release"
            and event.release_id
            and event.release_id.startswith("release-")
        )
        if reaction_numbers:
            self.next_event_number = max(self.next_event_number, max(reaction_numbers) + 1)
        if release_numbers:
            self.next_release_number = max(self.next_release_number, max(release_numbers) + 1)

        def _ordinal(prefix: str, value: str | None) -> int | None:
            if value and value.startswith(prefix):
                try:
                    return int(value.removeprefix(prefix))
                except ValueError:
                    return None
            return None

        fate_numbers = [
            ordinal
            for event in self.events
            for ordinal in [_ordinal("fate-", event.fate_id)]
            if ordinal is not None
        ]
        sequester_numbers = [
            ordinal
            for event in self.events
            for ordinal in [_ordinal("sequester-", event.sequester_id)]
            if ordinal is not None
        ]
        sequester_numbers.extend(
            ordinal
            for lot in self.sequestered_lots
            for ordinal in [_ordinal("sequester-", lot.sequester_id)]
            if ordinal is not None
        )
        if fate_numbers:
            self.next_fate_number = max(self.next_fate_number, max(fate_numbers) + 1)
        if sequester_numbers:
            self.next_sequester_number = max(self.next_sequester_number, max(sequester_numbers) + 1)

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
            "next_event_number": self.next_event_number,
            "next_release_number": self.next_release_number,
            "fate_schedule": [fate.to_dict() for fate in self.fate_schedule],
            "fate_events": [event.to_dict() for event in self.fate_events],
            "blocked_release_ids": sorted(self.blocked_release_ids),
            "releases_blocked": self.releases_blocked,
            "deactivated_counts": dict(self.deactivated_counts),
            "sequestered_counts": dict(self.sequestered_counts),
            "sequestered_lots": [lot.to_dict() for lot in self.sequestered_lots],
            "next_fate_number": self.next_fate_number,
            "next_sequester_number": self.next_sequester_number,
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
            next_event_number=int(data.get("next_event_number", 1)),
            next_release_number=int(data.get("next_release_number", 1)),
            fate_schedule=[
                WaitingFate.from_dict(item) for item in data.get("fate_schedule", [])
            ],
            fate_events=[
                WaitingFate.from_dict(item) for item in data.get("fate_events", [])
            ],
            blocked_release_ids={
                str(item) for item in data.get("blocked_release_ids", [])
            },
            releases_blocked=bool(data.get("releases_blocked", False)),
            deactivated_counts={
                str(k): int(v) for k, v in data.get("deactivated_counts", {}).items()
            },
            sequestered_counts={
                str(k): int(v) for k, v in data.get("sequestered_counts", {}).items()
            },
            sequestered_lots=[
                SequesteredLot.from_dict(item)
                for item in data.get("sequestered_lots", [])
            ],
            next_fate_number=int(data.get("next_fate_number", 1)),
            next_sequester_number=int(data.get("next_sequester_number", 1)),
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

    def schedule_waiting_fate(
        self,
        time: float,
        fate_type: str,
        release_id: str | None = None,
        sequester_id: str | None = None,
        products: dict[str, int] | None = None,
        amount: int | None = None,
        label: str = "",
    ) -> WaitingFate:
        """Schedule a competing destination for waiting intermediates.

        ``inactivate``/``sequester`` target one ``release_id`` batch or every
        waiting batch (optionally restricted to ``products`` species);
        ``amount`` of None takes the whole remaining quantity, otherwise the
        batch is split and the untouched remainder stays on its release track.
        ``release_sequestered`` returns isolated material (identified by
        ``sequester_id``, or all lots) to a fresh pending release at ``time``.
        ``block_releases``/``allow_releases`` toggle the release gate, either
        globally or for one ``release_id``; blocking is not destruction.
        """
        if time < self.state.time:
            raise ValueError("fate time precedes current trajectory time")
        if fate_type not in FATE_TYPES:
            raise ValueError(f"unknown waiting fate type: {fate_type}")
        if amount is not None and amount <= 0:
            raise ValueError("fate amount must be positive or None")
        normalized_products = None
        if products is not None:
            normalized_products = {}
            for species_id, value in products.items():
                if species_id not in self.model.species_ids:
                    raise ValueError(f"fate references unknown species {species_id}")
                value = int(value)
                if value <= 0:
                    raise ValueError("fate product amounts must be positive")
                normalized_products[species_id] = value
        if fate_type in (FATE_INACTIVATE, FATE_SEQUESTER):
            if sequester_id is not None:
                raise ValueError(f"{fate_type} cannot target a sequestered lot")
        elif fate_type == FATE_RELEASE_SEQUESTERED:
            if release_id is not None:
                raise ValueError("release_sequestered targets a sequester_id")
        else:  # gate toggles
            if amount is not None or normalized_products is not None:
                raise ValueError(f"{fate_type} only toggles the release gate")
        fate = WaitingFate(
            time=float(time),
            fate_type=fate_type,
            release_id=release_id,
            sequester_id=sequester_id,
            products=normalized_products,
            amount=amount,
            label=label,
        )
        self.state.fate_schedule.append(fate)
        self.state.fate_schedule.sort(key=lambda item: item.time)
        return fate

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

    def _next_due_fate(self) -> WaitingFate | None:
        return self.state.fate_schedule[0] if self.state.fate_schedule else None

    def _release_is_blocked(self, release: PendingRelease) -> bool:
        return self.state.releases_blocked or (
            release.release_id in self.state.blocked_release_ids
        )

    def _next_open_release(self) -> PendingRelease | None:
        for release in self.state.pending_releases:
            if not self._release_is_blocked(release):
                return release
        return None

    def _next_open_release_time(self) -> float | None:
        release = self._next_open_release()
        if release is None:
            return None
        # A batch still blocked at its scheduled time waits for the gate to
        # re-open; it never releases in the past.
        return max(release.scheduled_time, self.state.time)

    def _prospective_condition_version(self) -> int:
        """Version that will be current once all changes at this instant apply."""
        version = self.state.condition_version
        pending_at_now = {
            change.time
            for change in self.state.condition_schedule
            if change.time == self.state.time
        }
        return version + len(pending_at_now)

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

    def _apply_one_release(self, release: PendingRelease) -> ReactionEvent:
        version_before_release = self.state.condition_version
        release_condition_version = self._prospective_condition_version()
        effective_time = max(release.scheduled_time, self.state.time)
        new_counts = dict(self.state.counts)
        new_waiting = dict(self.state.waiting_counts)
        for species_id, amount in release.products.items():
            if new_waiting.get(species_id, 0) < amount:
                raise RuntimeError(f"waiting inventory is missing {species_id} for {release.release_id}")
            new_counts[species_id] = new_counts.get(species_id, 0) + amount
            new_waiting[species_id] = new_waiting.get(species_id, 0) - amount
        self.state.time = effective_time
        self.state.counts = new_counts
        complete_waiting = {species.id: 0 for species in self.model.species}
        complete_waiting.update(new_waiting)
        self.state.waiting_counts = complete_waiting
        self.state.pending_releases = [
            item for item in self.state.pending_releases if item.release_id != release.release_id
        ]
        self.state.blocked_release_ids.discard(release.release_id)
        origin_text = (
            f"; origin_release_id={release.origin_release_id} via {release.origin_fate}"
            if release.origin_release_id is not None
            else ""
        )
        event = ReactionEvent(
            time=effective_time,
            reaction_id=release.reaction_id,
            waiting_time=effective_time - release.created_time,
            propensity=0.0,
            total_propensity=0.0,
            changes=dict(release.products),
            reason=(
                f"scheduled release {release.release_id} from event {release.source_event_id}; "
                f"release used condition_version={release.condition_version}; "
                f"condition_version_before_release={version_before_release}; "
                f"condition_version_after_release={release_condition_version}{origin_text}"
            ),
            counts_after=dict(new_counts),
            condition_version=release.condition_version,
            event_id=release.release_id,
            event_type="release",
            release_id=release.release_id,
            source_event_id=release.source_event_id,
            release_condition_version=release_condition_version,
            origin_release_id=release.origin_release_id,
            waiting_after={k: v for k, v in new_waiting.items() if v != 0},
            deactivated_after={k: v for k, v in self.state.deactivated_counts.items() if v != 0},
            sequestered_after={k: v for k, v in self.state.sequestered_counts.items() if v != 0},
            blocked_after=tuple(sorted(self.state.blocked_release_ids)),
            pending_releases=self._snapshot_pending_releases(),
        )
        self.state.events.append(event)
        self.state.status = "running"
        return event

    def _target_products(
        self, release: PendingRelease, products: dict[str, int] | None
    ) -> dict[str, int]:
        if products is None:
            return dict(release.products)
        return {
            species_id: min(amount, release.products.get(species_id, 0))
            for species_id, amount in products.items()
            if release.products.get(species_id, 0) > 0
        }

    def _record_fate_event(
        self,
        fate: WaitingFate,
        fate_id: str,
        changes: dict[str, int] | None = None,
        release_id: str | None = None,
        source_event_id: str | None = None,
        reaction_id: str = "",
        sequester_id: str | None = None,
        origin_release_id: str | None = None,
    ) -> ReactionEvent:
        event = ReactionEvent(
            time=fate.time,
            reaction_id=reaction_id,
            waiting_time=0.0,
            propensity=0.0,
            total_propensity=0.0,
            changes=changes or {},
            reason=(
                f"waiting fate {fate_id} type={fate.fate_type}"
                + (f" release_id={release_id}" if release_id else "")
                + (f" sequester_id={sequester_id}" if sequester_id else "")
                + (f" label={fate.label}" if fate.label else "")
                + f"; resolved under condition_version={self.state.condition_version}"
            ),
            counts_after=dict(self.state.counts),
            condition_version=self.state.condition_version,
            event_id=fate_id,
            event_type=fate.fate_type,
            release_id=release_id,
            source_event_id=source_event_id,
            fate_type=fate.fate_type,
            fate_id=fate_id,
            sequester_id=sequester_id,
            origin_release_id=origin_release_id,
            waiting_after={k: v for k, v in self.state.waiting_counts.items() if v != 0},
            deactivated_after={k: v for k, v in self.state.deactivated_counts.items() if v != 0},
            sequestered_after={k: v for k, v in self.state.sequestered_counts.items() if v != 0},
            blocked_after=tuple(sorted(self.state.blocked_release_ids)),
        )
        self.state.events.append(event)
        return event

    def _move_out_of_waiting(
        self, release: PendingRelease, moved: dict[str, int]
    ) -> None:
        """Remove ``moved`` units from one pending batch and the waiting ledger."""
        new_waiting = dict(self.state.waiting_counts)
        remainder = dict(release.products)
        for species_id, amount in moved.items():
            available = new_waiting.get(species_id, 0)
            if amount <= 0 or amount > available or amount > remainder.get(species_id, 0):
                raise RuntimeError(
                    f"fate requests {amount} of {species_id} but only "
                    f"{remainder.get(species_id, 0)} remain in {release.release_id}"
                )
            new_waiting[species_id] = available - amount
            remainder[species_id] -= amount
        complete_waiting = {species.id: 0 for species in self.model.species}
        complete_waiting.update(new_waiting)
        self.state.waiting_counts = complete_waiting
        if any(value > 0 for value in remainder.values()):
            updated = replace(release, products={k: v for k, v in remainder.items() if v > 0})
            index = next(
                idx
                for idx, item in enumerate(self.state.pending_releases)
                if item.release_id == release.release_id
            )
            self.state.pending_releases[index] = updated
        else:
            self.state.pending_releases = [
                item
                for item in self.state.pending_releases
                if item.release_id != release.release_id
            ]
            self.state.blocked_release_ids.discard(release.release_id)

    def _apply_fate(self, fate: WaitingFate) -> list[ReactionEvent]:
        fate_id = f"fate-{self.state.next_fate_number:06d}"
        self.state.next_fate_number += 1
        events: list[ReactionEvent] = []
        self.state.time = fate.time

        if fate.fate_type in (FATE_INACTIVATE, FATE_SEQUESTER):
            if fate.release_id is not None:
                target = next(
                    (
                        item
                        for item in self.state.pending_releases
                        if item.release_id == fate.release_id
                    ),
                    None,
                )
                if target is None:
                    raise RuntimeError(
                        f"fate {fate_id} targets unknown or gone release {fate.release_id}"
                    )
                targets = [target]
            else:
                targets = list(self.state.pending_releases)
            ledger = (
                self.state.deactivated_counts
                if fate.fate_type == FATE_INACTIVATE
                else self.state.sequestered_counts
            )
            for release in targets:
                wanted = self._target_products(release, fate.products)
                if fate.amount is not None and fate.release_id is not None:
                    species_ids = list(wanted)
                    if len(species_ids) != 1:
                        raise RuntimeError(
                            "a partial fate amount needs exactly one species in one batch"
                        )
                    species_id = species_ids[0]
                    if fate.amount > wanted[species_id]:
                        raise RuntimeError(
                            f"fate requests {fate.amount} of {species_id} from "
                            f"{release.release_id} but only {wanted[species_id]} wait"
                        )
                    wanted = {species_id: fate.amount}
                moved = {k: v for k, v in wanted.items() if v > 0}
                if not moved:
                    continue
                self._move_out_of_waiting(release, moved)
                for species_id, amount in moved.items():
                    ledger[species_id] = ledger.get(species_id, 0) + amount
                if fate.fate_type == FATE_SEQUESTER:
                    sequester_id = f"sequester-{self.state.next_sequester_number:06d}"
                    self.state.next_sequester_number += 1
                    self.state.sequestered_lots.append(
                        SequesteredLot(
                            sequester_id=sequester_id,
                            release_id=release.release_id,
                            source_event_id=release.source_event_id,
                            reaction_id=release.reaction_id,
                            products=dict(moved),
                            created_time=release.created_time,
                            scheduled_time=release.scheduled_time,
                            condition_version=release.condition_version,
                            sequestered_time=fate.time,
                            fate_id=fate_id,
                        )
                    )
                    self.state.sequestered_lots.sort(key=lambda lot: lot.sequester_id)
                else:
                    sequester_id = None
                events.append(
                    self._record_fate_event(
                        fate,
                        fate_id,
                        changes=dict(moved),
                        release_id=release.release_id,
                        source_event_id=release.source_event_id,
                        reaction_id=release.reaction_id,
                        sequester_id=sequester_id,
                    )
                )
        elif fate.fate_type == FATE_RELEASE_SEQUESTERED:
            if fate.sequester_id is not None:
                lots = [
                    lot
                    for lot in self.state.sequestered_lots
                    if lot.sequester_id == fate.sequester_id
                ]
                if not lots:
                    raise RuntimeError(
                        f"fate {fate_id} targets unknown or returned lot {fate.sequester_id}"
                    )
            else:
                lots = list(self.state.sequestered_lots)
            for lot in lots:
                moved = self._target_products(
                    PendingRelease(
                        release_id=lot.release_id,
                        source_event_id=lot.source_event_id,
                        reaction_id=lot.reaction_id,
                        products=dict(lot.products),
                        created_time=lot.created_time,
                        scheduled_time=lot.scheduled_time,
                        condition_version=lot.condition_version,
                    ),
                    fate.products,
                )
                if fate.amount is not None:
                    species_ids = list(moved)
                    if len(species_ids) != 1:
                        raise RuntimeError(
                            "a partial sequester release needs exactly one species"
                        )
                    species_id = species_ids[0]
                    if fate.amount > moved[species_id]:
                        raise RuntimeError(
                            f"fate requests {fate.amount} of {species_id} from "
                            f"{lot.sequester_id} but only {moved[species_id]} are isolated"
                        )
                    moved = {species_id: fate.amount}
                moved = {k: v for k, v in moved.items() if v > 0}
                if not moved:
                    continue
                remainder = dict(lot.products)
                for species_id, amount in moved.items():
                    if amount > self.state.sequestered_counts.get(species_id, 0):
                        raise RuntimeError(
                            f"sequestered inventory missing {species_id} for {lot.sequester_id}"
                        )
                    self.state.sequestered_counts[species_id] -= amount
                    self.state.waiting_counts[species_id] = (
                        self.state.waiting_counts.get(species_id, 0) + amount
                    )
                    remainder[species_id] -= amount
                new_release_id = f"release-{self.state.next_release_number:06d}"
                self.state.next_release_number += 1
                self.state.pending_releases.append(
                    PendingRelease(
                        release_id=new_release_id,
                        source_event_id=lot.source_event_id,
                        reaction_id=lot.reaction_id,
                        products=dict(moved),
                        created_time=lot.created_time,
                        scheduled_time=fate.time,
                        condition_version=lot.condition_version,
                        origin_release_id=lot.release_id,
                        origin_fate=FATE_RELEASE_SEQUESTERED,
                    )
                )
                self.state.pending_releases.sort(
                    key=lambda item: (item.scheduled_time, item.release_id)
                )
                if any(value > 0 for value in remainder.values()):
                    updated_lot = replace(
                        lot,
                        products={k: v for k, v in remainder.items() if v > 0},
                    )
                    self.state.sequestered_lots = [
                        updated_lot if item.sequester_id == lot.sequester_id else item
                        for item in self.state.sequestered_lots
                    ]
                else:
                    self.state.sequestered_lots = [
                        item
                        for item in self.state.sequestered_lots
                        if item.sequester_id != lot.sequester_id
                    ]
                events.append(
                    self._record_fate_event(
                        fate,
                        fate_id,
                        changes=dict(moved),
                        release_id=new_release_id,
                        source_event_id=lot.source_event_id,
                        reaction_id=lot.reaction_id,
                        sequester_id=lot.sequester_id,
                        origin_release_id=lot.release_id,
                    )
                )
        else:  # gate toggles
            blocking = fate.fate_type == FATE_BLOCK_RELEASES
            if fate.release_id is None:
                self.state.releases_blocked = blocking
                if blocking:
                    # The global flag blocks every batch; the id set keeps
                    # only explicitly per-id blocks across a global toggle.
                    pass
            else:
                if blocking:
                    if not any(
                        item.release_id == fate.release_id
                        for item in self.state.pending_releases
                    ):
                        raise RuntimeError(
                            f"fate {fate_id} blocks unknown or gone release {fate.release_id}"
                        )
                    self.state.blocked_release_ids.add(fate.release_id)
                else:
                    self.state.blocked_release_ids.discard(fate.release_id)
            source = next(
                (
                    item
                    for item in self.state.pending_releases
                    if item.release_id == fate.release_id
                ),
                None,
            )
            events.append(
                self._record_fate_event(
                    fate,
                    fate_id,
                    release_id=fate.release_id,
                    source_event_id=source.source_event_id if source else None,
                    reaction_id=source.reaction_id if source else "",
                )
            )

        applied = replace(fate, fate_id=fate_id, version=self.state.condition_version)
        self.state.fate_events.append(applied)
        self.state.fate_schedule = [
            item for item in self.state.fate_schedule if item is not fate
        ]
        return events

    def _pump_due(self, allow_equal: bool) -> list[ReactionEvent]:
        """Resolve every deterministic item due at the current time.

        Ordering at one instant: open releases (stable release_id order), then
        competing fates in declaration order, then condition changes. A block
        declared at the release instant therefore cannot recall that release,
        and equal-instant fates resolve under the pre-change condition version.
        """
        produced: list[ReactionEvent] = []
        progressed = True
        while progressed:
            progressed = False
            release = self._next_open_release()
            if release is not None and (
                release.scheduled_time < self.state.time
                or (allow_equal and release.scheduled_time == self.state.time)
            ):
                produced.append(self._apply_one_release(release))
                progressed = True
                continue
            fate = self._next_due_fate()
            if fate is not None and (
                fate.time < self.state.time
                or (allow_equal and fate.time == self.state.time)
            ):
                produced.extend(self._apply_fate(fate))
                progressed = True
                continue
            changes = [
                change
                for change in self.state.condition_schedule
                if change.time < self.state.time
                or (allow_equal and change.time == self.state.time)
            ]
            if changes:
                self._apply_due_condition_changes(allow_equal=allow_equal)
                progressed = True
        return produced

    def step(self) -> ReactionEvent | None:
        """Advance the trajectory by exactly one logical event.

        Deterministic releases/fates/condition changes due at the current
        instant are resolved before any new stochastic draw; advancing to an
        intervening deterministic item commits that item as the step result.
        """
        due = self._pump_due(allow_equal=False)
        if due:
            return due[0]

        while True:
            due = self._pump_due(allow_equal=True)
            if due:
                return due[0]

            next_change = self._next_condition_change()
            next_fate = self._next_due_fate()
            next_open_release = self._next_open_release_time()

            def intervention_before(commit_time: float, strict_change: bool) -> float | None:
                candidates: list[float] = []
                if next_open_release is not None and next_open_release <= commit_time:
                    candidates.append(next_open_release)
                if next_fate is not None and next_fate.time <= commit_time:
                    candidates.append(next_fate.time)
                if next_change is not None and (
                    next_change.time < commit_time
                    or (not strict_change and next_change.time == commit_time)
                ):
                    candidates.append(next_change.time)
                return min(candidates) if candidates else None

            if self.state.pending_event is not None:
                pending = self.state.pending_event
                pending_time = float(pending["time"])
                jump = intervention_before(pending_time, strict_change=False)
                if jump is not None:
                    self.state.pending_event = None
                    self._discard_pending_event(pending)
                    self.state.time = jump
                    continue
                selected = self.model.reaction(str(pending["reaction_id"]))
                event_time = pending_time
                if event_time < self.state.time:
                    raise RuntimeError("pending event is earlier than current trajectory time")
                self.state.pending_event = None
                return self._apply_event(
                    selected,
                    event_time,
                    float(pending["waiting_time"]),
                    float(pending["propensity"]),
                    float(pending["total_propensity"]),
                    int(pending.get("condition_version", self.state.condition_version)),
                    str(pending.get("reason", "mass-action event continued from saved state")),
                )

            rng = self._rng()
            available, propensities, total = self._available()
            if total <= 0.0:
                self.state.rng_state = rng.getstate()
                candidates = [
                    value
                    for value in (
                        next_open_release,
                        next_fate.time if next_fate else None,
                        next_change.time if next_change else None,
                    )
                    if value is not None
                ]
                if candidates:
                    self.state.time = min(candidates)
                    continue
                self.state.status = "exhausted"
                return None

            waiting_time = rng.expovariate(total)
            cutoff = rng.random() * total
            cumulative = 0.0
            selected = available[-1]
            for reaction in available:
                cumulative += propensities[reaction.id]
                if cutoff <= cumulative:
                    selected = reaction
                    break
            if not self.rule.is_possible(selected, self.state.counts):
                raise RuntimeError(f"rule selected impossible reaction {selected.id}")
            event_time = self.state.time + waiting_time
            jump = intervention_before(event_time, strict_change=True)
            if jump is not None:
                self.state.time = jump
                continue
            event = self._build_event(
                selected,
                event_time,
                waiting_time,
                propensities[selected.id],
                total,
                self.state.condition_version,
                self._event_reason(propensities[selected.id], total),
            )
            self.state.rng_state = rng.getstate()
            return self._apply_built_event(event)
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
            deactivated_after={
                k: v for k, v in self.state.deactivated_counts.items() if v != 0
            },
            sequestered_after={
                k: v for k, v in self.state.sequestered_counts.items() if v != 0
            },
            blocked_after=tuple(sorted(self.state.blocked_release_ids)),
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
        return self._apply_built_event(event)

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
                due = self._pump_due(allow_equal=True)
                if due:
                    events_used += len(due)
                    continue
                if end_time is not None and self.state.time >= end_time:
                    self.state.status = "time_limit"
                    break

                next_change = self._next_condition_change()
                next_fate = self._next_due_fate()
                next_open_release = self._next_open_release_time()

                def intervention_time(commit_time: float) -> float | None:
                    candidates: list[float] = []
                    if next_open_release is not None and next_open_release <= commit_time:
                        candidates.append(next_open_release)
                    if next_fate is not None and next_fate.time <= commit_time:
                        candidates.append(next_fate.time)
                    if next_change is not None and next_change.time < commit_time:
                        candidates.append(next_change.time)
                    return min(candidates) if candidates else None

                if self.state.pending_event is not None:
                    pending = self.state.pending_event
                    pending_time = float(pending["time"])
                    jump = intervention_time(pending_time)
                    if jump is not None and (end_time is None or jump <= end_time):
                        self.state.pending_event = None
                        self._discard_pending_event(pending)
                        self.state.time = jump
                        continue
                    if end_time is not None and pending_time > end_time:
                        self.state.time = end_time
                        self.state.status = "time_limit"
                        break
                    self.step()
                    events_used += 1
                    continue

                rng_state_before = self.state.rng_state
                rng = self._rng()
                available, propensities, total = self._available()
                if total <= 0.0:
                    jump = None
                    candidates = [
                        value
                        for value in (
                            next_open_release,
                            next_fate.time if next_fate else None,
                            next_change.time if next_change else None,
                        )
                        if value is not None
                    ]
                    if candidates:
                        jump = min(candidates)
                    if jump is not None and (end_time is None or jump <= end_time):
                        self.state.time = jump
                        continue
                    if end_time is not None and self.state.time < end_time:
                        self.state.time = end_time
                    self.state.status = (
                        "exhausted"
                        if not self.state.pending_releases
                        and not self.state.fate_schedule
                        else "time_limit"
                    )
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

                jump = intervention_time(event.time)
                if jump is not None and (end_time is None or jump <= end_time):
                    self.state.time = jump
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
                follow_up = self._pump_due(allow_equal=True)
                events_used += 1 + len(follow_up)
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

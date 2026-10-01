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
    waiting_created: tuple[dict[str, Any], ...] = ()

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
            waiting_created=tuple(dict(item) for item in data.get("waiting_created", ())),
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


@dataclass(frozen=True)
class WaitingItem:
    """A concrete batch of molecules held outside the free pool until release."""

    release_id: int
    reaction_id: str
    species_id: str
    amount: int
    created_time: float
    release_time: float
    created_version: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WaitingItem":
        return cls(
            release_id=int(data["release_id"]),
            reaction_id=str(data["reaction_id"]),
            species_id=str(data["species_id"]),
            amount=int(data["amount"]),
            created_time=float(data["created_time"]),
            release_time=float(data["release_time"]),
            created_version=int(data.get("created_version", 0)),
        )


@dataclass(frozen=True)
class ReleaseEvent:
    """A waiting batch entering the free pool at its scheduled instant."""

    time: float
    release_id: int
    reaction_id: str
    species_id: str
    amount: int
    reason: str
    counts_after: dict[str, int]
    waiting_after: dict[str, int]
    condition_version: int = 0
    created_version: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReleaseEvent":
        return cls(
            time=float(data["time"]),
            release_id=int(data["release_id"]),
            reaction_id=str(data["reaction_id"]),
            species_id=str(data["species_id"]),
            amount=int(data["amount"]),
            reason=str(data["reason"]),
            counts_after={str(k): int(v) for k, v in data["counts_after"].items()},
            waiting_after={str(k): int(v) for k, v in data["waiting_after"].items()},
            condition_version=int(data.get("condition_version", 0)),
            created_version=int(data.get("created_version", 0)),
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
    failure_reason: str | None = None
    waiting: list[WaitingItem] = field(default_factory=list)
    release_events: list[ReleaseEvent] = field(default_factory=list)
    next_release_id: int = 0

    def __post_init__(self) -> None:
        complete_counts = {species.id: 0 for species in self.model.species}
        complete_counts.update(self.counts)
        self.counts = complete_counts
        for species_id, count in self.counts.items():
            if count < 0:
                raise ValueError(f"count cannot be negative: {species_id}")
        self.condition_schedule.sort(key=lambda change: change.time)

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
            "failure_reason": self.failure_reason,
            "waiting": [item.to_dict() for item in self.waiting],
            "release_events": [item.to_dict() for item in self.release_events],
            "next_release_id": self.next_release_id,
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
            failure_reason=data.get("failure_reason"),
            waiting=[WaitingItem.from_dict(item) for item in data.get("waiting", [])],
            release_events=[
                ReleaseEvent.from_dict(item) for item in data.get("release_events", [])
            ],
            next_release_id=int(data.get("next_release_id", 0)),
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

    def _waiting_totals(self) -> dict[str, int]:
        totals = {species.id: 0 for species in self.model.species}
        for item in self.state.waiting:
            totals[item.species_id] = totals.get(item.species_id, 0) + item.amount
        return totals

    def _next_release_time(self) -> float | None:
        return self.state.waiting[0].release_time if self.state.waiting else None

    def _apply_due_releases(self, allow_equal: bool = True) -> list[ReleaseEvent]:
        """Move every waiting batch due at the current time into the free pool.

        Releases tied with the current time are applied before a stochastic
        event or condition change tied with that same instant; the molecules
        of a batch that already started waiting are unaffected by later rate
        changes and keep their scheduled release time and identity.
        """

        def reached(item: WaitingItem) -> bool:
            return item.release_time < self.state.time or (
                allow_equal and item.release_time == self.state.time
            )

        due = [item for item in self.state.waiting if reached(item)]
        if not due:
            return []
        # Identity order: same-instant batches release in creation order so
        # branches and save points replay identical release sequences.
        due.sort(key=lambda item: (item.release_time, item.release_id))
        applied: list[ReleaseEvent] = []
        remaining = {item.release_id: item for item in self.state.waiting}
        for item in due:
            self.state.time = item.release_time
            self.state.counts[item.species_id] = (
                self.state.counts.get(item.species_id, 0) + item.amount
            )
            del remaining[item.release_id]
            self.state.waiting = sorted(
                remaining.values(), key=lambda entry: (entry.release_time, entry.release_id)
            )
            reason = (
                f"scheduled release of waiting batch {item.release_id} from reaction "
                f"{item.reaction_id}; batch created at t={item.created_time:.12g} under "
                f"condition_version={item.created_version}; released under "
                f"condition_version={self.state.condition_version}"
            )
            release = ReleaseEvent(
                time=item.release_time,
                release_id=item.release_id,
                reaction_id=item.reaction_id,
                species_id=item.species_id,
                amount=item.amount,
                reason=reason,
                counts_after=dict(self.state.counts),
                waiting_after=self._waiting_totals(),
                condition_version=self.state.condition_version,
                created_version=item.created_version,
            )
            self.state.release_events.append(release)
            applied.append(release)
        return applied

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
                rate_constant=rate_constants[reaction.id],
                name=reaction.name,
                delayed_products={
                    species_id: (amount, delay)
                    for species_id, (amount, delay) in reaction.delayed_products.items()
                },
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

    def step(self) -> ReactionEvent | ReleaseEvent | None:
        if self.state.pending_event is not None:
            pending = self.state.pending_event
            next_change = self._next_condition_change()
            pending_time = float(pending["time"])
            next_release = self._next_release_time()
            # Any waiting batch due no later than the stored tentative event
            # invalidates that draw: release first, then redraw afterwards.
            if next_release is not None and next_release <= pending_time:
                self.state.pending_event = None
                self._discard_pending_event(pending)
                self.state.time = next_release
                releases = self._apply_due_releases(allow_equal=True)
                self._apply_due_condition_changes(allow_equal=True)
                return releases[0]
            if next_change is not None and next_change.time < pending_time:
                # A condition now intervenes before the stored pre-drawn event.
                # Discard that draw and its random numbers, then redraw under
                # the new version so resume matches a continuous run.
                self.state.pending_event = None
                self._discard_pending_event(pending)
                self.state.time = next_change.time
                self._apply_due_condition_changes(allow_equal=True)
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
        next_release = self._next_release_time()
        rng = self._rng()
        available, propensities, total = self._available()
        if total <= 0.0:
            if next_release is not None:
                self.state.time = next_release
                releases = self._apply_due_releases(allow_equal=True)
                self._apply_due_condition_changes(allow_equal=True)
                return releases[0]
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

        if next_release is not None and next_release <= self.state.time + waiting_time:
            # A release precedes (or ties with) the tentative reaction:
            # discard the draw and its random numbers, release first.
            self.state.time = next_release
            releases = self._apply_due_releases(allow_equal=True)
            self._apply_due_condition_changes(allow_equal=True)
            return releases[0]

        event = self._build_event(
            selected,
            self.state.time + waiting_time,
            waiting_time,
            propensities[selected.id],
            total,
            self.state.condition_version,
            self._event_reason(propensities[selected.id], total),
        )[0]
        self.state.rng_state = rng.getstate()
        applied = self._apply_built_event(event)
        self._apply_due_condition_changes(allow_equal=True)
        return applied

    def _make_waiting_items(
        self,
        selected: Reaction,
        event_time: float,
        descriptors: list[dict[str, Any]] | None = None,
    ) -> list[WaitingItem]:
        items: list[WaitingItem] = []
        if descriptors is None:
            descriptors = [
                {"species_id": species_id, "amount": amount, "delay": delay}
                for species_id, (amount, delay) in selected.delayed_products.items()
            ]
        for descriptor in descriptors:
            species_id = str(descriptor["species_id"])
            amount = int(descriptor["amount"])
            delay = float(descriptor["delay"])
            item = WaitingItem(
                release_id=self.state.next_release_id,
                reaction_id=selected.id,
                species_id=species_id,
                amount=amount,
                created_time=event_time,
                release_time=event_time + delay,
                created_version=self.state.condition_version,
            )
            self.state.next_release_id += 1
            items.append(item)
        return items

    def _build_event(
        self,
        selected: Reaction,
        event_time: float,
        waiting_time: float,
        propensity: float,
        total_propensity: float,
        condition_version: int,
        reason: str,
    ) -> tuple[ReactionEvent, list[WaitingItem]]:
        if not self.rule.is_possible(selected, self.state.counts):
            raise RuntimeError(f"rule selected impossible reaction {selected.id}")
        changes = selected.initiation_stoichiometry
        new_counts = dict(self.state.counts)
        for species_id, change in changes.items():
            new_counts[species_id] = new_counts.get(species_id, 0) + change
        descriptors = [
            {
                "species_id": species_id,
                "amount": amount,
                "delay": delay,
            }
            for species_id, (amount, delay) in selected.delayed_products.items()
        ]
        if descriptors:
            reason = (
                f"{reason}; {len(descriptors)} waiting batch(es) scheduled: "
                + ", ".join(
                    f"{int(descriptor['amount'])}x{descriptor['species_id']}@"
                    f"{event_time + float(descriptor['delay']):.12g}"
                    for descriptor in descriptors
                )
            )
        event = ReactionEvent(
            time=event_time,
            reaction_id=selected.id,
            waiting_time=waiting_time,
            propensity=propensity,
            total_propensity=total_propensity,
            changes={k: v for k, v in changes.items() if v != 0},
            reason=reason,
            counts_after=new_counts,
            condition_version=condition_version,
            waiting_created=tuple(descriptors),
        )
        return event, []

    def _apply_built_event(self, event: ReactionEvent) -> ReactionEvent:
        self.state.time = event.time
        self.state.counts = dict(event.counts_after)
        waiting_items = self._make_waiting_items(
            self.model.reaction(event.reaction_id),
            event.time,
            list(event.waiting_created),
        )
        self.state.waiting.extend(waiting_items)
        self.state.waiting.sort(key=lambda item: (item.release_time, item.release_id))
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
        built = self._build_event(
            selected,
            event_time,
            waiting_time,
            propensity,
            total_propensity,
            condition_version,
            reason,
        )
        event, built_items = built
        assert not built_items
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
        event, _ = self._build_event(
            selected,
            self.state.time + waiting_time,
            waiting_time,
            propensities[selected.id],
            total,
            self.state.condition_version,
            self._event_reason(propensities[selected.id], total),
        )
        return event

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
                self._apply_due_condition_changes(allow_equal=False)
                # When a run is capped exactly on a boundary, complete every
                # release and condition declaration tied with that instant.
                if end_time is not None and self.state.time == end_time:
                    self._apply_due_releases(allow_equal=True)
                    self._apply_due_condition_changes(allow_equal=True)
                if end_time is not None and self.state.time >= end_time:
                    self.state.status = "time_limit"
                    break

                next_change = self._next_condition_change()
                next_release = self._next_release_time()

                if self.state.pending_event is not None:
                    pending = self.state.pending_event
                    pending_time = float(pending["time"])
                    intervening_release = (
                        next_release is not None and next_release <= pending_time
                    )
                    intervening_change = (
                        next_change is not None and next_change.time < pending_time
                    )
                    if intervening_release or intervening_change:
                        self.state.pending_event = None
                        self._discard_pending_event(pending)
                        if intervening_release and (
                            intervening_change is False
                            or (next_release is not None and next_release <= next_change.time)
                        ):
                            boundary = next_release
                            if end_time is not None and end_time <= boundary:
                                self.state.time = end_time
                                self._apply_due_releases(allow_equal=True)
                                self._apply_due_condition_changes(allow_equal=True)
                                self.state.status = "time_limit"
                                break
                            self.state.time = boundary
                            self._apply_due_releases(allow_equal=True)
                        else:
                            if end_time is not None and end_time <= next_change.time:
                                self.state.time = end_time
                                self._apply_due_condition_changes(allow_equal=True)
                                self.state.status = "time_limit"
                                break
                            self.state.time = next_change.time
                            self._apply_due_condition_changes(allow_equal=True)
                        continue
                    if end_time is not None and pending_time > end_time:
                        self.state.time = end_time
                        # Releases due exactly at the end instant still happen;
                        # they complete before the run is capped; a condition
                        # declared for that same instant then takes effect too.
                        self._apply_due_releases(allow_equal=True)
                        self._apply_due_condition_changes(allow_equal=True)
                        self.state.status = "time_limit"
                        break
                    self._apply_due_releases(allow_equal=False)
                    self.step()
                    events_used += 1
                    continue

                rng_state_before = self.state.rng_state
                rng = self._rng()
                event = self._draw_event(rng)
                if event is None:
                    # No stochastic event is possible, but waiting batches may
                    # still release material that enables later reactions.
                    if next_release is not None and (
                        end_time is None or next_release <= end_time
                    ):
                        self.state.rng_state = rng.getstate()
                        self.state.time = next_release
                        self._apply_due_releases(allow_equal=True)
                        continue
                    self.state.rng_state = rng.getstate()
                    self.state.status = "exhausted"
                    break

                release_blocks = (
                    next_release is not None
                    and event.time >= next_release
                    and (end_time is None or next_release <= end_time)
                )
                change_blocks = (
                    next_change is not None
                    and event.time > next_change.time
                    and (end_time is None or next_change.time <= end_time)
                )
                if release_blocks or change_blocks:
                    # A scheduled boundary precedes (or ties, for releases)
                    # the tentative event: discard the draw without consuming
                    # random numbers, advance to the boundary, then redraw.
                    self.state.rng_state = rng_state_before
                    use_release = release_blocks and (
                        not change_blocks
                        or (next_release is not None and next_release <= next_change.time)
                    )
                    if use_release:
                        self.state.time = next_release
                        self._apply_due_releases(allow_equal=True)
                    else:
                        self.state.time = next_change.time
                        self._apply_due_condition_changes(allow_equal=True)
                    continue

                if end_time is not None and event.time > end_time:
                    # Commit the draw itself, but keep the pre-draw RNG state
                    # so a resume with a newly intervening condition can roll
                    # the discarded numbers back and stay on one trajectory.
                    self.state.time = end_time
                    self.state.pending_event = {
                        "time": event.time,
                        "reaction_id": event.reaction_id,
                        "waiting_time": event.waiting_time,
                        "propensity": event.propensity,
                        "total_propensity": event.total_propensity,
                        "reason": self._event_reason(
                            event.propensity, event.total_propensity
                        ),
                        "condition_version": event.condition_version,
                        "waiting_created": [dict(item) for item in event.waiting_created],
                        "rng_state_before": self._serialize_rng_state(rng_state_before),
                    }
                    self.state.rng_state = rng.getstate()
                    # A waiting batch due exactly at the end instant still
                    # completes; it is applied before the run is capped so a
                    # resume immediately after sees the released material.
                    self._apply_due_releases(allow_equal=True)
                    self._apply_due_condition_changes(allow_equal=True)
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

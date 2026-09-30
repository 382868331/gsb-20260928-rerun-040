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
    failure_reason: str | None = None

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

    def step(self) -> ReactionEvent | None:
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
        changes = selected.stoichiometry
        new_counts = dict(self.state.counts)
        for species_id, change in changes.items():
            new_counts[species_id] = new_counts.get(species_id, 0) + change
        return ReactionEvent(
            time=event_time,
            reaction_id=selected.id,
            waiting_time=waiting_time,
            propensity=propensity,
            total_propensity=total_propensity,
            changes={k: v for k, v in changes.items() if v != 0},
            reason=reason,
            counts_after=new_counts,
            condition_version=condition_version,
        )

    def _apply_built_event(self, event: ReactionEvent) -> ReactionEvent:
        self.state.time = event.time
        self.state.counts = dict(event.counts_after)
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
                if self.state.pending_event is not None:
                    pending = self.state.pending_event
                    pending_time = float(self.state.pending_event["time"])
                    next_change = self._next_condition_change()
                    if next_change is not None and next_change.time < pending_time:
                        # A condition intervenes before the pre-drawn event:
                        # discard it together with its consumed random draws.
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
                    if end_time is not None and pending_time > end_time:
                        self.state.time = end_time
                        self.state.status = "time_limit"
                        break
                    self._apply_due_condition_changes(allow_equal=False)
                    self.step()
                    events_used += 1
                    continue

                self._apply_due_condition_changes(allow_equal=False)
                if end_time is not None and self.state.time >= end_time:
                    self.state.status = "time_limit"
                    break

                next_change = self._next_condition_change()
                rng_state_before = self.state.rng_state
                rng = self._rng()
                event = self._draw_event(rng)
                if event is None:
                    self.state.rng_state = rng.getstate()
                    self.state.status = "exhausted"
                    break

                if (
                    next_change is not None
                    and event.time > next_change.time
                    and (end_time is None or next_change.time <= end_time)
                ):
                    # Conditions change first: discard the tentative draw
                    # without consuming random numbers and redraw afterwards.
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

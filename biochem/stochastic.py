"""Stochastic reaction runner and resumable trajectory state."""

from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass, field
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
    version: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "rate_constants": dict(self.rate_constants),
            "label": self.label,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConditionChange":
        return cls(
            time=float(data["time"]),
            rate_constants={str(k): float(v) for k, v in data["rate_constants"].items()},
            label=str(data.get("label", "")),
            version=int(data.get("version", 0)),
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

    def clone(self) -> "TrajectoryState":
        """Deep-copy the full state, including the RNG state, so divergent
        condition scenarios never share subsequent random draws."""

        return TrajectoryState.from_dict(copy.deepcopy(self.to_dict()))


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
        for change in self.state.condition_schedule:
            if change.time > self.state.time:
                return change
        return None

    def _apply_due_condition_changes(self) -> None:
        """Apply scheduled changes whose time has been reached."""
        due = [
            change
            for change in self.state.condition_schedule
            if change.time <= self.state.time
        ]
        if not due:
            return
        # Every distinct change time introduces one condition version, even
        # when several changes take effect at that same instant.
        self.state.condition_version += 1
        new_version = self.state.condition_version
        rate_constants = {
            reaction.id: reaction.rate_constant for reaction in self.model.reactions
        }
        for change in due:
            rate_constants.update(change.rate_constants)
            applied = ConditionChange(
                time=change.time,
                rate_constants=dict(change.rate_constants),
                label=change.label,
                version=new_version,
            )
            self.state.condition_events.append(applied)
        self.state.condition_schedule = [
            change
            for change in self.state.condition_schedule
            if change.time > self.state.time
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

    def step(self) -> ReactionEvent | None:
        """Advance by exactly one reaction event, honouring the ordering between
        pending events and condition changes exactly as a continuous run does.
        """

        while True:
            # A pending event was already drawn before the last save. It only
            # survives if no condition change takes effect before it; an event
            # at the exact change time keeps the old condition version.
            if self.state.pending_event is not None:
                pending = self.state.pending_event
                pending_time = float(pending["time"])
                next_change = self._next_condition_change()
                if next_change is not None and next_change.time < pending_time:
                    self._discard_pending_event(next_change.time)
                    continue
                if pending_time < self.state.time:
                    raise RuntimeError(
                        "pending event is earlier than current trajectory time"
                    )
                selected = self.model.reaction(str(pending["reaction_id"]))
                self.state.pending_event = None
                post_draw = pending.get("rng_state_after")
                if post_draw is not None:
                    self.state.rng_state = self._decode_rng_state(post_draw)
                return self._apply_event(
                    selected,
                    pending_time,
                    float(pending["waiting_time"]),
                    float(pending["propensity"]),
                    float(pending["total_propensity"]),
                    str(
                        pending.get(
                            "reason",
                            "mass-action event continued from saved state",
                        )
                    ),
                    int(pending.get("condition_version", self.state.condition_version)),
                )

            self._apply_due_condition_changes()
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

            next_change = self._next_condition_change()
            event_time = self.state.time + waiting_time
            if next_change is not None and next_change.time < event_time:
                # The tentative draw straddles a condition change: discard it
                # without consuming RNG draws, exactly as a continuous run.
                self.state.time = next_change.time
                continue

            event = self._build_event(
                selected,
                event_time,
                waiting_time,
                propensities[selected.id],
                total,
                self._mass_action_reason(propensities[selected.id], total),
                self.state.condition_version,
            )
            self.state.rng_state = rng.getstate()
            return self._apply_built_event(event)

    @staticmethod
    def _encode_rng_state(rng_state: tuple[Any, ...]) -> list[Any]:
        version, values, gauss = rng_state
        return [version, list(values), gauss]

    @staticmethod
    def _decode_rng_state(data: Any) -> tuple[Any, ...]:
        if data is None:
            return random.Random().getstate()
        return (int(data[0]), tuple(int(v) for v in data[1]), data[2])

    def _discard_pending_event(self, change_time: float) -> None:
        """Roll back a committed draw that a now-earlier condition change
        invalidates, restoring the RNG state from before that draw."""

        pending = self.state.pending_event
        pre_draw = pending.get("rng_state_before")
        if pre_draw is not None:
            self.state.rng_state = self._decode_rng_state(pre_draw)
        self.state.pending_event = None
        self.state.time = float(change_time)

    @staticmethod
    def _mass_action_reason(propensity: float, total: float) -> str:
        return (
            f"mass-action propensity {propensity:.12g} "
            f"selected from total {total:.12g}; all reactant counts were sufficient"
        )

    def _build_event(
        self,
        selected: Reaction,
        event_time: float,
        waiting_time: float,
        propensity: float,
        total_propensity: float,
        reason: str,
        condition_version: int,
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
        reason: str,
        condition_version: int,
    ) -> ReactionEvent:
        event = self._build_event(
            selected,
            event_time,
            waiting_time,
            propensity,
            total_propensity,
            reason,
            condition_version,
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
            self._mass_action_reason(propensities[selected.id], total),
            self.state.condition_version,
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
                # Apply any condition changes due at the current time first.
                self._apply_due_condition_changes()

                # Resolve an already-committed pending event, preserving the
                # strict ordering against intervening condition changes.
                if self.state.pending_event is not None:
                    pending_time = float(self.state.pending_event["time"])
                    next_change = self._next_condition_change()
                    if next_change is not None and next_change.time < pending_time:
                        if end_time is not None and next_change.time > end_time:
                            self.state.time = end_time
                            self.state.status = "time_limit"
                            break
                        self._discard_pending_event(next_change.time)
                        continue
                    if end_time is not None and pending_time > end_time:
                        self.state.time = end_time
                        self.state.status = "time_limit"
                        break
                    self.step()
                    events_used += 1
                    continue

                if end_time is not None and self.state.time >= end_time:
                    self.state.status = "time_limit"
                    break

                rng_state_before = self.state.rng_state
                rng = self._rng()
                next_change = self._next_condition_change()
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
                    # Conditions change before this event would occur; discard
                    # the tentative draw (rng_state untouched) and redraw after
                    # the change so the same schedule replays identically.
                    self.state.time = next_change.time
                    continue

                if end_time is not None and event.time > end_time:
                    # Commit the draw: the waiting time and reaction are already
                    # decided, so a resume must not re-sample them.  Store both
                    # the pre- and post-draw RNG states: if a condition change
                    # scheduled before the pending event appears on resume, the
                    # draw must be rolled back and retried under new conditions.
                    self.state.time = end_time
                    self.state.pending_event = {
                        "time": event.time,
                        "reaction_id": event.reaction_id,
                        "waiting_time": event.waiting_time,
                        "propensity": event.propensity,
                        "total_propensity": event.total_propensity,
                        "reason": event.reason,
                        "condition_version": event.condition_version,
                        "rng_state_before": self._encode_rng_state(rng_state_before),
                        "rng_state_after": self._encode_rng_state(rng.getstate()),
                    }
                    self.state.status = "time_limit"
                    break

                self.state.rng_state = rng.getstate()
                self._apply_built_event(event)
                events_used += 1
        except Exception as exc:
            self.state.status = "failed"
            self.state.failure_reason = f"{type(exc).__name__}: {exc}"
            raise
        return self.state

    def branch(
        self,
        rate_constants: dict[str, float] | None = None,
        label: str = "",
    ) -> "StochasticRunner":
        """Fork an independent runner from this checkpoint. The clone carries
        its own RNG state, so different condition scenarios draw independent
        subsequent event streams."""

        forked = StochasticRunner.from_state(self.state.clone(), rule=self.rule)
        if rate_constants is not None:
            forked.schedule_condition_change(
                self.state.time, dict(rate_constants), label=label
            )
        return forked

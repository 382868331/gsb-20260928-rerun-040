"""Stochastic reaction runner and resumable trajectory state."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any

from .models import ExperimentModel, Reaction
from .rules import MassActionRule, ReactionRule


@dataclass(frozen=True)
class ConditionChange:
    time: float
    rate_constants: dict[str, float]
    status: str = "scheduled"

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "rate_constants": dict(self.rate_constants),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConditionChange":
        return cls(
            time=float(data["time"]),
            rate_constants={
                str(reaction_id): float(value)
                for reaction_id, value in data.get("rate_constants", {}).items()
            },
            status=str(data.get("status", "scheduled")),
        )


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
    condition_changes: list[ConditionChange] = field(default_factory=list)
    failed_at: float | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        complete_counts = {species.id: 0 for species in self.model.species}
        complete_counts.update(self.counts)
        self.counts = complete_counts
        for species_id, count in self.counts.items():
            if count < 0:
                raise ValueError(f"count cannot be negative: {species_id}")

    def to_dict(self) -> dict[str, Any]:
        rng_version, rng_values, rng_gauss = self.rng_state
        return {
            "schema_version": 2,
            "time": self.time,
            "counts": dict(self.counts),
            "events": [event.to_dict() for event in self.events],
            "rng_state": [rng_version, list(rng_values), rng_gauss],
            "status": self.status,
            "pending_event": self.pending_event,
            "condition_changes": [change.to_dict() for change in self.condition_changes],
            "failed_at": self.failed_at,
            "failure_reason": self.failure_reason,
            "model": self.model.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryState":
        model = ExperimentModel.from_dict(data["model"])
        rng_data = data["rng_state"]
        rng_state = (int(rng_data[0]), tuple(int(v) for v in rng_data[1]), rng_data[2])
        state = cls(
            model=model,
            time=float(data.get("time", 0.0)),
            counts={str(k): int(v) for k, v in data.get("counts", {}).items()},
            events=[ReactionEvent.from_dict(item) for item in data.get("events", [])],
            rng_state=rng_state,
            status=str(data.get("status", "ready")),
            pending_event=data.get("pending_event"),
            condition_changes=[
                ConditionChange.from_dict(item)
                for item in data.get("condition_changes", [])
            ],
            failed_at=None if data.get("failed_at") is None else float(data["failed_at"]),
            failure_reason=data.get("failure_reason"),
        )
        if int(data.get("schema_version", 1)) < 2:
            state.legacy_format = True
        return state


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
        runner._normalize_legacy_state()
        return runner

    def _normalize_legacy_state(self) -> None:
        """Advance old state RNGs by the two samples consumed to schedule an event."""
        if getattr(self.state, "_state_format_normalized", False):
            return
        if self.state.pending_event is not None:
            self.state.pending_event.setdefault("source", "current_conditions")
        if self.state.pending_event is None:
            if getattr(self.state, "legacy_format", False):
                self._draw_next_event()
        elif getattr(self.state, "legacy_format", False):
            rng = self._rng()
            rng.expovariate(1.0)
            rng.random()
            self.state.rng_state = rng.getstate()
        self.state._state_format_normalized = True
        if hasattr(self.state, "legacy_format"):
            del self.state.legacy_format

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

    def _next_change_index(self, through_time: float | None = None) -> int | None:
        candidates = [
            index
            for index, change in enumerate(self.state.condition_changes)
            if change.status == "scheduled"
            and (through_time is None or change.time <= through_time)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda index: self.state.condition_changes[index].time)

    def _invalidate_pending_for_change(self, change_time: float) -> None:
        pending = self.state.pending_event
        if pending is not None and float(pending["time"]) >= change_time:
            self.state.pending_event = None

    def _apply_due_condition_changes(self, through_time: float) -> None:
        while True:
            index = self._next_change_index(through_time)
            if index is None:
                return
            change = self.state.condition_changes[index]
            same_time_indices = [
                candidate_index
                for candidate_index, candidate in enumerate(self.state.condition_changes)
                if candidate.status == "scheduled" and candidate.time == change.time
            ]
            merged_rates: dict[str, float] = {}
            for candidate_index in same_time_indices:
                merged_rates.update(self.state.condition_changes[candidate_index].rate_constants)
            self._invalidate_pending_for_change(change.time)
            try:
                new_model = self.model.with_rate_constants(merged_rates)
                new_model.validate()
            except (KeyError, ValueError) as exc:
                for candidate_index in same_time_indices:
                    candidate = self.state.condition_changes[candidate_index]
                    self.state.condition_changes[candidate_index] = ConditionChange(
                        time=candidate.time,
                        rate_constants=dict(candidate.rate_constants),
                        status="failed",
                    )
                self._fail(change.time, str(exc))
                raise
            self.model = new_model
            self.state.model = new_model
            self.state.time = change.time
            for candidate_index in same_time_indices:
                candidate = self.state.condition_changes[candidate_index]
                self.state.condition_changes[candidate_index] = ConditionChange(
                    time=candidate.time,
                    rate_constants=dict(candidate.rate_constants),
                    status="applied",
                )
            self.state.status = "condition_changed"
            if self.state.pending_event is None:
                self._draw_next_event()

    def _draw_next_event(self) -> None:
        rng = self._rng()
        available, propensities, total = self._available()
        if total <= 0.0:
            self.state.rng_state = rng.getstate()
            self.state.pending_event = None
            return

        waiting_time = rng.expovariate(total)
        cutoff = rng.random() * total
        cumulative = 0.0
        selected = available[-1]
        for candidate in available:
            cumulative += propensities[candidate.id]
            if cutoff <= cumulative:
                selected = candidate
                break

        self.state.rng_state = rng.getstate()
        self.state.pending_event = {
            "time": self.state.time + waiting_time,
            "reaction_id": selected.id,
            "waiting_time": waiting_time,
            "propensity": propensities[selected.id],
            "total_propensity": total,
            "source": "current_conditions",
        }

    def _fail(self, failed_at: float, reason: str) -> None:
        self.state.status = "failed"
        self.state.failed_at = failed_at
        self.state.failure_reason = reason

    def schedule_rate_change(self, time: float, rate_constants: dict[str, float]) -> ConditionChange:
        if time < self.state.time:
            raise ValueError("condition change precedes current trajectory time")
        if not rate_constants:
            raise ValueError("condition change must update at least one rate constant")
        values = {str(key): float(value) for key, value in rate_constants.items()}
        for existing in self.state.condition_changes:
            if existing.time == time and existing.rate_constants == values:
                return existing
        try:
            probe_model = self.model.with_rate_constants(values)
            probe_model.validate()
        except (KeyError, ValueError) as exc:
            self.state.condition_changes.append(
                ConditionChange(time=time, rate_constants=values, status="failed")
            )
            self._fail(time, str(exc))
            raise
        change = ConditionChange(time=time, rate_constants=values)
        self.state.condition_changes.append(change)
        self.state.condition_changes.sort(key=lambda item: item.time)
        self._invalidate_pending_for_change(time)
        return change

    def step(self) -> ReactionEvent | None:
        if self.state.pending_event is None:
            self._draw_next_event()
        if self.state.pending_event is not None:
            pending = self.state.pending_event
            selected = self.model.reaction(str(pending["reaction_id"]))
            event_time = float(pending["time"])
            if event_time < self.state.time:
                reason = "pending event is earlier than current trajectory time"
                self._fail(event_time, reason)
                raise RuntimeError(reason)
            self.state.pending_event = None
            event = self._apply_event(
                selected,
                event_time,
                float(pending["waiting_time"]),
                float(pending["propensity"]),
                float(pending["total_propensity"]),
                (
                    f"mass-action propensity {float(pending['propensity']):.12g} "
                    f"selected from total {float(pending['total_propensity']):.12g}; "
                    "all reactant counts were sufficient"
                ),
            )
            self._draw_next_event()
            return event

        self._draw_next_event()
        if self.state.pending_event is None:
            return None
        return self.step()

    def _build_event(
        self,
        selected: Reaction,
        event_time: float,
        waiting_time: float,
        propensity: float,
        total_propensity: float,
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
    ) -> ReactionEvent:
        event = self._build_event(
            selected,
            event_time,
            waiting_time,
            propensity,
            total_propensity,
            reason,
        )
        return self._apply_built_event(event)

    def run(
        self,
        end_time: float | None = None,
        max_events: int | None = None,
    ) -> TrajectoryState:
        if end_time is not None and end_time < self.state.time:
            raise ValueError("end_time precedes current trajectory time")
        events_used = 0
        self._apply_due_condition_changes(self.state.time)
        if self.state.pending_event is None and self.state.status not in {"failed", "exhausted"}:
            self._draw_next_event()
        while max_events is None or events_used < max_events:
            pending = self.state.pending_event
            if pending is None:
                next_change = self._next_change_index()
                if next_change is None:
                    self.state.status = "exhausted"
                    break
                change = self.state.condition_changes[next_change]
                if end_time is not None and change.time > end_time:
                    self.state.time = end_time
                    self.state.status = "time_limit"
                    break
                self._apply_due_condition_changes(change.time)
                continue

            event_time = float(pending["time"])
            next_change = self._next_change_index()
            change_time = (
                self.state.condition_changes[next_change].time
                if next_change is not None
                else None
            )
            if change_time is not None and change_time <= event_time:
                if end_time is not None and change_time > end_time:
                    self.state.time = end_time
                    self.state.status = "time_limit"
                    break
                self._apply_due_condition_changes(change_time)
                continue
            if end_time is not None and event_time > end_time:
                self.state.time = end_time
                self.state.status = "time_limit"
                break

            try:
                self.step()
            except RuntimeError as exc:
                self._fail(event_time, str(exc))
                raise
            events_used += 1

        if (
            end_time is not None
            and self.state.time < end_time
            and self.state.status not in {"failed", "exhausted"}
        ):
            self.state.time = end_time
            self.state.status = "time_limit"
        return self.state

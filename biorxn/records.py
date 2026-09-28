"""运行事件与轨迹记录。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    """每次数量变化都对应一条事件，包含原因和发生时间。"""

    index: int
    time: float
    reaction_id: str
    reason: str
    before: dict[str, int]
    change: dict[str, int]
    after: dict[str, int]
    propensity: float
    total_propensity: float
    wait_time: float
    rng_draw: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "time": self.time,
            "reaction_id": self.reaction_id,
            "reason": self.reason,
            "before": self.before,
            "change": self.change,
            "after": self.after,
            "propensity": self.propensity,
            "total_propensity": self.total_propensity,
            "wait_time": self.wait_time,
            "rng_draw": list(self.rng_draw),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        draw = data["rng_draw"]
        return cls(
            index=int(data["index"]),
            time=float(data["time"]),
            reaction_id=str(data["reaction_id"]),
            reason=str(data["reason"]),
            before=dict(data["before"]),
            change=dict(data["change"]),
            after=dict(data["after"]),
            propensity=float(data["propensity"]),
            total_propensity=float(data["total_propensity"]),
            wait_time=float(data["wait_time"]),
            rng_draw=(float(draw[0]), float(draw[1])),
        )


@dataclass
class RunRecord:
    run_id: str
    seed: int
    initial_time: float = 0.0
    initial_counts: dict[str, int] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def event_count(self) -> int:
        return len(self.events)

    def counts_at(self, time: float) -> dict[str, int]:
        counts = dict(self.initial_counts)
        for event in self.events:
            if event.time > time:
                break
            for species_id, delta in event.change.items():
                counts[species_id] = counts.get(species_id, 0) + delta
        return counts

    def final_counts(self) -> dict[str, int]:
        if not self.events:
            return dict(self.initial_counts)
        return dict(self.events[-1].after)

    def final_time(self) -> float:
        return self.events[-1].time if self.events else self.initial_time

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "biorxn.record/v1",
            "run_id": self.run_id,
            "seed": self.seed,
            "initial_time": self.initial_time,
            "initial_counts": self.initial_counts,
            "events": [event.to_dict() for event in self.events],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunRecord":
        return cls(
            run_id=str(data["run_id"]),
            seed=int(data["seed"]),
            initial_time=float(data.get("initial_time", 0.0)),
            initial_counts=dict(data.get("initial_counts", {})),
            events=[Event.from_dict(item) for item in data.get("events", [])],
            notes=dict(data.get("notes", {})),
        )

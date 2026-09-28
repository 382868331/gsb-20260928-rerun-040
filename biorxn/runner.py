"""小规模离散随机反应运行器（Gillespie SSA）。"""

from __future__ import annotations

import random
import uuid
import math
from dataclasses import dataclass, field

from .model import Model
from .records import Event, RunRecord
from .rules import RuleRegistry, default_registry


@dataclass
class RunConfig:
    max_events: int = 1_000_000
    stop_on_no_propensity: bool = True


@dataclass
class GillespieRunner:
    model: Model
    seed: int
    counts: dict[str, int]
    time: float = 0.0
    run_id: str = ""
    rules: RuleRegistry = field(default_factory=lambda: default_registry)
    record: RunRecord | None = None
    config: RunConfig = field(default_factory=RunConfig)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self.counts = {species_id: int(self.counts.get(species_id, 0)) for species_id in self.model.species_ids}
        if any(value < 0 for value in self.counts.values()):
            raise ValueError("物种数量不能为负")
        if not self.run_id:
            self.run_id = uuid.uuid4().hex
        self.record = RunRecord(
            run_id=self.run_id,
            seed=self.seed,
            initial_time=self.time,
            initial_counts=dict(self.counts),
        )

    @property
    def rng(self) -> random.Random:
        return self._rng

    def propensities(self) -> dict[str, float]:
        return self.rules.propensities(self.model, self.counts)

    def step(self) -> Event | None:
        propensities = self.propensities()
        total = sum(propensities.values())
        if total <= 0.0:
            return None
        r1, r2 = self._rng.random(), self._rng.random()
        wait_time = math.log(1.0 / (1.0 - r1)) / total
        threshold = r2 * total
        cumulative = 0.0
        chosen_id = ""
        chosen_propensity = 0.0
        for reaction_id, propensity in propensities.items():
            cumulative += propensity
            if cumulative >= threshold:
                chosen_id = reaction_id
                chosen_propensity = propensity
                break
        reaction = self.model.reactions[chosen_id]
        before = dict(self.counts)
        change = reaction.changes()
        for species_id, required in reaction.reactants.items():
            if self.counts[species_id] < required:
                raise RuntimeError(f"反应物不足却触发反应 {chosen_id}")
        self.counts = {
            species_id: self.counts.get(species_id, 0) + change.get(species_id, 0)
            for species_id in self.model.species_ids
        }
        self.time += wait_time
        event = Event(
            index=len(self.record.events),
            time=self.time,
            reaction_id=chosen_id,
            reason=f"reaction:{chosen_id}",
            before=before,
            change=change,
            after=dict(self.counts),
            propensity=chosen_propensity,
            total_propensity=total,
            wait_time=wait_time,
            rng_draw=(r1, r2),
        )
        self.record.events.append(event)
        return event

    def _draw_next(self) -> tuple[str, float, float, float, float] | None:
        propensities = self.propensities()
        total = sum(propensities.values())
        if total <= 0.0:
            return None
        r1, r2 = self._rng.random(), self._rng.random()
        wait_time = math.log(1.0 / (1.0 - r1)) / total
        threshold = r2 * total
        cumulative = 0.0
        chosen_id = ""
        chosen_propensity = 0.0
        for reaction_id, propensity in propensities.items():
            cumulative += propensity
            if cumulative >= threshold:
                chosen_id = reaction_id
                chosen_propensity = propensity
                break
        return chosen_id, chosen_propensity, total, wait_time, r2

    def _fire(self, chosen_id: str, chosen_propensity: float, total: float, wait_time: float, r2: float) -> Event:
        r1 = 1.0 - math.exp(-total * wait_time)
        reaction = self.model.reactions[chosen_id]
        before = dict(self.counts)
        change = reaction.changes()
        for species_id, required in reaction.reactants.items():
            if self.counts[species_id] < required:
                raise RuntimeError(f"反应物不足却触发反应 {chosen_id}")
        self.counts = {
            species_id: self.counts.get(species_id, 0) + change.get(species_id, 0)
            for species_id in self.model.species_ids
        }
        self.time += wait_time
        event = Event(
            index=len(self.record.events),
            time=self.time,
            reaction_id=chosen_id,
            reason=f"reaction:{chosen_id}",
            before=before,
            change=change,
            after=dict(self.counts),
            propensity=chosen_propensity,
            total_propensity=total,
            wait_time=wait_time,
            rng_draw=(r1, r2),
        )
        self.record.events.append(event)
        return event

    def run(
        self,
        duration: float | None = None,
        until_time: float | None = None,
        max_events: int | None = None,
    ) -> RunRecord:
        if duration is not None and until_time is not None:
            raise ValueError("duration 与 until_time 只能指定一个")
        target_time = until_time
        if duration is not None:
            if duration < 0:
                raise ValueError("duration 不能为负")
            target_time = self.time + duration
        limit = max_events if max_events is not None else self.config.max_events
        while len(self.record.events) < limit:
            if target_time is not None and self.time >= target_time:
                break
            rng_state = self._rng.getstate()
            drawn = self._draw_next()
            if drawn is None:
                if self.config.stop_on_no_propensity:
                    self.record.notes["stopped_reason"] = "zero_propensity"
                    break
                continue
            chosen_id, propensity, total, wait_time, r2 = drawn
            if target_time is not None and self.time + wait_time > target_time:
                self._rng.setstate(rng_state)
                break
            self._fire(chosen_id, propensity, total, wait_time, r2)
        self.record.notes["final_time"] = self.time
        self.record.notes["final_counts"] = dict(self.counts)
        return self.record

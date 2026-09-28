"""确定性计算入口。

与随机运行器共享 Model 和质量作用规则。这里使用显式欧拉法，
适合第一版小规模网络与短时间核对；不读取随机预设曲线。
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import Model
from .rules import RuleRegistry, default_registry


@dataclass
class DeterministicResult:
    times: list[float]
    states: list[dict[str, float]]

    @property
    def final_state(self) -> dict[str, float]:
        return dict(self.states[-1])


def deterministic_euler(
    model: Model,
    initial_counts: dict[str, int | float],
    duration: float,
    step_size: float,
    rules: RuleRegistry | None = None,
) -> DeterministicResult:
    if duration <= 0:
        raise ValueError("duration 必须为正")
    if step_size <= 0:
        raise ValueError("step_size 必须为正")
    registry = rules or default_registry
    state = {
        species_id: float(initial_counts.get(species_id, 0.0))
        for species_id in model.species_ids
    }
    times = [0.0]
    states = [dict(state)]
    steps = int(duration // step_size)
    for step in range(1, steps + 1):
        fluxes = registry.propensities(model, state)
        next_state = dict(state)
        for reaction in model.reactions.values():
            flux = fluxes[reaction.id] * step_size
            for species_id, delta in reaction.changes().items():
                next_state[species_id] = next_state.get(species_id, 0.0) + delta * flux
        state = {key: max(0.0, value) for key, value in next_state.items()}
        times.append(step * step_size)
        states.append(dict(state))
    return DeterministicResult(times=times, states=states)

"""反应规则：把当前数量映射为反应倾向度。

规则与运行器分离。运行器只依赖 propensity 函数；
确定性通量默认使用同一个质量作用规则，因此随机与确定性读取同一份网络。
"""

from __future__ import annotations

import math
from collections.abc import Callable

from .model import Model, Reaction

PropensityRule = Callable[[Reaction, dict[str, int]], float]


def mass_action(reaction: Reaction, counts: dict[str, int]) -> float:
    """基元质量作用倾向度。

    对整数粒子数使用组合数 C(n,k)/k! 形式：
    0 阶为 1，1 阶为 n，2 阶同类为 n(n-1)/2。
    确定性浓度场景传入浮点数时退化为 n**k/k!。
    """
    value = float(reaction.rate)
    for species_id, stoichiometry in reaction.reactants.items():
        amount = counts.get(species_id, 0)
        if isinstance(amount, int):
            if amount < stoichiometry:
                return 0.0
            combinations = math.comb(amount, stoichiometry)
        else:
            combinations = _float_combination(float(amount), stoichiometry)
        value *= combinations
    return max(0.0, value)


def _float_combination(amount: float, stoichiometry: int) -> float:
    if amount < stoichiometry:
        return 0.0
    result = 1.0
    for k in range(stoichiometry):
        result *= amount - k
    return result / math.factorial(stoichiometry)


def can_fire(reaction: Reaction, counts: dict[str, int]) -> bool:
    return all(
        counts.get(species_id, 0) >= required
        for species_id, required in reaction.reactants.items()
    )


class RuleRegistry:
    def __init__(self) -> None:
        self._rules: dict[str, PropensityRule] = {}
        self.register("mass_action", mass_action)

    def register(self, name: str, rule: PropensityRule) -> None:
        self._rules[name] = rule

    def get(self, name: str) -> PropensityRule:
        try:
            return self._rules[name]
        except KeyError as exc:
            raise KeyError(f"未注册的反应规则: {name}") from exc

    def propensities(
        self, model: Model, counts: dict[str, int | float]
    ) -> dict[str, float]:
        result = {}
        for reaction in model.reactions.values():
            if not can_fire(reaction, counts):
                result[reaction.id] = 0.0
            else:
                result[reaction.id] = self.get(reaction.rule)(reaction, counts)
        return result


default_registry = RuleRegistry()

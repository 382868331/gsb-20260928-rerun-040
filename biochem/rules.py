"""Reaction-rate rules independent of the simulation runner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .models import ExperimentModel, Reaction


class ReactionRule(Protocol):
    def is_possible(self, reaction: Reaction, counts: dict[str, int]) -> bool: ...

    def propensity(self, reaction: Reaction, counts: dict[str, int]) -> float: ...


@dataclass(frozen=True)
class MassActionRule:
    """Mass-action propensity using distinct molecular combinations."""

    def is_possible(self, reaction: Reaction, counts: dict[str, int]) -> bool:
        return all(
            counts.get(species_id, 0) >= amount
            for species_id, amount in reaction.reactants.items()
        )

    def propensity(self, reaction: Reaction, counts: dict[str, int]) -> float:
        if not self.is_possible(reaction, counts):
            return 0.0
        value = reaction.rate_constant
        for species_id, stoichiometry in reaction.reactants.items():
            available = counts.get(species_id, 0)
            value *= math.comb(available, stoichiometry)
        return float(value)

    def propensities(
        self, model: ExperimentModel, counts: dict[str, int]
    ) -> dict[str, float]:
        return {
            reaction.id: self.propensity(reaction, counts)
            for reaction in model.reactions
        }

    def derivative(
        self, model: ExperimentModel, concentrations: dict[str, float]
    ) -> dict[str, float]:
        """Continuous mass-action derivative for deterministic calculations."""
        rates: dict[str, float] = {}
        for reaction in model.reactions:
            rate = float(reaction.rate_constant)
            for species_id, stoichiometry in reaction.reactants.items():
                rate *= concentrations.get(species_id, 0.0) ** stoichiometry
            rates[reaction.id] = rate

        derivative = {species_id: 0.0 for species_id in model.species_ids}
        for reaction in model.reactions:
            for species_id, change in reaction.stoichiometry.items():
                derivative[species_id] += change * rates[reaction.id]
        return derivative

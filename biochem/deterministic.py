"""Deterministic mass-action calculations sharing the same model."""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from .models import ExperimentModel
from .rules import MassActionRule


@dataclass(frozen=True)
class DeterministicPoint:
    time: float
    values: dict[str, float]
    waiting: dict[str, float] | None = None


def rk4_scan(
    model: ExperimentModel,
    end_time: float,
    step_size: float,
    initial: dict[str, float] | None = None,
    rule: MassActionRule | None = None,
) -> list[DeterministicPoint]:
    model.validate()
    if end_time < 0 or step_size <= 0:
        raise ValueError("end_time must be non-negative and step_size positive")
    delayed_species = {
        species_id
        for reaction in model.reactions
        for species_id in reaction.delayed_products
    }
    delayed = {
        reaction.id: dict(reaction.delayed_products)
        for reaction in model.reactions
        if reaction.delayed_products
    }
    values = {species.id: 0.0 for species in model.species}
    values.update(
        {key: float(value) for key, value in (initial or model.initial_counts).items()}
    )

    def reaction_rates(current: dict[str, float]) -> dict[str, float]:
        rates: dict[str, float] = {}
        for reaction in model.reactions:
            rate = float(reaction.rate_constant)
            for species_id, stoichiometry in reaction.reactants.items():
                rate *= current.get(species_id, 0.0) ** stoichiometry
            rates[reaction.id] = rate
        return rates

    def immediate_derivative(
        current: dict[str, float], rates: dict[str, float]
    ) -> dict[str, float]:
        """Derivative excluding delayed products, which enter via the queue."""

        derivative = {species_id: 0.0 for species_id in model.species_ids}
        for reaction in model.reactions:
            changes = dict(reaction.stoichiometry)
            for species_id in delayed.get(reaction.id, {}):
                # Withheld product: contributes neither to the free pool
                # derivative nor as a consumption term.
                changes.pop(species_id, None)
            for species_id, change in changes.items():
                derivative[species_id] += change * rates[reaction.id]
        return derivative

    waiting = {species.id: 0.0 for species in model.species}
    release_queue: list[tuple[float, int, dict[str, float]]] = []
    order_counter = 0

    def snapshot(time: float) -> DeterministicPoint:
        return DeterministicPoint(
            time,
            dict(values),
            {species_id: waiting.get(species_id, 0.0) for species_id in values}
            if delayed_species
            else None,
        )

    points = [snapshot(0.0)]
    time = 0.0
    while time < end_time - 1e-12:
        dt = min(step_size, end_time - time)
        rates1 = reaction_rates(values)
        k1 = immediate_derivative(values, rates1)
        second = {key: values[key] + 0.5 * dt * k1[key] for key in values}
        rates2 = reaction_rates(second)
        k2 = immediate_derivative(second, rates2)
        third = {key: values[key] + 0.5 * dt * k2[key] for key in values}
        rates3 = reaction_rates(third)
        k3 = immediate_derivative(third, rates3)
        fourth = {key: values[key] + dt * k3[key] for key in values}
        rates4 = reaction_rates(fourth)
        k4 = immediate_derivative(fourth, rates4)
        values = {
            key: max(
                0.0,
                values[key]
                + dt * (k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key]) / 6.0,
            )
            for key in values
        }
        if delayed:
            for reaction_id, products in delayed.items():
                average_rate = (
                    rates1[reaction_id]
                    + 2 * rates2[reaction_id]
                    + 2 * rates3[reaction_id]
                    + rates4[reaction_id]
                ) / 6.0
                flux = average_rate * dt
                amount = {
                    species_id: flux * amount_multiplier
                    for species_id, (amount_multiplier, _) in products.items()
                }
                delay = next(iter(products.values()))[1]
                for species_id in amount:
                    waiting[species_id] += amount[species_id]
                heapq.heappush(
                    release_queue, (time + dt + delay, order_counter, amount)
                )
                order_counter += 1
        time += dt
        while release_queue and release_queue[0][0] <= time + 1e-12:
            _, _, amount = heapq.heappop(release_queue)
            for species_id, value in amount.items():
                waiting[species_id] -= value
                values[species_id] += value
        shown_time = end_time if abs(time - end_time) < 1e-12 else time
        points.append(snapshot(shown_time))

    return points

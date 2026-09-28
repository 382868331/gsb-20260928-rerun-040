"""Deterministic mass-action calculations sharing the same model."""

from __future__ import annotations

from dataclasses import dataclass
from .models import ExperimentModel
from .rules import MassActionRule


@dataclass(frozen=True)
class DeterministicPoint:
    time: float
    values: dict[str, float]


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
    mass_action = rule or MassActionRule()
    values = {species.id: 0.0 for species in model.species}
    values.update(
        {key: float(value) for key, value in (initial or model.initial_counts).items()}
    )
    points = [DeterministicPoint(0.0, dict(values))]
    time = 0.0
    while time < end_time - 1e-12:
        dt = min(step_size, end_time - time)
        k1 = mass_action.derivative(model, values)
        second = {key: values[key] + 0.5 * dt * k1[key] for key in values}
        k2 = mass_action.derivative(model, second)
        third = {key: values[key] + 0.5 * dt * k2[key] for key in values}
        k3 = mass_action.derivative(model, third)
        fourth = {key: values[key] + dt * k3[key] for key in values}
        k4 = mass_action.derivative(model, fourth)
        values = {
            key: max(
                0.0,
                values[key]
                + dt * (k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key]) / 6.0,
            )
            for key in values
        }
        time += dt
        points.append(DeterministicPoint(end_time if abs(time - end_time) < 1e-12 else time, dict(values)))
    return points

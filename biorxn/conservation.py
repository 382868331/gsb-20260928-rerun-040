"""用模型声明的守恒关系核对每一步和整段运行。"""

from __future__ import annotations

from .model import Model
from .records import Event, RunRecord


def conserved_values(law: dict[str, int], counts: dict[str, int | float]) -> float:
    return sum(coefficient * counts.get(species_id, 0) for species_id, coefficient in law.items())


def verify_law(
    law: dict[str, int],
    record: RunRecord,
    tolerance: float = 0.0,
) -> list[dict[str, float | int]]:
    expected = conserved_values(law, record.initial_counts)
    violations = []
    for event_index, event in enumerate(record.events):
        value = conserved_values(law, event.after)
        if abs(value - expected) > tolerance:
            violations.append(
                {"event": event_index, "time": event.time, "value": value, "expected": expected}
            )
    return violations


def verify_model_conservation(model: Model, record: RunRecord) -> dict[str, list]:
    return {
        _law_key(law): verify_law(law, record)
        for law in model.conservation_laws()
    }


def assert_record_conserved(model: Model, record: RunRecord) -> None:
    report = verify_model_conservation(model, record)
    failures = {key: items for key, items in report.items() if items}
    if failures:
        raise AssertionError(f"检测到守恒违反: {failures}")


def _law_key(law: dict[str, int]) -> str:
    return ",".join(
        f"{coefficient:+d}{species_id}"
        for species_id, coefficient in sorted(law.items())
        if coefficient
    )

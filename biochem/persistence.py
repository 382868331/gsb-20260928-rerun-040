"""JSON persistence for models and resumable trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import ExperimentModel
from .stochastic import TrajectoryState


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_model(model: ExperimentModel, path: str | Path) -> None:
    write_json(path, model.to_dict())


def load_model(path: str | Path) -> ExperimentModel:
    return ExperimentModel.from_dict(read_json(path))


def save_state(state: TrajectoryState, path: str | Path) -> None:
    write_json(path, state.to_dict())


def load_state(path: str | Path) -> TrajectoryState:
    return TrajectoryState.from_dict(read_json(path))

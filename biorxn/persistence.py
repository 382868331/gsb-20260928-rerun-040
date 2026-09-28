"""模型、运行记录和可继续轨迹快照的 JSON 持久化。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model import Model
from .records import RunRecord
from .runner import GillespieRunner, RunConfig


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_model(model: Model, path: str | Path) -> None:
    save_json(path, model.to_dict())


def load_model(path: str | Path) -> Model:
    return Model.from_dict(load_json(path))


def save_record(record: RunRecord, path: str | Path) -> None:
    save_json(path, record.to_dict())


def load_record(path: str | Path) -> RunRecord:
    return RunRecord.from_dict(load_json(path))


def save_snapshot(runner: GillespieRunner, path: str | Path) -> None:
    version, internal, gauss_next = runner.rng.getstate()
    payload = {
        "schema": "biorxn.snapshot/v1",
        "model": runner.model.to_dict(),
        "run": runner.record.to_dict(),
        "state": {
            "time": runner.time,
            "counts": dict(runner.counts),
            "rng": {
                "version": version,
                "internal": list(internal),
                "gauss_next": gauss_next,
            },
            "config": runner.config.__dict__,
        },
    }
    save_json(path, payload)


def load_snapshot(path: str | Path) -> GillespieRunner:
    payload = load_json(path)
    model = Model.from_dict(payload["model"])
    run_data = payload["run"]
    state = payload["state"]
    runner = GillespieRunner(
        model=model,
        seed=int(run_data["seed"]),
        counts={key: int(value) for key, value in state["counts"].items()},
        time=float(state["time"]),
        run_id=str(run_data["run_id"]),
        config=RunConfig(**state.get("config", {})),
    )
    runner.record = RunRecord.from_dict(run_data)
    rng_state = state["rng"]
    runner.rng.setstate(
        (
            int(rng_state["version"]),
            tuple(int(value) for value in rng_state["internal"]),
            rng_state["gauss_next"],
        )
    )
    return runner

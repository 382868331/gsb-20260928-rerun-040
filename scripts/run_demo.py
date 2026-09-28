#!/usr/bin/env python3
"""运行第一版演示：随机短实验、守恒核对、断点续跑、确定性对照。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biorxn.conservation import verify_model_conservation
from biorxn.deterministic import deterministic_euler
from biorxn.experiments import run_short_experiments, write_replicates_csv, write_summary_csv
from biorxn.networks import enzyme_toggle_network, initial_enzyme_toggle
from biorxn.persistence import load_snapshot, save_model, save_record, save_snapshot


def main() -> None:
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    model = enzyme_toggle_network()
    initial = initial_enzyme_toggle()
    save_model(model, output / "model.json")

    print("模型:", model.id)
    print("守恒关系:", model.conservation_laws())

    runner = __import__("biorxn.runner", fromlist=["GillespieRunner"]).GillespieRunner(
        model=model, seed=20260929, counts=initial
    )
    record_first = runner.run(duration=3.0)
    save_snapshot(runner, output / "snapshot_t3.json")
    save_record(record_first, output / "record_t3.json")
    continued = load_snapshot(output / "snapshot_t3.json")
    record_full = continued.run(duration=5.0)
    save_snapshot(continued, output / "snapshot_t8.json")
    save_record(record_full, output / "record_t8.json")
    violations = verify_model_conservation(model, record_full)

    print(f"0-3 事件数: {record_first.event_count()}")
    print(f"续跑至 t={continued.time:.4f}，累计事件数: {record_full.event_count()}")
    print("守恒违反数:", {key: len(items) for key, items in violations.items()})

    summary = run_short_experiments(
        model=model,
        initial_counts=initial,
        duration=2.0,
        replicate_count=200,
        base_seed=500_000,
    )
    write_summary_csv(summary, output / "summary_t2_n200.csv")
    write_replicates_csv(summary.results, output / "replicates_t2_n200.csv")
    deterministic = deterministic_euler(model, initial, duration=2.0, step_size=0.02)

    print("\n200 次随机短实验 t=0->2 的平均变化:")
    for species_id, value in summary.mean_changes.items():
        print(f"  {species_id}: {value:+.3f} (确定性末态 {deterministic.final_state[species_id] - initial[species_id]:+.3f})")


if __name__ == "__main__":
    main()

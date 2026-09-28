from pathlib import Path

import pytest

from biorxn.deterministic import deterministic_euler
from biorxn.experiments import run_short_experiments
from biorxn.model import Model, Reaction, Species
from biorxn.networks import enzyme_toggle_network, initial_enzyme_toggle
from biorxn.persistence import load_snapshot, save_snapshot
from biorxn.runner import GillespieRunner


def test_conservation_laws_include_enzyme_and_substrate_pools():
    laws = enzyme_toggle_network().conservation_laws()
    enzyme_law = {"E": 1, "ES": 1, "S": 0, "P": 0}
    assert enzyme_law in laws
    other = next(law for law in laws if law != enzyme_law)
    substrate_law = {
        species_id: enzyme_law.get(species_id, 0) + other.get(species_id, 0)
        for species_id in ("E", "S", "ES", "P")
    }
    assert substrate_law == {"E": 0, "ES": 1, "S": 1, "P": 1}


def test_reaction_cannot_fire_without_enough_material():
    model = enzyme_toggle_network()
    runner = GillespieRunner(
        model=model,
        seed=7,
        counts={"E": 0, "S": 60, "ES": 0, "P": 0},
    )
    assert all(value == 0.0 for value in runner.propensities().values())
    record = runner.run(duration=10.0)
    assert record.events == []
    assert runner.counts == {"E": 0, "S": 60, "ES": 0, "P": 0}


def test_random_run_preserves_conserved_quantities_and_event_reason():
    model = enzyme_toggle_network()
    runner = GillespieRunner(model=model, seed=42, counts=initial_enzyme_toggle())
    record = runner.run(duration=5.0)
    assert record.events
    assert all(event.reason == f"reaction:{event.reaction_id}" for event in record.events)
    for event in record.events:
        assert event.after["E"] + event.after["ES"] == 10
        assert event.after["S"] + event.after["ES"] + event.after["P"] == 60


def test_snapshot_continues_exact_same_trajectory(tmp_path: Path):
    model = enzyme_toggle_network()
    full = GillespieRunner(model=model, seed=123, counts=initial_enzyme_toggle())
    full_record = full.run(duration=8.0)

    split = GillespieRunner(model=model, seed=123, counts=initial_enzyme_toggle())
    split.run(duration=3.0)
    snapshot = tmp_path / "snapshot.json"
    save_snapshot(split, snapshot)
    restored = load_snapshot(snapshot)
    restored_record = restored.run(until_time=8.0)

    assert restored.time == pytest.approx(full.time)
    assert restored.counts == full.counts
    assert [event.to_dict() for event in restored_record.events] == [
        event.to_dict() for event in full_record.events
    ]


def test_short_experiments_compare_average_changes():
    model = enzyme_toggle_network()
    summary = run_short_experiments(
        model=model,
        initial_counts=initial_enzyme_toggle(),
        duration=1.0,
        replicate_count=80,
    )
    assert len(summary.results) == 80
    assert summary.mean_changes["ES"] >= 0
    assert summary.mean_changes["P"] > 0
    assert summary.mean_changes["E"] + summary.mean_changes["ES"] == pytest.approx(0.0)
    assert (
        summary.mean_changes["S"]
        + summary.mean_changes["ES"]
        + summary.mean_changes["P"]
    ) == pytest.approx(0.0)


def test_deterministic_and_stochastic_share_same_model():
    model = enzyme_toggle_network()
    initial = initial_enzyme_toggle()
    result = deterministic_euler(model, initial, duration=1.0, step_size=0.05)
    final_state = result.final_state
    assert final_state["E"] + final_state["ES"] == pytest.approx(10.0)
    assert final_state["S"] + final_state["ES"] + final_state["P"] == pytest.approx(60.0)


def test_model_rejects_unknown_species():
    with pytest.raises(Exception):
        Model(
            species=[Species("A")],
            reactions=[Reaction("bad", reactants={"X": 1}, products={"A": 1}, rate=1.0)],
        )

import json
import pytest

from biochem.analysis import run_short_experiments, validate_trajectory_conservation
from biochem.deterministic import rk4_scan
from biochem.models import Reaction, build_conservation_network
from biochem.persistence import load_state, save_state
from biochem.rules import MassActionRule
from biochem.stochastic import StochasticRunner


def test_insufficient_reactants_have_zero_propensity():
    rule = MassActionRule()
    reaction = Reaction("need_two", {"S": 2}, {"P": 1}, 2.0)

    assert rule.propensity(reaction, {"S": 1}) == 0.0
    assert not rule.is_possible(reaction, {"S": 1})
    assert rule.propensity(reaction, {"S": 2}) == 2.0
    assert rule.propensity(reaction, {"S": 3}) == 6.0


def test_events_record_time_reason_and_nonnegative_counts():
    model = build_conservation_network()
    runner = StochasticRunner(model, seed=7)
    state = runner.run(end_time=3)

    assert state.events
    for event in state.events:
        assert event.time >= 0
        assert event.waiting_time >= 0
        assert event.reaction_id in {"bind", "unbind", "catalyze"}
        assert event.propensity > 0
        assert event.total_propensity > 0
        assert "mass-action propensity" in event.reason
        assert all(count >= 0 for count in event.counts_after.values())
        assert event.counts_after["E"] + event.counts_after["ES"] == 3
        assert event.counts_after["S"] + event.counts_after["ES"] + event.counts_after["P"] == 12


def test_seed_is_reproducible():
    first = StochasticRunner(build_conservation_network(), seed=42).run(4)
    second = StochasticRunner(build_conservation_network(), seed=42).run(4)

    assert [event.to_dict() for event in first.events] == [
        event.to_dict() for event in second.events
    ]
    assert first.counts == second.counts


def test_saved_midpoint_continues_on_same_trajectory(tmp_path):
    uninterrupted = StochasticRunner(build_conservation_network(), seed=99).run(6)

    split_runner = StochasticRunner(build_conservation_network(), seed=99)
    split_runner.run(end_time=2.5)
    state_path = tmp_path / "state.json"
    save_state(split_runner.state, state_path)

    raw = json.loads(state_path.read_text(encoding="utf-8"))
    assert raw["model"]["reactions"]
    assert "rng_state" in raw

    resumed = StochasticRunner.from_state(load_state(state_path))
    resumed.run(end_time=6)

    assert resumed.state.time == uninterrupted.time
    assert resumed.state.counts == uninterrupted.counts
    assert [event.to_dict() for event in resumed.state.events] == [
        event.to_dict() for event in uninterrupted.events
    ]


def test_multiple_arbitrary_and_boundary_splits_match_continuous_run(tmp_path):
    model = build_conservation_network()
    uninterrupted = StochasticRunner(model, seed=1234)
    uninterrupted.run(end_time=8)

    event_times = [event.time for event in uninterrupted.state.events]
    checkpoints = [0.73, event_times[1], 2.41, event_times[4], 5.8, event_times[-2]]

    split_runner = StochasticRunner(model, seed=1234)
    state_path = tmp_path / "segmented.json"
    for checkpoint in checkpoints:
        split_runner.run(end_time=checkpoint)
        save_state(split_runner.state, state_path)
        reloaded = StochasticRunner.from_state(load_state(state_path))
        split_runner = reloaded

    split_runner.run(end_time=8)

    assert [event.to_dict() for event in split_runner.state.events] == [
        event.to_dict() for event in uninterrupted.state.events
    ]
    assert split_runner.state.counts == uninterrupted.state.counts


def test_condition_change_is_replayable_across_restart(tmp_path):
    model = build_conservation_network()
    continuous = StochasticRunner(model, seed=321)
    continuous.schedule_rate_change(2.0, {"bind": 0.4, "catalyze": 0.08})
    continuous.run(end_time=6)

    state_path = tmp_path / "conditions.json"
    before = StochasticRunner(model, seed=321)
    before.schedule_rate_change(2.0, {"bind": 0.4, "catalyze": 0.08})
    before.run(end_time=1.25)
    save_state(before.state, state_path)

    resumed = StochasticRunner.from_state(load_state(state_path))
    resumed.run(end_time=2.0)
    save_state(resumed.state, state_path)

    after_change = StochasticRunner.from_state(load_state(state_path))
    after_change.run(end_time=6)

    recorded = [change.to_dict() for change in after_change.state.condition_changes]
    assert recorded[0]["rate_constants"] == {"bind": 0.4, "catalyze": 0.08}
    assert recorded[0]["status"] == "applied"
    assert [event.to_dict() for event in after_change.state.events] == [
        event.to_dict() for event in continuous.state.events
    ]


def test_condition_change_at_event_boundary_takes_precedence_and_replays(tmp_path):
    model = build_conservation_network()
    probe = StochasticRunner(model, seed=77)
    probe.run(max_events=3)
    boundary = probe.state.events[1].time

    continuous = StochasticRunner(model, seed=77)
    continuous.schedule_rate_change(boundary, {"bind": 0.0})
    continuous.run(end_time=boundary + 3)

    state_path = tmp_path / "boundary-condition.json"
    first = StochasticRunner(model, seed=77)
    first.run(end_time=boundary - 0.1)
    save_state(first.state, state_path)

    resumed = StochasticRunner.from_state(load_state(state_path))
    resumed.schedule_rate_change(boundary, {"bind": 0.0})
    resumed.run(end_time=boundary + 3)

    assert [event.to_dict() for event in resumed.state.events] == [
        event.to_dict() for event in continuous.state.events
    ]


def test_invalid_condition_change_keeps_last_trusted_state_on_failure(tmp_path):
    model = build_conservation_network()
    runner = StochasticRunner(model, seed=5)
    runner.run(end_time=1.0)
    trusted_events = [event.to_dict() for event in runner.state.events]
    trusted_counts = dict(runner.state.counts)

    with pytest.raises(KeyError):
        runner.schedule_rate_change(1.5, {"missing": 0.1})

    state_path = tmp_path / "failed.json"
    save_state(runner.state, state_path)
    recovered = load_state(state_path)

    assert recovered.status == "failed"
    assert recovered.failed_at == 1.5
    assert "unknown reaction" in (recovered.failure_reason or "")
    assert recovered.counts == trusted_counts
    assert [event.to_dict() for event in recovered.events] == trusted_events


def test_built_in_network_conservation_and_short_experiments():
    model = build_conservation_network()
    trajectories, summary = run_short_experiments(model, runs=40, duration=4)

    assert len(trajectories) == 40
    assert summary.mean_events > 0
    assert summary.mean_change["P"] > 0
    assert summary.mean_change["S"] < 0
    for state in trajectories:
        validate_trajectory_conservation(state)


def test_deterministic_scan_uses_same_model_and_preserves_totals():
    model = build_conservation_network()
    points = rk4_scan(model, end_time=5, step_size=0.05)

    assert len(points) == 101
    final = points[-1].values
    assert final["P"] > 0
    assert abs((final["E"] + final["ES"]) - 3) < 1e-9
    assert abs((final["S"] + final["ES"] + final["P"]) - 12) < 1e-9


def test_exhausted_network_stops_cleanly():
    model = build_conservation_network()
    runner = StochasticRunner(model, seed=1)
    runner.state.counts.update({"S": 0, "ES": 0, "P": 0})

    result = runner.run(end_time=100)

    assert result.status == "exhausted"
    assert result.events == []
    assert result.counts == {"E": 3, "S": 0, "ES": 0, "P": 0}

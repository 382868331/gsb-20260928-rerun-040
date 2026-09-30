import json

from biochem.analysis import compare_scenarios, run_short_experiments, validate_trajectory_conservation
from biochem.deterministic import rk4_scan
from biochem.models import Reaction, build_conservation_network
from biochem.persistence import load_state, save_state
from biochem.rules import MassActionRule
from biochem.stochastic import ConditionChange, StochasticRunner


def _condition_at_time(events, change_time):
    return next(event for event in events if abs(event.time - change_time) < 1e-12)


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


def event_dicts(state):
    return [event.to_dict() for event in state.events]


def test_multiple_segments_match_continuous_run(tmp_path):
    continuous = StochasticRunner(build_conservation_network(), seed=123).run(6)

    runner = StochasticRunner(build_conservation_network(), seed=123)
    for index, end_time in enumerate((1.3, 2.7, 4.1, 6.0)):
        runner.run(end_time=end_time)
        path = tmp_path / f"segment{index}.json"
        save_state(runner.state, path)
        runner = StochasticRunner.from_state(load_state(path))

    assert runner.state.time == continuous.time
    assert runner.state.counts == continuous.counts
    assert event_dicts(runner.state) == event_dicts(continuous)


def test_boundary_exactly_on_event_time(tmp_path):
    continuous = StochasticRunner(build_conservation_network(), seed=99).run(6)
    boundary = continuous.events[3].time

    runner = StochasticRunner(build_conservation_network(), seed=99)
    runner.run(end_time=boundary)
    save_state(runner.state, tmp_path / "boundary.json")
    resumed = StochasticRunner.from_state(load_state(tmp_path / "boundary.json"))
    resumed.run(end_time=6)

    assert event_dicts(resumed.state) == event_dicts(continuous)
    assert resumed.state.counts == continuous.counts


def test_condition_change_is_replayable_across_segments(tmp_path):
    continuous = StochasticRunner(build_conservation_network(), seed=5)
    continuous.schedule_condition_change(2.0, {"catalyze": 2.0, "bind": 0.01}, label="boost")
    continuous.run(6)

    runner = StochasticRunner(build_conservation_network(), seed=5)
    runner.schedule_condition_change(2.0, {"catalyze": 2.0, "bind": 0.01}, label="boost")
    for index, end_time in enumerate((1.0, 3.0, 6.0)):
        runner.run(end_time=end_time)
        path = tmp_path / f"cond{index}.json"
        save_state(runner.state, path)
        runner = StochasticRunner.from_state(load_state(path))

    assert event_dicts(runner.state) == event_dicts(continuous.state)
    assert runner.state.counts == continuous.state.counts
    assert [c.to_dict() for c in runner.state.condition_events] == [
        c.to_dict() for c in continuous.state.condition_events
    ]
    assert runner.state.condition_events[0].rate_constants == {"catalyze": 2.0, "bind": 0.01}


def test_condition_change_actually_changes_rates():
    model = build_conservation_network()
    runner = StochasticRunner(model, seed=5)
    runner.schedule_condition_change(0.5, {"bind": 0.0, "catalyze": 50.0})
    runner.run(4)

    assert runner.state.condition_events
    after = [e for e in runner.state.events if e.time > 0.5]
    assert after
    assert all(e.reaction_id != "bind" for e in after)
    assert runner.model.reaction("bind").rate_constant == 0.0


def _save_and_resume(runner, path, end_time):
    save_state(runner.state, path)
    resumed = StochasticRunner.from_state(load_state(path))
    resumed.run(end_time=end_time)
    return resumed


def test_segments_match_continuous_around_condition_time(tmp_path):
    """Save before, exactly at, and right after a condition takes effect.

    The committed draw straddling the change must be rolled back on resume
    rather than replayed under stale conditions."""

    def build():
        runner = StochasticRunner(build_conservation_network(), seed=5)
        runner.schedule_condition_change(2.0, {"bind": 0.005}, label="slow-bind")
        return runner

    continuous = build()
    continuous.run(6)

    # Save point just before the change with a pending event crossing it.
    pre = build()
    pre.run(end_time=1.999)
    pre = _save_and_resume(pre, tmp_path / "pre.json", 6)
    assert event_dicts(pre.state) == event_dicts(continuous.state)

    # Save point exactly at the change time.
    at = build()
    at.run(end_time=2.0)
    at = _save_and_resume(at, tmp_path / "at.json", 6)
    assert event_dicts(at.state) == event_dicts(continuous.state)

    # Save point immediately after the change.
    post = build()
    post.run(end_time=2.001)
    post = _save_and_resume(post, tmp_path / "post.json", 6)
    assert event_dicts(post.state) == event_dicts(continuous.state)

    # Many small segments, including one boundary that lands inside the
    # committed draw window (pending event saved while a change lies ahead).
    segmented = build()
    for index, end_time in enumerate((0.4, 1.1, 1.7, 1.99, 2.0, 2.3, 3.4, 6.0)):
        segmented.run(end_time=end_time)
        save_state(segmented.state, tmp_path / f"seg{index}.json")
        segmented = StochasticRunner.from_state(load_state(tmp_path / f"seg{index}.json"))
    assert event_dicts(segmented.state) == event_dicts(continuous.state)


def test_events_record_condition_version():
    runner = StochasticRunner(build_conservation_network(), seed=5)
    runner.schedule_condition_change(2.0, {"catalyze": 2.0}, label="v1")
    runner.run(end_time=1.5)
    assert all(event.condition_version == 0 for event in runner.state.events)

    runner.run(end_time=8.0)
    # Events strictly before the change use version 0; strictly after use 1.
    before = [event for event in runner.state.events if event.time < 2.0]
    after = [event for event in runner.state.events if event.time > 2.0]
    assert before and all(event.condition_version == 0 for event in before)
    assert after and all(event.condition_version == 1 for event in after)

    recorded = runner.state.condition_events
    assert recorded and recorded[0].version == 1
    assert recorded[0].label == "v1"


def test_multiple_conditions_same_time_share_one_version():
    runner = StochasticRunner(build_conservation_network(), seed=8)
    runner.schedule_condition_change(1.5, {"bind": 0.0}, label="stop-bind")
    runner.schedule_condition_change(1.5, {"catalyze": 5.0}, label="fast-cat")
    runner.run(4)

    applied = runner.state.condition_events
    assert len(applied) == 2
    assert {change.version for change in applied} == {1}
    after = [event for event in runner.state.events if event.time > 1.5]
    assert after and all(event.condition_version == 1 for event in after)
    assert runner.model.reaction("bind").rate_constant == 0.0
    assert runner.model.reaction("catalyze").rate_constant == 5.0


def test_event_exactly_at_change_time_keeps_old_version():
    plain = StochasticRunner(build_conservation_network(), seed=5)
    plain.run(6)
    boundary_time = plain.state.events[2].time

    continuous = StochasticRunner(build_conservation_network(), seed=5)
    continuous.schedule_condition_change(boundary_time, {"bind": 0.005})
    continuous.run(6)

    boundary_event = _condition_at_time(continuous.state.events, boundary_time)
    # An event at the exact change instant is the pre-change draw: version 0.
    assert boundary_event.condition_version == 0
    after = [event for event in continuous.state.events if event.time > boundary_time]
    assert after and all(event.condition_version == 1 for event in after)

    # A save/resume landing exactly on that boundary replays identically.
    segmented = StochasticRunner(build_conservation_network(), seed=5)
    segmented.schedule_condition_change(boundary_time, {"bind": 0.005})
    segmented.run(end_time=boundary_time)
    segmented = StochasticRunner.from_state(
        type(segmented.state).from_dict(segmented.state.to_dict())
    )
    segmented.run(6)
    assert event_dicts(segmented.state) == event_dicts(continuous.state)


def test_branched_scenarios_have_independent_rng_state(tmp_path):
    runner = StochasticRunner(build_conservation_network(), seed=21)
    runner.run(end_time=1.5)
    save_state(runner.state, tmp_path / "checkpoint.json")

    base = StochasticRunner.from_state(load_state(tmp_path / "checkpoint.json"))
    scenario_a = base.branch({"bind": 0.5}, label="fast-bind")
    scenario_b = base.branch({"bind": 0.005}, label="slow-bind")

    # Branching must not mutate or consume the shared checkpoint RNG state.
    scenario_a.run(5)
    scenario_b.run(5)

    replay_a = StochasticRunner.from_state(load_state(tmp_path / "checkpoint.json"))
    replay_a = replay_a.branch({"bind": 0.5}, label="fast-bind")
    replay_a.run(5)
    assert event_dicts(replay_a.state) == event_dicts(scenario_a.state)

    # Independent draws: advancing one scenario never changes the other.
    assert scenario_a.state.rng_state != scenario_b.state.rng_state or (
        [event.to_dict() for event in scenario_a.state.events]
        != [event.to_dict() for event in scenario_b.state.events]
    )
    assert scenario_a.state.condition_events[0].label == "fast-bind"
    assert scenario_b.state.condition_events[0].label == "slow-bind"

    # The original checkpoint is untouched by both branches.
    untouched = load_state(tmp_path / "checkpoint.json")
    assert untouched.events == runner.state.events
    assert untouched.condition_schedule == []


def test_compare_scenarios_runs_independent_forks():
    base = StochasticRunner(build_conservation_network(), seed=33)
    base.run(end_time=1.0)

    results = compare_scenarios(
        base,
        {"fast": {"catalyze": 5.0}, "slow": {"catalyze": 0.01}},
        end_time=4.0,
    )

    assert set(results) == {"fast", "slow"}
    assert results["fast"].condition_events[0].version == 1
    assert results["slow"].condition_events[0].version == 1
    # Prefix history before the fork is identical; later draws are independent.
    checkpoint_events = len(base.state.events)
    for state in results.values():
        validate_trajectory_conservation(state)
        assert event_dicts(state)[:checkpoint_events] == event_dicts(base.state)


def test_failed_run_keeps_last_trusted_state_and_reason(tmp_path):
    runner = StochasticRunner(build_conservation_network(), seed=11)
    runner.run(end_time=1.0)
    trusted_counts = dict(runner.state.counts)
    trusted_events = len(runner.state.events)

    runner.state.pending_event = {
        "time": runner.state.time + 0.1,
        "reaction_id": "no-such-reaction",
        "waiting_time": 0.1,
        "propensity": 1.0,
        "total_propensity": 1.0,
        "reason": "corrupted",
    }
    try:
        runner.run(end_time=3.0)
    except KeyError:
        pass
    else:
        raise AssertionError("expected the corrupted pending event to fail")

    assert runner.state.status == "failed"
    assert "no-such-reaction" in runner.state.failure_reason
    assert runner.state.counts == trusted_counts
    assert len(runner.state.events) == trusted_events

    save_state(runner.state, tmp_path / "failed.json")
    restored = load_state(tmp_path / "failed.json")
    assert restored.status == "failed"
    assert restored.failure_reason == runner.state.failure_reason
    assert restored.counts == trusted_counts

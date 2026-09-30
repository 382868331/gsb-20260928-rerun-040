import json

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


CONDITION_SEED = 15
CONDITION_TIME = 2.0


def _continuous_with_two_conditions():
    runner = StochasticRunner(build_conservation_network(), seed=CONDITION_SEED)
    runner.schedule_condition_change(
        CONDITION_TIME, {"catalyze": 5.0}, label="boost"
    )
    runner.schedule_condition_change(
        CONDITION_TIME, {"bind": 0.5}, label="damp"
    )
    runner.run(6)
    return runner


def test_resume_around_condition_change_matches_continuous_run(tmp_path):
    continuous = _continuous_with_two_conditions()
    first_event_after_change = next(
        event for event in continuous.state.events if event.time > CONDITION_TIME
    )
    assert continuous.state.events[0].time < CONDITION_TIME
    assert first_event_after_change.time > CONDITION_TIME

    save_times = (
        CONDITION_TIME - 0.25,
        CONDITION_TIME,
        CONDITION_TIME + 0.25,
    )
    for save_time in save_times:
        runner = StochasticRunner(build_conservation_network(), seed=CONDITION_SEED)
        runner.schedule_condition_change(
            CONDITION_TIME, {"catalyze": 5.0}, label="boost"
        )
        runner.schedule_condition_change(
            CONDITION_TIME, {"bind": 0.5}, label="damp"
        )
        runner.run(end_time=save_time)
        path = tmp_path / f"save_{save_time}.json"
        save_state(runner.state, path)
        resumed = StochasticRunner.from_state(load_state(path))
        resumed.run(end_time=6)

        assert event_dicts(resumed.state) == event_dicts(continuous.state)
        assert resumed.state.counts == continuous.state.counts


def test_simultaneous_changes_form_one_named_condition_version():
    continuous = _continuous_with_two_conditions()

    applied = continuous.state.condition_events
    assert [change.time for change in applied] == [CONDITION_TIME, CONDITION_TIME]
    assert [change.version for change in applied] == [1, 1]
    assert applied[0].rates_after["catalyze"] == 5.0
    assert applied[0].rates_after["bind"] == 0.12
    assert applied[1].rates_after["catalyze"] == 5.0
    assert applied[1].rates_after["bind"] == 0.5

    versions = {event.condition_version for event in continuous.state.events}
    assert versions <= {0, 1}
    before = [e for e in continuous.state.events if e.time < CONDITION_TIME]
    after = [e for e in continuous.state.events if e.time > CONDITION_TIME]
    assert before and after
    assert all(event.condition_version == 0 for event in before)
    assert all(event.condition_version == 1 for event in after)
    assert all("condition_version=1" in event.reason for event in after)


def test_pending_event_across_change_is_redrawn_without_consuming_extra_draws(tmp_path):
    continuous = _continuous_with_two_conditions()

    runner = StochasticRunner(build_conservation_network(), seed=CONDITION_SEED)
    # The condition is only present in the resumed continuation: the saved
    # pending event was drawn under the old rates and crosses t=2.
    runner.run(end_time=CONDITION_TIME - 0.25)
    assert runner.state.pending_event is not None
    assert runner.state.pending_event["time"] > CONDITION_TIME
    save_state(runner.state, tmp_path / "pending.json")

    resumed = StochasticRunner.from_state(load_state(tmp_path / "pending.json"))
    resumed.schedule_condition_change(
        CONDITION_TIME, {"catalyze": 5.0}, label="boost"
    )
    resumed.schedule_condition_change(
        CONDITION_TIME, {"bind": 0.5}, label="damp"
    )
    resumed.run(6)

    assert event_dicts(resumed.state) == event_dicts(continuous.state)
    assert resumed.state.rng_state == continuous.state.rng_state


def test_branches_from_checkpoint_have_independent_random_state():
    base = StochasticRunner(build_conservation_network(), seed=CONDITION_SEED)
    base.run(end_time=1.5)

    branch_a = base.fork()
    branch_b = base.fork()
    branch_a.schedule_condition_change(
        CONDITION_TIME, {"catalyze": 5.0}, label="boost"
    )
    branch_b.schedule_condition_change(
        CONDITION_TIME, {"catalyze": 0.01}, label="suppress"
    )

    branch_a.run(6)
    branch_b.run(6)

    assert event_dicts(branch_a.state) != event_dicts(branch_b.state)
    assert branch_a.state.rng_state != branch_b.state.rng_state
    # Mutating one branch must not bleed into the other or the checkpoint.
    assert len(base.state.events) < len(branch_a.state.events)
    assert not base.state.condition_events
    assert branch_a.state.condition_events[0].label == "boost"
    assert branch_b.state.condition_events[0].label == "suppress"

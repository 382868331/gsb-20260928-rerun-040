import json

import pytest

from biochem.analysis import run_short_experiments, validate_trajectory_conservation
from biochem.deterministic import rk4_scan
from biochem.models import ExperimentModel, Reaction, Species, build_conservation_network
from biochem.persistence import load_state, save_state
from biochem.rules import MassActionRule
from biochem.stochastic import PendingRelease, StochasticRunner


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


def delayed_model(delay: float = 1.0) -> ExperimentModel:
    species = tuple(Species(species_id) for species_id in ("E", "S", "ES", "P"))
    reactions = (
        Reaction("bind", {"E": 1, "S": 1}, {"ES": 1}, rate_constant=10.0),
        Reaction(
            "catalyze",
            {"ES": 1},
            {"E": 1, "P": 1},
            rate_constant=10.0,
            product_delays={"P": delay},
        ),
    )
    return ExperimentModel(
        species=species,
        reactions=reactions,
        initial_counts={"E": 1, "S": 3, "ES": 0, "P": 0},
    )


def test_delayed_product_has_identity_and_does_not_react_until_released():
    runner = StochasticRunner(delayed_model(), seed=4)
    state = runner.run(0.6)

    source = next(event for event in state.events if event.reaction_id == "catalyze")
    assert source.event_id == "event-000002"
    assert state.waiting_counts["P"] == 3
    assert sum(state.waiting_counts.values()) == 3
    assert state.counts["P"] == 0
    pending = state.pending_releases[0]
    assert pending.release_id == "release-000001"
    assert pending.source_event_id == source.event_id
    assert pending.products == {"P": 1}
    assert pending.condition_version == 0

    runner.run(2)
    release = next(event for event in state.events if event.event_type == "release")
    assert release.release_id == pending.release_id
    assert release.source_event_id == source.event_id
    assert release.changes == {"P": 1}
    assert state.counts["P"] == 3
    assert state.waiting_counts["P"] == 0
    assert state.pending_releases == []
    validate_trajectory_conservation(state)


def test_delayed_release_and_condition_share_instant_with_distinct_versions():
    runner = StochasticRunner(delayed_model(0.4), seed=4)
    state = runner.run(0.2)
    source = next(event for event in state.events if event.reaction_id == "catalyze")
    release_time = source.time + 0.4
    pending_release = next(
        release for release in state.pending_releases
        if release.scheduled_time == release_time
    )
    runner.schedule_condition_change(release_time, {"catalyze": 100.0})

    runner.run(release_time)
    release = next(
        event for event in state.events
        if event.event_type == "release" and event.release_id == pending_release.release_id
    )
    assert release.time == release_time
    assert release.condition_version == 0
    assert release.release_condition_version == 1
    assert state.condition_events[0].version == 1
    assert state.condition_version == 1
    runner.run(2)
    later_reactions = [
        event for event in state.events
        if event.event_type == "reaction" and event.time > release_time
    ]
    assert all(event.condition_version == 1 for event in later_reactions)
    validate_trajectory_conservation(state)


def test_checkpoints_around_multiple_simultaneous_releases_match_continuous(tmp_path):
    continuous = StochasticRunner(delayed_model(0.5), seed=8)
    continuous.run(3)
    source_events = [
        event for event in continuous.state.events if event.reaction_id == "catalyze"
    ]
    assert len(source_events) >= 2
    release_time = source_events[0].time + 0.5
    simultaneous = [
        event for event in continuous.state.events
        if event.event_type == "release" and event.time == release_time
    ]
    assert len(simultaneous) >= 1

    for index, checkpoint_time in enumerate(
        (release_time - 0.01, release_time, release_time + 0.01)
    ):
        runner = StochasticRunner(delayed_model(0.5), seed=8)
        runner.run(checkpoint_time)
        path = tmp_path / f"delay-{index}.json"
        save_state(runner.state, path)
        resumed = StochasticRunner.from_state(load_state(path))
        resumed.run(3)
        assert event_dicts(resumed.state) == event_dicts(continuous.state)
        assert resumed.state.waiting_counts == continuous.state.waiting_counts

    release_ids = [event.release_id for event in continuous.state.events if event.event_type == "release"]
    assert len(release_ids) == len(set(release_ids))


def test_multiple_releases_at_same_time_apply_each_identity_once():
    runner = StochasticRunner(delayed_model(), seed=1)
    runner.state.time = 0.5
    runner.state.counts = {"E": 0, "S": 0, "ES": 0, "P": 0}
    runner.state.waiting_counts = {"E": 0, "S": 0, "ES": 0, "P": 2}
    runner.state.pending_releases = [
        PendingRelease(
            release_id="release-000002",
            source_event_id="event-000002",
            reaction_id="catalyze",
            products={"P": 1},
            created_time=0.0,
            scheduled_time=1.0,
            condition_version=0,
        ),
        PendingRelease(
            release_id="release-000001",
            source_event_id="event-000001",
            reaction_id="catalyze",
            products={"P": 1},
            created_time=0.0,
            scheduled_time=1.0,
            condition_version=0,
        ),
    ]
    runner.state.pending_releases.sort(key=lambda release: (release.scheduled_time, release.release_id))

    runner.run(1.0)

    releases = [event for event in runner.state.events if event.event_type == "release"]
    assert [event.release_id for event in releases] == [
        "release-000001",
        "release-000002",
    ]
    assert [event.source_event_id for event in releases] == [
        "event-000001",
        "event-000002",
    ]
    assert runner.state.counts["P"] == 2
    assert runner.state.waiting_counts["P"] == 0
    assert runner.state.pending_releases == []


def test_branches_can_change_only_a_pending_release_time():
    base_runner = StochasticRunner(delayed_model(1.0), seed=4)
    base_runner.run(0.6)
    source = next(
        event for event in base_runner.state.events if event.reaction_id == "catalyze"
    )
    original_time = source.time + 1.0

    early = base_runner.fork()
    late = base_runner.fork()
    pending_id = early.state.pending_releases[0].release_id
    early.reschedule_pending_release(pending_id, original_time - 0.4)
    late.reschedule_pending_release(pending_id, original_time + 0.4)
    early.run(3)
    late.run(3)

    early_release = next(event for event in early.state.events if event.release_id == pending_id)
    late_release = next(event for event in late.state.events if event.release_id == pending_id)
    assert early_release.time == original_time - 0.4
    assert late_release.time == original_time + 0.4
    assert early_release.source_event_id == source.event_id
    assert late_release.source_event_id == source.event_id
    assert early.state.counts["P"] == late.state.counts["P"]
    validate_trajectory_conservation(early.state)
    validate_trajectory_conservation(late.state)


def _fate_runner(delay: float = 2.0, seed: int = 4):
    runner = StochasticRunner(delayed_model(delay), seed=seed)
    runner.run(0.6)
    return runner


def frozen_model() -> ExperimentModel:
    species = tuple(Species(species_id) for species_id in ("E", "S", "ES", "P"))
    reactions = (
        Reaction("bind", {"E": 1, "S": 1}, {"ES": 1}, rate_constant=0.0),
        Reaction(
            "catalyze",
            {"ES": 1},
            {"E": 1, "P": 1},
            rate_constant=0.0,
            product_delays={"P": 1.0},
        ),
    )
    return ExperimentModel(
        species=species,
        reactions=reactions,
        initial_counts={"E": 1, "S": 5, "ES": 0, "P": 0},
    )


def _two_batch_runner():
    """Two 2-unit waiting batches competing for the same release gate."""
    runner = StochasticRunner(frozen_model(), seed=1)
    runner.state.counts = {"E": 1, "S": 1, "ES": 0, "P": 0}
    runner.state.waiting_counts = {"E": 0, "S": 0, "ES": 0, "P": 4}
    runner.state.pending_releases = [
        PendingRelease(
            release_id="release-000001",
            source_event_id="event-000001",
            reaction_id="catalyze",
            products={"P": 2},
            created_time=0.0,
            scheduled_time=2.5,
            condition_version=0,
        ),
        PendingRelease(
            release_id="release-000002",
            source_event_id="event-000002",
            reaction_id="catalyze",
            products={"P": 2},
            created_time=0.0,
            scheduled_time=3.0,
            condition_version=0,
        ),
    ]
    runner.state.next_release_number = 3
    return runner


def test_partial_inactivation_splits_batch_and_keeps_identity():
    runner = _fate_runner()
    target = runner.state.pending_releases[0]
    runner.schedule_waiting_fate(
        1.0, "inactivate", release_id=target.release_id,
        products={"P": 1}, amount=1,
    )
    runner.run(3)
    state = runner.state

    fate = next(event for event in state.events if event.event_type == "inactivate")
    assert fate.changes == {"P": 1}
    assert fate.release_id == target.release_id
    assert fate.source_event_id == target.source_event_id
    assert fate.fate_id == "fate-000001"
    assert state.deactivated_counts["P"] == 1
    assert state.counts["P"] == 2
    assert state.waiting_counts["P"] == 0
    releases = [event for event in state.events if event.event_type == "release"]
    assert len({event.release_id for event in releases}) == 2
    assert all(event.source_event_id != target.source_event_id or True for event in releases)
    assert any(event.release_id != target.release_id for event in releases)
    assert state.fate_events[0].fate_type == "inactivate"
    validate_trajectory_conservation(state)


def test_full_inactivation_leaves_zero_remainder_without_second_release():
    runner = _fate_runner()
    target = runner.state.pending_releases[0]
    runner.schedule_waiting_fate(
        1.0, "inactivate", release_id=target.release_id, products={"P": 1}
    )
    runner.run(3)
    state = runner.state

    assert target.release_id not in {
        event.release_id for event in state.events if event.event_type == "release"
    }
    assert state.pending_releases == []
    assert state.deactivated_counts["P"] == 1
    assert state.counts["P"] == 2
    validate_trajectory_conservation(state)


def test_inactivation_request_more_than_waiting_fails_without_negative_counts():
    runner = _fate_runner()
    target = runner.state.pending_releases[0]
    runner.schedule_waiting_fate(
        1.0, "inactivate", release_id=target.release_id,
        products={"P": 1}, amount=99,
    )
    with pytest.raises(RuntimeError):
        runner.run(3)
    assert runner.state.status == "failed"
    assert all(count >= 0 for count in runner.state.waiting_counts.values())


def test_sequestered_lot_can_return_and_release_with_traceable_origin():
    runner = _fate_runner()
    target = runner.state.pending_releases[0]
    runner.schedule_waiting_fate(1.0, "sequester", release_id=target.release_id)
    runner.schedule_waiting_fate(2.5, "release_sequestered")
    runner.run(3.5)
    state = runner.state

    sequester_event = next(
        event for event in state.events if event.event_type == "sequester"
    )
    assert sequester_event.sequester_id == "sequester-000001"
    assert sequester_event.source_event_id == target.source_event_id
    assert state.sequestered_counts["P"] == 0
    return_event = next(
        event for event in state.events if event.event_type == "release_sequestered"
    )
    assert return_event.origin_release_id == target.release_id
    new_release = next(
        event
        for event in state.events
        if event.event_type == "release"
        and event.release_id == return_event.release_id
    )
    assert new_release.origin_release_id == target.release_id
    assert new_release.source_event_id == target.source_event_id
    assert state.counts["P"] == 3
    release_ids = [
        event.release_id for event in state.events if event.event_type == "release"
    ]
    assert len(release_ids) == len(set(release_ids))
    validate_trajectory_conservation(state)


def test_blocking_release_defers_until_allow_and_is_not_destruction():
    base = _fate_runner(delay=1.0)
    target = base.state.pending_releases[0]
    scheduled_time = target.scheduled_time

    continuous = base.fork()
    continuous.run(3)
    blocked = base.fork()
    blocked.schedule_waiting_fate(
        scheduled_time - 0.1, "block_releases", release_id=target.release_id
    )
    blocked.schedule_waiting_fate(
        2.2, "allow_releases", release_id=target.release_id
    )
    blocked.run(3)

    release = next(
        event
        for event in blocked.state.events
        if event.release_id == target.release_id and event.event_type == "release"
    )
    assert release.time == 2.2
    assert release.source_event_id == target.source_event_id
    assert blocked.state.counts["P"] == continuous.state.counts["P"] == 3
    assert blocked.state.deactivated_counts["P"] == 0
    assert blocked.state.pending_releases == []
    validate_trajectory_conservation(blocked.state)


def test_release_at_block_instant_precedes_block_and_cannot_be_recalled():
    runner = _fate_runner(delay=1.0)
    target = runner.state.pending_releases[0]
    release_time = target.scheduled_time
    runner.schedule_waiting_fate(release_time, "block_releases")
    runner.run(release_time)

    released = [
        event
        for event in runner.state.events
        if event.event_type == "release" and event.release_id == target.release_id
    ]
    assert len(released) == 1
    assert runner.state.releases_blocked is True
    # The already free product cannot be blocked away retroactively.
    assert runner.state.counts["P"] == released[0].changes["P"]


def test_competing_fates_match_across_checkpoints(tmp_path):
    def configured():
        runner = _fate_runner()
        ids = [release.release_id for release in runner.state.pending_releases]
        runner.schedule_waiting_fate(
            1.0, "inactivate", release_id=ids[0], products={"P": 1}, amount=1
        )
        runner.schedule_waiting_fate(1.0, "sequester", release_id=ids[1])
        runner.schedule_waiting_fate(1.2, "block_releases")
        runner.schedule_waiting_fate(2.4, "allow_releases")
        runner.schedule_waiting_fate(2.8, "release_sequestered")
        return runner

    continuous = configured()
    continuous.run(4)

    split = configured()
    for index, checkpoint_time in enumerate((0.8, 1.0, 1.2, 2.4, 2.8, 3.2)):
        split.run(checkpoint_time)
        path = tmp_path / f"fate-{index}.json"
        save_state(split.state, path)
        split = StochasticRunner.from_state(load_state(path))
    split.run(4)

    assert event_dicts(split.state) == event_dicts(continuous.state)
    assert split.state.counts == continuous.state.counts
    assert split.state.deactivated_counts == continuous.state.deactivated_counts
    assert split.state.sequestered_counts == continuous.state.sequestered_counts
    validate_trajectory_conservation(split.state)


def test_fate_branches_keep_their_own_conditions_at_shared_instant():
    base = _fate_runner(delay=1.0)
    target = base.state.pending_releases[0]
    release_time = target.scheduled_time

    keep = base.fork()
    stop = base.fork()
    stop.schedule_waiting_fate(
        release_time - 0.1, "block_releases", release_id=target.release_id
    )
    stop.schedule_waiting_fate(release_time, "inactivate", release_id=target.release_id)
    keep.run(2.5)
    stop.run(2.5)

    keep_release = next(
        event
        for event in keep.state.events
        if event.release_id == target.release_id and event.event_type == "release"
    )
    assert keep_release.time == release_time
    stop_events = [
        event
        for event in stop.state.events
        if event.release_id == target.release_id and event.event_type == "release"
    ]
    assert stop_events == []
    assert any(
        event.event_type == "inactivate"
        and event.release_id == target.release_id
        for event in stop.state.events
    )
    assert keep.state.counts["P"] != stop.state.counts["P"]
    assert stop.state.deactivated_counts["P"] >= 1
    validate_trajectory_conservation(keep.state)
    validate_trajectory_conservation(stop.state)


def test_two_batches_partial_fates_and_gate_cycle_release_remainders():
    runner = _two_batch_runner()
    runner.schedule_waiting_fate(
        1.0, "inactivate", release_id="release-000001", products={"P": 1}, amount=1
    )
    runner.schedule_waiting_fate(
        1.5, "convert", release_id="release-000002", products={"P": 1}, amount=1
    )
    runner.schedule_waiting_fate(2.0, "block_releases")
    runner.schedule_waiting_fate(4.0, "allow_releases")

    runner.run(2.0)
    occupancy = runner.release_resource_occupancy()
    assert occupancy["releases_blocked"] is True
    assert [batch["release_id"] for batch in occupancy["batches"]] == [
        "release-000001",
        "release-000002",
    ]
    assert all(batch["blocked"] for batch in occupancy["batches"])
    assert all(batch["products"] == {"P": 1} for batch in occupancy["batches"])
    assert occupancy["waiting_counts"]["P"] == 2
    assert occupancy["deactivated_counts"]["P"] == 1
    assert occupancy["converted_counts"]["P"] == 1

    runner.run(5)
    state = runner.state
    inactivate = next(event for event in state.events if event.event_type == "inactivate")
    assert inactivate.changes == {"P": 1}
    assert inactivate.release_id == "release-000001"
    assert inactivate.source_event_id == "event-000001"
    convert = next(event for event in state.events if event.event_type == "convert")
    assert convert.changes == {"P": 1}
    assert convert.release_id == "release-000002"
    assert convert.source_event_id == "event-000002"

    releases = {
        event.release_id: event
        for event in state.events
        if event.event_type == "release"
    }
    assert set(releases) == {"release-000001", "release-000002"}
    assert releases["release-000001"].changes == {"P": 1}
    assert releases["release-000001"].source_event_id == "event-000001"
    assert releases["release-000002"].changes == {"P": 1}
    assert releases["release-000002"].source_event_id == "event-000002"
    assert all(event.time == 4.0 for event in releases.values())

    assert state.waiting_counts["P"] == 0
    assert state.counts["P"] == 2
    assert state.deactivated_counts["P"] == 1
    assert state.converted_counts["P"] == 1
    assert state.cancelled_counts["P"] == 0
    assert state.pending_releases == []
    validate_trajectory_conservation(state)


def test_convert_and_gate_cycle_match_across_checkpoints(tmp_path):
    def configured():
        runner = _two_batch_runner()
        runner.schedule_waiting_fate(
            1.0, "inactivate", release_id="release-000001", products={"P": 1}, amount=1
        )
        runner.schedule_waiting_fate(
            1.5, "convert", release_id="release-000002", products={"P": 1}, amount=1
        )
        runner.schedule_waiting_fate(2.0, "block_releases")
        runner.schedule_waiting_fate(4.0, "allow_releases")
        return runner

    continuous = configured()
    continuous.run(5)

    split = configured()
    for index, checkpoint_time in enumerate((0.5, 1.0, 1.5, 2.0, 3.1, 4.0, 4.6)):
        split.run(checkpoint_time)
        path = tmp_path / f"convert-{index}.json"
        save_state(split.state, path)
        split = StochasticRunner.from_state(load_state(path))
    split.run(5)

    assert event_dicts(split.state) == event_dicts(continuous.state)
    assert split.state.counts == continuous.state.counts
    assert split.state.waiting_counts == continuous.state.waiting_counts
    assert split.state.deactivated_counts == continuous.state.deactivated_counts
    assert split.state.converted_counts == continuous.state.converted_counts
    assert split.state.cancelled_counts == continuous.state.cancelled_counts
    validate_trajectory_conservation(split.state)


def test_cancel_release_skips_batch_without_undoing_source_event():
    runner = _fate_runner()
    target = runner.state.pending_releases[1]
    runner.schedule_waiting_fate(1.0, "cancel_release", release_id=target.release_id)
    runner.schedule_waiting_fate(1.5, "block_releases")
    runner.schedule_waiting_fate(3.0, "allow_releases")
    runner.run(4)
    state = runner.state

    cancel = next(event for event in state.events if event.event_type == "cancel_release")
    assert cancel.changes == {"P": 1}
    assert cancel.release_id == target.release_id
    assert cancel.source_event_id == target.source_event_id
    assert state.cancelled_counts["P"] == 1
    # Cancelling the release does not undo the source chemical event.
    assert any(
        event.event_type == "reaction" and event.event_id == target.source_event_id
        for event in state.events
    )
    releases = [event for event in state.events if event.event_type == "release"]
    assert target.release_id not in {event.release_id for event in releases}
    assert len(releases) == 2
    assert all(event.time == 3.0 for event in releases)
    release_ids = [event.release_id for event in releases]
    assert len(release_ids) == len(set(release_ids))
    assert state.counts["P"] == 2
    assert state.pending_releases == []
    validate_trajectory_conservation(state)


def test_partial_cancel_keeps_remainder_on_release_track():
    runner = _two_batch_runner()
    runner.schedule_waiting_fate(
        1.0, "cancel_release", release_id="release-000001", products={"P": 1}, amount=1
    )
    runner.run(5)
    state = runner.state

    assert state.cancelled_counts["P"] == 1
    releases = {
        event.release_id: event
        for event in state.events
        if event.event_type == "release"
    }
    assert releases["release-000001"].changes == {"P": 1}
    assert releases["release-000001"].source_event_id == "event-000001"
    assert releases["release-000002"].changes == {"P": 2}
    assert state.counts["P"] == 3
    validate_trajectory_conservation(state)


def test_cancel_unknown_or_released_batch_fails():
    runner = _fate_runner()
    runner.schedule_waiting_fate(1.0, "cancel_release", release_id="release-999999")
    with pytest.raises(RuntimeError):
        runner.run(2)
    assert runner.state.status == "failed"
    assert all(count >= 0 for count in runner.state.waiting_counts.values())
    assert all(count >= 0 for count in runner.state.cancelled_counts.values())

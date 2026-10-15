import pytest
from exl3_mlx.process_identity import (
    PS_CAP,
    birth_time,
    validate_terminal_process_table,
)

OLD = "Wed Oct 14 00:34:34 2026"
NEW = "Wed Oct 14 19:53:01 2026"
LATER = "Wed Oct 14 19:54:01 2026"


def row(pid, birth=NEW, ppid=1, pgid=1):
    return f"{pid} {ppid} {pgid} {birth} <defunct>\n".encode()


def test_later_generation_is_reported_without_mutation():
    required = {32069: [OLD], 47866: []}
    value = validate_terminal_process_table(row(32069), required)
    assert value["all_recorded_generations_absent"] is True
    assert value["later_generation_pid_reuse"][0]["pid"] == 32069
    assert required == {32069: [OLD], 47866: []}


@pytest.mark.parametrize(
    "births,current",
    [
        ([OLD], OLD),
        ([NEW], OLD),
        ([], NEW),
        ([OLD, LATER], NEW),
        (["unknown"], NEW),
        ([OLD], "unknown"),
    ],
)
def test_live_or_ambiguous_generation_refuses(births, current):
    with pytest.raises(ValueError):
        validate_terminal_process_table(row(32069, current), {32069: births})


@pytest.mark.parametrize("ppid,pgid", [(32069, 2), (1, 32069)])
def test_parent_group_requires_present_later_holder(ppid, pgid):
    with pytest.raises(ValueError):
        validate_terminal_process_table(row(44, LATER, ppid, pgid), {32069: [OLD]})
    value = validate_terminal_process_table(
        row(32069) + row(44, LATER, ppid, pgid), {32069: [OLD]}
    )
    assert len(value["later_generation_relations"]) == 1
    with pytest.raises(ValueError):
        validate_terminal_process_table(
            row(32069) + row(44, OLD, ppid, pgid), {32069: [OLD]}
        )


@pytest.mark.parametrize(
    "required",
    [
        {},
        {True: [OLD]},
        {"32069": [OLD]},
        {32069: [OLD, OLD]},
        {32069: [{}]},
        {32069: (OLD,)},
        {0: []},
    ],
)
def test_invalid_required_map_refuses(required):
    with pytest.raises(ValueError):
        validate_terminal_process_table(row(32069), required)


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"bad\n",
        row(32069) * 2,
        b"x" * (PS_CAP + 1),
        b"\xff",
        row(32069, "Tue Oct 14 19:53:01 2026"),
    ],
)
def test_malformed_bounded_snapshot_refuses(raw):
    with pytest.raises(ValueError):
        validate_terminal_process_table(raw, {32069: [OLD]})


def test_space_padded_day_and_second_snapshot_reappearance():
    assert birth_time("Thu Oct  1 05:46:30 2026").day == 1
    validate_terminal_process_table(row(32069), {32069: [OLD]})
    with pytest.raises(ValueError):
        validate_terminal_process_table(row(32069, OLD), {32069: [OLD]})


def test_report_snapshots_birth_lists_independently_of_caller():
    required = {32069: [OLD], 47866: []}
    value = validate_terminal_process_table(row(32069), required)
    required[32069].append(NEW)
    required[47866].append(OLD)
    assert value["recorded_generations"] == {"32069": [OLD], "47866": []}
    assert value["later_generation_pid_reuse"][0]["recorded_births"] == [OLD]
    # The two report fields are independent snapshots, too.
    value["recorded_generations"]["32069"].append(LATER)
    assert value["later_generation_pid_reuse"][0]["recorded_births"] == [OLD]
    # A second check receives current caller facts and must reject the now-live birth.
    with pytest.raises(ValueError):
        validate_terminal_process_table(row(32069), required)

"""Pure CPU process-generation closure for caller-supplied, bounded ps snapshots.

Birth strings must come from the same host and time convention. Numeric PID reuse
alone does not identify a surviving process. Missing or ambiguous generations and
unproved parent/group relationships fail closed. This is finite evidence, not
continuous process attestation or a reason to stop unrelated processes.
"""

import datetime

PS_CAP = 4 * 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def birth_time(value):
    require(type(value) is str and value, "Missing process birth")
    try:
        result = datetime.datetime.strptime(value, "%a %b %d %H:%M:%S %Y")
    except (ValueError, TypeError) as exc:
        raise ValueError("Malformed process birth") from exc
    parts = value.split()
    normalized = result.strftime("%a %b %d %H:%M:%S %Y").split()
    require(
        len(parts) == 5
        and parts[:2] == normalized[:2]
        and int(parts[2]) == result.day
        and parts[3:] == normalized[3:],
        "Ambiguous process birth",
    )
    return result


def validate_terminal_process_table(raw, required):
    """Validate supplied ps bytes against {pid: [recorded birth strings]}.

    An empty list means the birth is unknown: that numeric PID must be absent.
    The caller must supply every recorded generation, including the controller,
    and check a second fresh snapshot immediately before publishing evidence.
    This function never queries, signals, starts, or stops processes.
    """
    require(type(raw) is bytes, "Process table must be bytes")
    require(len(raw) <= PS_CAP, "Process table bound")
    require(
        type(required) is dict and required, "Recorded process generations required"
    )
    known = {}
    for pid, births in required.items():
        require(
            type(pid) is int and pid > 0 and type(births) is list,
            "Recorded process identity",
        )
        require(all(type(x) is str for x in births), "Recorded births must be strings")
        require(len(births) == len(set(births)), "Duplicate recorded process birth")
        known[pid] = [birth_time(x) for x in births]
    rows = {}
    for line in raw.decode("utf-8", "strict").splitlines():
        words = line.split(None, 8)
        require(len(words) == 9, "Malformed process row")
        pid, ppid, pgid = map(int, words[:3])
        require(
            pid > 0 and ppid >= 0 and pgid >= 0 and pid not in rows,
            "Process table identity",
        )
        start = " ".join(words[3:8])
        rows[pid] = {
            "ppid": ppid,
            "pgid": pgid,
            "start": start,
            "birth": birth_time(start),
            "command": words[8],
        }
    require(rows, "Empty process table")
    reused = {}
    for pid, births in known.items():
        if pid not in rows:
            continue
        current = rows[pid]
        require(
            births and current["birth"] > max(births),
            "Recorded training generation still present or birth uncertain",
        )
        reused[pid] = {
            "pid": pid,
            "recorded_births": list(required[pid]),
            "current_birth": current["start"],
            "command": current["command"],
        }
    relations = []
    for pid, row in rows.items():
        for field in ("ppid", "pgid"):
            holder = row[field]
            if holder not in known:
                continue
            # A bare old numeric group/parent is not evidence of a new generation.
            # Require its current holder and a temporally compatible related row.
            require(
                holder in reused and row["birth"] >= rows[holder]["birth"],
                "Recorded descendant/group remains or generation uncertain",
            )
            relations.append(
                {
                    "pid": pid,
                    "relation": field,
                    "holder_pid": holder,
                    "current_birth": row["start"],
                    "holder_birth": rows[holder]["start"],
                }
            )
    return {
        "schema": "process-generation-terminal-observation-v1",
        "recorded_pids": sorted(required),
        "recorded_generations": {str(p): list(required[p]) for p in sorted(required)},
        "rows_observed": len(rows),
        "all_recorded_generations_absent": True,
        "later_generation_pid_reuse": [reused[p] for p in sorted(reused)],
        "later_generation_relations": relations,
        "scope": "One supplied finite host-local snapshot distinguishes later PID reuse; caller must collect and validate a second fresh snapshot before publication; missing, equal, earlier or ambiguous births/relations block. No signals or exhaustive daemonization attestation.",
    }

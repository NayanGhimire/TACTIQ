"""Attacking analytics module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..schemas import _get_zone, _get_channel


@dataclass
class AttackSequence:
    """A single attacking sequence ending in a shot or chance."""
    start_event_id: str
    shot_event_id: Optional[str]
    final_third_entries: int
    box_entries: int
    channels_used: List[str]
    passed_into_box: bool
    resulted_in_shot: bool
    resulted_in_goal: bool
    total_events: int


def calculate_attack_patterns(
    events: List[Dict[str, Any]],
    team_id: str,
) -> Dict[str, Any]:
    """Calculate attacking patterns from events for a given team.

    Tracks final third entries, box entries, shots, crosses, through balls,
    and channel distribution.
    """
    final_third_entries = 0
    box_entries = 0
    shots = []
    crosses = []
    cutbacks = []
    shots_by_channel = {}
    shots_by_zone = {}
    attacking_events = []

    in_final_third = False
    in_box = False

    for ev in events:
        etype = ev.get("event_type", "")
        x0 = ev.get("start_x", 0)
        y0 = ev.get("start_y", 0)
        x1 = ev.get("end_x", 0)
        y1 = ev.get("end_y", 0)
        zone = _get_zone(x1)
        channel = _get_channel(y1)

        if etype == "pass":
            attacking_events.append(ev)

        # Track final third entries
        if zone == "attacking_third" and not in_final_third:
            final_third_entries += 1
            in_final_third = True
        elif zone != "attacking_third":
            in_final_third = False

        # Track box entries (approx: attacking third inside central-ish)
        if zone == "attacking_third" and 45 <= y1 <= 65 and not in_box:
            box_entries += 1
            in_box = True
        elif not (zone == "attacking_third" and 45 <= y1 <= 65):
            in_box = False

        # Shots
        if etype == "shot":
            shots.append({
                "event_id": ev.get("event_id", ""),
                "x_start": x0,
                "y_start": y0,
                "channel": channel,
                "zone": zone,
                "outcome": ev.get("outcome", "unknown"),
                "completed": bool(ev.get("completed", False)),
            })

        # Crosses
        if etype == "cross":
            crosses.append({
                "event_id": ev.get("event_id", ""),
                "channel": channel,
                "completed": bool(ev.get("completed", False)),
            })

    total_shots = len(shots)
    shots_on_target = sum(1 for s in shots if s["outcome"] == "on_target")
    shots_off_target = sum(1 for s in shots if s["outcome"] == "off_target")
    goals = sum(1 for s in shots if s.get("completed", False) and s["outcome"] == "goal")

    for s in shots:
        ch = s["channel"]
        shots_by_channel[ch] = shots_by_channel.get(ch, 0) + 1
        zn = s["zone"]
        shots_by_zone[zn] = shots_by_zone.get(zn, 0) + 1

    return {
        "team_id": team_id,
        "final_third_entries": final_third_entries,
        "box_entries": box_entries,
        "total_shots": total_shots,
        "shots_on_target": shots_on_target,
        "shots_off_target": shots_off_target,
        "goals": goals,
        "shot_conversion_rate": round(goals / total_shots, 2) if total_shots > 0 else 0.0,
        "shots_by_channel": shots_by_channel,
        "shots_by_zone": shots_by_zone,
        "crosses": len(crosses),
        "cutbacks": len(cutbacks),
        "shots": shots,
        "cross_events": crosses,
        "attacking_events_count": len(attacking_events),
    }
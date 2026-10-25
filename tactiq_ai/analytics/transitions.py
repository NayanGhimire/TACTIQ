"""Transitions and game rhythm analytics module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..schemas import _get_zone


@dataclass
class TransitionSequence:
    """A transition sequence: from possession change to forward progress."""
    possession_change_event: str
    recovery_event: Optional[str]
    final_event: str
    duration_seconds: float
    forward_progression_m: float
    speed_mps: float
    passes_used: int
    resulted_in_shot: bool
    resulted_in_goal: bool


def calculate_transitions(
    events: List[Dict[str, Any]],
    team_id: str,
) -> Dict[str, Any]:
    """Calculate transition and game rhythm metrics.

    Tracks:
    - Time between winning possession and progressing forward
    - Counter-pressing activity
    - Possession duration
    - Tempo changes
    """
    possession_changes = []
    transitions = []
    possession_durations = []
    counter_presses = 0
    dangerous_turnovers = 0

    current_possession_team = None
    possession_start_ts = None
    possession_start_event = None

    for ev in events:
        etype = ev.get("event_type", "")
        ts = ev.get("timestamp_seconds", 0)
        x0 = ev.get("start_x", 0)
        x1 = ev.get("end_x", 0)
        y0 = ev.get("start_y", 0)
        y1 = ev.get("end_y", 0)

        # Track possession changes
        if etype in ("tackle", "recovery", "ball_loss", "interception", "clearance"):
            if etype in ("tackle", "recovery", "interception"):
                # Possession change to this team
                if current_possession_team != team_id:
                    possession_changes.append({
                        "event_id": ev.get("event_id", ""),
                        "timestamp": ts,
                        "type": etype,
                        "x": x1,
                        "y": y1,
                        "zone": _get_zone(x1),
                    })
                    current_possession_team = team_id
                    possession_start_ts = ts
                    possession_start_event = ev.get("event_id", "")
                else:
                    # Already had possession
                    pass
            elif etype == "ball_loss":
                # Possession lost
                possession_changes.append({
                    "event_id": ev.get("event_id", ""),
                    "timestamp": ts,
                    "type": "ball_loss",
                    "x": x1,
                    "y": y1,
                    "zone": _get_zone(x1),
                })
                current_possession_team = None

        # Track possession duration
        if possession_start_ts is not None and current_possession_team == team_id:
            possession_durations.append(ts - possession_start_ts)

        # Counter-press: pressure immediately after losing possession
        if etype == "pressure" and current_possession_team != team_id:
            # Check if recent possession change
            recent_changes = [c for c in possession_changes
                            if ts - c["timestamp"] < 5.0]
            if recent_changes and recent_changes[-1]["type"] == "ball_loss":
                counter_presses += 1

        # Dangerous turnovers: opponent in final third after our turnover
        if etype in ("tackle", "recovery") and _get_zone(x1) == "attacking_third":
            dangerous_turnovers += 1

    total_possessions = len(possession_changes)
    avg_possession_duration = round(sum(possession_durations) / max(len(possession_durations), 1), 2)

    # Tempo: average time between events
    if len(events) >= 2:
        timestamps = [e["timestamp_seconds"] for e in events]
        avg_time_between_events = round(sum(t2 - t1 for t1, t2 in zip(timestamps, timestamps[1:])) / (len(timestamps) - 1), 2)
    else:
        avg_time_between_events = 0.0

    return {
        "team_id": team_id,
        "total_possessions": total_possessions,
        "avg_possession_duration_s": avg_possession_duration,
        "counter_presses": counter_presses,
        "dangerous_turnovers": dangerous_turnovers,
        "avg_time_between_events_s": avg_time_between_events,
        "possession_changes": possession_changes,
        "transitions": transitions,
    }
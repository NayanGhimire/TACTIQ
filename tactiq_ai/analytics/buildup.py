"""Build-up analytics module.

Calculates and returns structured data for how teams build up from defence,
using event data from the MatchProcessor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import PITCH_LENGTH_M, PITCH_WIDTH_M
from ..schemas import _get_zone, _get_channel


@dataclass
class PassSegment:
    """A single pass in a build-up sequence."""
    event_id: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    event_type: str
    completed: bool
    under_pressure: bool
    zone: str  # defensive_third / middle_third / attacking_third
    channel: str  # left_wide / left_halfspace / central / right_halfspace / right_wide


def calculate_build_up_segments(
    events: List[Dict[str, Any]],
    team_id: str,
) -> Dict[str, Any]:
    """Calculate build-up play metrics from a list of events for a given team.

    Focuses on sequences starting in the defensive third.

    Returns dict with:
    - segments: list of PassSegment objects
    - third_progression: how many segments crossed third boundaries
    - completion_rate: overall pass completion
    - pressure_stats: how often build-up occurred under pressure
    - turnover_stats: how many turnovers during build-up
    """
    segments = []
    defensive_third_starts = 0
    third_crossings = 0
    passes_in_build_up = 0
    completed_in_build_up = 0
    pressure_build_up = 0
    turnovers_in_build_up = 0
    in_defensive_third = True

    for ev in events:
        if ev.get("event_type") not in ("pass", "carry"):
            continue

        # Normalise coordinates to 0-100 frame
        x0 = ev.get("start_x", 0)
        y0 = ev.get("start_y", 0)
        x1 = ev.get("end_x", 0)
        y1 = ev.get("end_y", 0)

        zone = _get_zone(x1)
        channel = _get_channel(y1)

        if zone == "defensive_third" and in_defensive_third:
            defensive_third_starts += 1
            in_defensive_third = True  # reset after counting

        passes_in_build_up += 1
        completed_in_build_up += int(ev.get("completed", False))
        pressure_build_up += int(ev.get("under_pressure", False))
        turnovers_in_build_up += int(ev.get("turnover", False))

        seg = PassSegment(
            event_id=ev.get("event_id", "unknown"),
            start_x=x0,
            start_y=y0,
            end_x=x1,
            end_y=y1,
            event_type=ev.get("event_type", "pass"),
            completed=bool(ev.get("completed", False)),
            under_pressure=bool(ev.get("under_pressure", False)),
            zone=zone,
            channel=channel,
        )
        segments.append(seg)

        # Track third crossings
        prev_zone = _get_zone(x0) if x0 else "unknown"
        if prev_zone == "defensive_third" and zone != "defensive_third":
            third_crossings += 1

    total_passes = passes_in_build_up
    completion_rate = round(completed_in_build_up / total_passes, 2) if total_passes > 0 else 0.0

    return {
        "segments": segments,
        "defensive_third_starts": defensive_third_starts,
        "third_crossings": third_crossings,
        "total_passes": total_passes,
        "completed_passes": completed_in_build_up,
        "completion_rate": completion_rate,
        "pressure_build_up": pressure_build_up,
        "turnovers_in_build_up": turnovers_in_build_up,
        "team_id": team_id,
    }


@dataclass
class BuildUpProfile:
    """Overall build-up profile for a team."""
    team_id: str
    total_sequences: int
    avg_passes_per_sequence: float
    avg_completion_rate: float
    avg_third_crossings: float
    common_zones: Dict[str, int]  # zone -> count
    common_channels: Dict[str, int]  # channel -> count
    pressure_frequency: float  # fraction of passes under pressure


def build_profile_from_sequences(
    sequences: Dict[str, List[Dict[str, Any]]],
    team_ids: Dict[str, str],
) -> Dict[str, BuildUpProfile]:
    """Build build-up profiles from dict of event sequences keyed by team."""
    profiles = {}
    for team_key, events in sequences.items():
        tid = team_ids.get(team_key, team_key)
        data = calculate_build_up_segments(events, tid)

        all_zones = {}
        all_channels = {}
        total_passes = data["total_passes"]

        for seg in data["segments"]:
            all_zones[seg.zone] = all_zones.get(seg.zone, 0) + 1
            all_channels[seg.channel] = all_channels.get(seg.channel, 0) + 1

        n_sequences = len([e for e in events if e.get("event_type") in ("pass", "carry")])

        profiles[team_key] = BuildUpProfile(
            team_id=tid,
            total_sequences=n_sequences,
            avg_passes_per_sequence=round(data["total_passes"] / max(n_sequences, 1), 1),
            avg_completion_rate=data["completion_rate"],
            avg_third_crossings=round(data["third_crossings"] / max(n_sequences, 1), 2),
            common_zones=dict(sorted(all_zones.items(), key=lambda x: -x[1])[:3]),
            common_channels=dict(sorted(all_channels.items(), key=lambda x: -x[1])[:3]),
            pressure_frequency=round(data["pressure_build_up"] / max(total_passes, 1), 3),
        )
    return profiles
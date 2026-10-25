"""Defensive analytics module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from ..schemas import _get_zone, _get_channel


@dataclass
class DefensiveAction:
    """A single defensive action."""
    event_id: str
    event_type: str  # tackle, interception, clearance, recovery, pressure
    x: float
    y: float
    zone: str
    channel: str
    successful: bool
    opponent_player: str


def calculate_defensive_behaviour(
    events: List[Dict[str, Any]],
    team_id: str,
) -> Dict[str, Any]:
    """Calculate defensive behaviour metrics from events for a given team.

    Focuses on pressure events, recoveries, tackles, interceptions, clearances.
    """
    pressures = []
    tackles = []
    recoveries = []
    interceptions = []
    clearances = []

    for ev in events:
        etype = ev.get("event_type", "")
        x = ev.get("end_x", 0)
        y = ev.get("end_y", 0)
        zone = _get_zone(x)
        channel = _get_channel(y)

        if etype == "pressure":
            pressures.append({
                "event_id": ev.get("event_id", ""),
                "x": x,
                "y": y,
                "zone": zone,
                "channel": channel,
                "resulted_in_turnover": bool(ev.get("turnover", False)),
            })
        elif etype == "tackle":
            tackles.append({
                "event_id": ev.get("event_id", ""),
                "x": x,
                "y": y,
                "zone": zone,
                "channel": channel,
                "won": bool(ev.get("completed", False)),
            })
        elif etype == "recovery":
            recoveries.append({
                "event_id": ev.get("event_id", ""),
                "x": x,
                "y": y,
                "zone": zone,
                "channel": channel,
            })
        elif etype == "interception":
            interceptions.append({
                "event_id": ev.get("event_id", ""),
                "x": x,
                "y": y,
                "zone": zone,
                "channel": channel,
            })
        elif etype == "clearance":
            clearances.append({
                "event_id": ev.get("event_id", ""),
                "x": x,
                "y": y,
                "zone": zone,
                "channel": channel,
            })

    total_defensive_actions = (
        len(pressures) + len(tackles) + len(recoveries) +
        len(interceptions) + len(clearances)
    )

    # Tackle success rate
    tackle_success = sum(1 for t in tackles if t["won"])
    tackle_success_rate = round(tackle_success / len(tackles), 2) if tackles else 0.0

    # Pressure success rate (turnover)
    pressure_success = sum(1 for p in pressures if p["resulted_in_turnover"])
    pressure_success_rate = round(pressure_success / len(pressures), 2) if pressures else 0.0

    # Zone distribution
    zone_counts = {}
    for lst in [pressures, tackles, recoveries, interceptions, clearances]:
        for a in lst:
            z = a["zone"]
            zone_counts[z] = zone_counts.get(z, 0) + 1

    channel_counts = {}
    for lst in [pressures, tackles, recoveries, interceptions, clearances]:
        for a in lst:
            c = a["channel"]
            channel_counts[c] = channel_counts.get(c, 0) + 1

    return {
        "team_id": team_id,
        "total_defensive_actions": total_defensive_actions,
        "pressures": len(pressures),
        "tackles": len(tackles),
        "recoveries": len(recoveries),
        "interceptions": len(interceptions),
        "clearances": len(clearances),
        "tackle_success_rate": tackle_success_rate,
        "pressure_success_rate": pressure_success_rate,
        "zone_distribution": zone_counts,
        "channel_distribution": channel_counts,
        "defensive_actions": {
            "pressures": pressures,
            "tackles": tackles,
            "recoveries": recoveries,
            "interceptions": interceptions,
            "clearances": clearances,
        },
    }
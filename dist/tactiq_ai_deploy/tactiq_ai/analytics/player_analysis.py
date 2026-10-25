"""Player intelligence analytics module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..schemas import _get_zone, _get_channel


@dataclass
class PlayerMetrics:
    """Comprehensive metrics for a single player."""
    player_id: str
    player_name: str
    team: str
    position: str

    # Involvement counts
    total_passes: int = 0
    completed_passes: int = 0
    attempted_passes: int = 0
    total_carries: int = 0
    successful_carries: int = 0
    total_shots: int = 0
    shots_on_target: int = 0
    goals: int = 0
    total_crosses: int = 0
    completed_crosses: int = 0
    pressures_applied: int = 0
    tackles: int = 0
    tackles_won: int = 0
    recoveries: int = 0
    interceptions: int = 0

    # Progression metrics
    progressive_passes: int = 0
    progressive_carries: int = 0
    final_third_passes: int = 0
    box_passes: int = 0

    # Possession
    possession_won: int = 0
    possession_lost: int = 0

    # Time on pitch
    minutes_played: float = 0.0

    # Advanced metrics
    pass_completion_rate: float = 0.0
    carry_success_rate: float = 0.0
    shot_conversion_rate: float = 0.0

    def overall_involvement_score(self) -> float:
        """Simple composite involvement score."""
        w = 0.3  # passing involvement
        c = 0.2  # carry involvement
        s = 0.15  # shooting contribution
        d = 0.15  # defensive contribution
        p = 0.2   # pressing/contribution
        return round(
            w * self.pass_completion_rate +
            c * self.carry_success_rate +
            s * self.shot_conversion_rate +
            d * (self.tackles_won / max(self.tackles, 1)) +
            p * (self.pressures_applied / max(1, self.pressures_applied)),
            3
        )


def calculate_player_metrics_from_events(
    events: List[Dict[str, Any]],
    player_id: str,
    player_name: str,
    team: str,
    position: str,
) -> PlayerMetrics:
    """Calculate comprehensive metrics for a single player from events."""
    metrics = PlayerMetrics(
        player_id=player_id,
        player_name=player_name,
        team=team,
        position=position,
    )

    for ev in events:
        etype = ev.get("event_type", "")
        p_ev_id = ev.get("player_id", "")

        if p_ev_id != player_id:
            continue

        # Passes
        if etype == "pass":
            metrics.total_passes += 1
            if ev.get("completed", False):
                metrics.completed_passes += 1

        # Carries
        elif etype == "carry":
            metrics.total_carries += 1
            if ev.get("completed", False):
                metrics.successful_carries += 1

        # Shots
        elif etype == "shot":
            metrics.total_shots += 1
            if ev.get("completed", False) and ev.get("outcome") == "on_target":
                metrics.shots_on_target += 1
            if ev.get("goal", False):
                metrics.goals += 1

        # Crosses
        elif etype == "cross":
            metrics.total_crosses += 1
            if ev.get("completed", False):
                metrics.completed_crosses += 1

        # Pressure
        elif etype == "pressure":
            metrics.pressures_applied += 1

        # Tackles
        elif etype == "tackle":
            metrics.tackles += 1
            if ev.get("completed", False):
                metrics.tackles_won += 1

        # Recoveries
        elif etype == "recovery":
            metrics.recoveries += 1

        # Interceptions
        elif etype == "interception":
            metrics.interceptions += 1

    # Calculate rates
    if metrics.total_passes > 0:
        metrics.pass_completion_rate = round(
            metrics.completed_passes / metrics.total_passes, 3)
    else:
        metrics.pass_completion_rate = 0.0

    if metrics.total_carries > 0:
        metrics.carry_success_rate = round(
            metrics.successful_carries / metrics.total_carries, 3)
    else:
        metrics.carry_success_rate = 0.0

    if metrics.total_shots > 0:
        metrics.shot_conversion_rate = round(
            metrics.goals / metrics.total_shots, 3)
    else:
        metrics.shot_conversion_rate = 0.0

    return metrics


def build_player_focus_profile(
    all_events: List[Dict[str, Any]],
    selected_player_id: str,
) -> Dict[str, Any]:
    """Build a player-focused profile following one player through the match."""
    # Find all events involving the selected player
    player_events = [e for e in all_events if e.get("player_id") == selected_player_id]

    # Get player basic info from first event
    if not player_events:
        return {"error": "No events found for selected player"}

    first_ev = player_events[0]
    team = first_ev.get("team_id", "home")
    name = first_ev.get("player_name", selected_player_id)
    position = first_ev.get("position", "unknown")

    # Calculate metrics
    metrics = calculate_player_metrics_from_events(
        all_events, selected_player_id, name, team, position
    )

    # Calculate zone/channel distribution
    zone_dist = {}
    channel_dist = {}
    for ev in player_events:
        x = ev.get("end_x", 0)
        y = ev.get("end_y", 0)
        z = _get_zone(x)
        c = _get_channel(y)
        zone_dist[z] = zone_dist.get(z, 0) + 1
        channel_dist[c] = channel_dist.get(c, 0) + 1

    # Get events by type
    events_by_type = {}
    for etype in ["pass", "carry", "shot", "cross", "pressure", "tackle", "recovery"]:
        type_events = [e for e in player_events if e.get("event_type") == etype]
        events_by_type[etype] = {
            "count": len(type_events),
            "completed": sum(1 for e in type_events if e.get("completed", False)),
        }

    return {
        "player_id": selected_player_id,
        "player_name": name,
        "team": team,
        "position": position,
        "metrics": metrics,
        "zone_distribution": zone_dist,
        "channel_distribution": channel_dist,
        "events_by_type": events_by_type,
        "total_player_events": len(player_events),
        "match_coverage_minutes": round(
            max((e["timestamp_seconds"] for e in player_events), default=0) / 60, 1),
    }
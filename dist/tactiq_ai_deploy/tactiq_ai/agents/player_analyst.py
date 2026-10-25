"""TACTIQ AI - Player Analyst Agent.

Agent 4: Tracks individual player involvement and contributions throughout the match.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..analytics.player_analysis import calculate_player_metrics_from_events, build_player_focus_profile

logger = logging.getLogger(__name__)


class PlayerAnalystAgent:
    """Agent 4: tracks individual involvement and identifies notable contributions."""

    def analyze(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze player-level metrics across the event stream."""
        players_seen = {}
        for ev in events:
            pid = ev.get("player_id", "")
            name = ev.get("player_name", "")
            team = ev.get("team_id", "")
            position = ev.get("position", "")
            if pid and pid not in players_seen:
                players_seen[pid] = {
                    "player_id": pid,
                    "player_name": name,
                    "team": team,
                    "position": position,
                    "events_by_type": {},
                    "metrics": None,
                }

        player_events: Dict[str, List[Dict[str, Any]]] = {pid: [] for pid in players_seen}

        for ev in events:
            pid = ev.get("player_id", "")
            if pid and pid in players_seen:
                player_events[pid].append(ev)

        findings = []
        for pid, info in players_seen.items():
            pe = player_events.get(pid, [])

            metrics = calculate_player_metrics_from_events(
                pe, pid, info["player_name"], info["team"], info["position"]
            )

            info["metrics"] = metrics
            info["total_player_events"] = len(pe)

            zone_dist = {}
            channel_dist = {}
            for ev in pe:
                x = ev.get("end_x", 0)
                y = ev.get("end_y", 0)
                from ..schemas import _get_zone, _get_channel
                z = _get_zone(x)
                c = _get_channel(y)
                zone_dist[z] = zone_dist.get(z, 0) + 1
                channel_dist[c] = channel_dist.get(c, 0) + 1

            events_by_type = {}
            for etype in ["pass", "carry", "shot", "cross", "pressure", "tackle", "recovery"]:
                type_events = [e for e in pe if e.get("event_type") == etype]
                if type_events:
                    events_by_type[etype] = {
                        "count": len(type_events),
                        "completed": sum(1 for e in type_events if e.get("completed", False)),
                    }

            info["events_by_type"] = events_by_type

            findings.append({
                "category": "player",
                "headline": f"{info['player_name']} ({info['team']}): {len(pe)} actions",
                "observation": (
                    f"Passes: {metrics.total_passes} ({metrics.completed_passes} completed, "
                    f"{metrics.pass_completion_rate*100:.0f}%)"
                ),
                "interpretation": (
                    f"Key contributor in {info['position']} position with "
                    f"{metrics.total_passes} pass involvements."
                ),
                "significance": (
                    "Player involvement data - supports deeper tactical analysis."
                ),
                "supporting_event_ids": [e.get("event_id", "") for e in pe[:5]],
                "sample_size": len(pe),
                "source_type": "calculated",
                "confidence": None,
                "visualization": "player_timeline",
                "player_id": pid,
                "team": info["team"],
                "position": info["position"],
                "data": {
                    "metrics": metrics,
                    "zone_distribution": zone_dist,
                    "channel_distribution": channel_dist,
                    "events_by_type": events_by_type,
                },
            })

        return findings
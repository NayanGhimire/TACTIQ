"""TACTIQ AI - Tactical Analyst Agent.

Agent 2: Calculates build-up, attacking, defensive, and transition statistics.
Calls the analytics modules (buildup, attack, defence, transitions).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..analytics.buildup import calculate_build_up_segments
from ..analytics.attack import calculate_attack_patterns
from ..analytics.defence import calculate_defensive_behaviour
from ..analytics.transitions import calculate_transitions

logger = logging.getLogger(__name__)


class TacticalAnalystAgent:
    """Agent 2: computes build-up, attack, defence, transition statistics."""

    def analyze(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run tactical analysis on all events for both teams.

        Returns a list of structured findings with evidence citations.
        """
        findings = []

        teams = sorted(set(e.get("team_id", "home") for e in events if e.get("team_id")))

        for team_id in teams:
            team_events = [e for e in events if e.get("team_id") == team_id]

            # Build-up
            buildup = calculate_build_up_segments(team_events, team_id)
            findings.append({
                "category": "build_up",
                "headline": f"{team_id} build-up: {buildup['completion_rate']*100:.0f}% completion",
                "observation": (
                    f"{buildup['total_passes']} passes attempted from defensive third; "
                    f"{buildup['completed_passes']} completed ({buildup['completion_rate']*100:.0f}%). "
                    f"{buildup['pressure_build_up']} under pressure."
                ),
                "interpretation": (
                    "Build-up play concentrated in defensive third with "
                    f"{buildup['third_crossings']} third crossings."
                ),
                "significance": (
                    "Sustained build-up suggests controlled possession from defence."
                ),
                "supporting_event_ids": [getattr(s, "event_id", "") for s in buildup.get("segments", [])[:5]],
                "sample_size": buildup["total_passes"],
                "source_type": "calculated",
                "confidence": None,
                "visualization": "passing_network",
                "team_id": team_id,
                "data": buildup,
            })

            # Attacking
            attack = calculate_attack_patterns(team_events, team_id)
            findings.append({
                "category": "attack",
                "headline": f"{team_id} attack: {attack['total_shots']} shots from {attack['final_third_entries']} entries",
                "observation": (
                    f"{attack['final_third_entries']} final-third entries, {attack['box_entries']} box entries, "
                    f"{attack['total_shots']} shots ({attack['shots_on_target']} on target)."
                ),
                "interpretation": (
                    f"Chance creation rate: {attack['shot_conversion_rate']*100:.0f}% "
                    f"({attack['goals']} goals from {attack['total_shots']} shots)."
                ),
                "significance": (
                    "High final-third entries indicate sustained attacking pressure."
                ),
                "supporting_event_ids": [getattr(s, "event_id", "") for s in attack.get("shots", [])[:5]],
                "sample_size": attack["final_third_entries"],
                "source_type": "calculated",
                "confidence": None,
                "visualization": "shot_map",
                "team_id": team_id,
                "data": attack,
            })

            # Defence
            defence = calculate_defensive_behaviour(team_events, team_id)
            findings.append({
                "category": "defense",
                "headline": f"{team_id} defence: {defence['total_defensive_actions']} actions",
                "observation": (
                    f"{defence['pressures']} pressures, {defence['tackles']} tackles ({defence['tackle_success_rate']*100:.0f}% won), "
                    f"{defence['recoveries']} recoveries."
                ),
                "interpretation": (
                    f"Pressure success rate: {defence['pressure_success_rate']*100:.0f}% "
                    f"(turnovers forced per pressure)."
                ),
                "significance": (
                    "High pressure success indicates effective pressing structure."
                ),
                "supporting_event_ids": [a["event_id"] for a in defence.get("defensive_actions", {}).get("pressures", [])[:5]],
                "sample_size": defence["total_defensive_actions"],
                "source_type": "calculated",
                "confidence": None,
                "visualization": "pressure_map",
                "team_id": team_id,
                "data": defence,
            })

            # Transitions
            transitions = calculate_transitions(team_events, team_id)
            findings.append({
                "category": "transition",
                "headline": f"{team_id} transitions: {transitions['total_possessions']} possessions",
                "observation": (
                    f"{transitions['total_possessions']} possession changes; avg duration "
                    f"{transitions['avg_possession_duration_s']}s; {transitions['counter_presses']} counter-presses."
                ),
                "interpretation": (
                    "Transition rhythm suggests " +
                    ("sustained control" if transitions['avg_possession_duration_s'] > 30 else "end-to-end game")
                ),
                "significance": (
                    "Transition patterns reveal how quickly a team attacks after regaining possession."
                ),
                "supporting_event_ids": [c["event_id"] for c in transitions.get("possession_changes", [])[:5]],
                "sample_size": transitions["total_possessions"],
                "source_type": "calculated",
                "confidence": None,
                "visualization": "possession_timeline",
                "team_id": team_id,
                "data": transitions,
            })

        return findings
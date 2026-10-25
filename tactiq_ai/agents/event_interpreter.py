"""Event Interpreter Agent.

Validates events, tracks match state, and identifies meaningful sequences.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..data.event_loader import MatchProcessor, validate_event_sequence
from ..schemas import _get_zone, _get_channel

logger = logging.getLogger(__name__)


class EventInterpreterAgent:
    """Agent 1: validates events, tracks match state, identifies meaningful sequences."""

    def __init__(self):
        self.processor: Optional[MatchProcessor] = None

    def interpret(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate events and track match state.

        Returns a dict of sequence-level findings.
        """
        if not events:
            return {"error": "no events"}

        validated = validate_event_sequence(events)

        # Find meaningful sequences: runs of passes in defensive third
        sequences = []
        current_seq = []
        for ev in validated:
            x1 = ev.get("end_x", 0)
            zone = _get_zone(x1)
            if ev.get("event_type") in ("pass", "carry") and zone == "defensive_third":
                current_seq.append(ev)
            else:
                if len(current_seq) >= 2:
                    sequences.append(current_seq)
                current_seq = []

        build_up_sequences = sequences

        return {
            "n_validated_events": len(validated),
            "n_sequences": len(build_up_sequences),
            "build_up_sequences": build_up_sequences,
            "teams_seen": sorted(set(e.get("team_id") for e in validated)),
            "event_types_seen": sorted(set(e.get("event_type") for e in validated)),
            "source_types": sorted(set(e.get("source", "synthetic") for e in validated)),
        }

    def setup_processor(self, events: List[Dict[str, Any]], match_id: str = "live_match") -> MatchProcessor:
        """Set up incremental processor."""
        self.processor = MatchProcessor(events, match_id)
        return self.processor

"""TACTIQ AI - Overlay Formatter Agent.

Agent 8: Creates timestamped, machine-readable insight objects for synchronized
on-screen rendering. All overlays include match ID, timestamps, category,
headline, explanation, priority, player IDs, event IDs, visualization type,
source type, and confidence info where available.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class OverlayFormatterAgent:
    """Agent that formats insights into machine-readable overlay JSON objects."""

    PRIORITY_MAP = {
        "model": 3,
        "attack": 4,
        "build_up": 2,
        "defense": 2,
        "transition": 1,
        "player": 3,
        "moment": 5,
    }

    def format_overlays(
        self,
        insights: List[Dict[str, Any]],
        events: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Format insights as overlays ready for broadcast rendering.

        Returns list of overlay dicts with machine-readable fields.
        """
        overlays = []

        for insight in insights:
            overlay = {
                "overlay_id": insight.get("insight_id", ""),
                "match_id": insight.get("match_id", ""),
                "start_timestamp": insight.get("timestamp_seconds", 0),
                "end_timestamp": insight.get("timestamp_seconds", 0) + 10,
                "duration_seconds": 10,
                "category": insight.get("category", ""),
                "headline": insight.get("headline", ""),
                "explanation": insight.get("observation", ""),
                "display_priority": self.PRIORITY_MAP.get(
                    insight.get("category", ""), 1
                ),
                "related_player_ids": insight.get("player_ids", []) or [],
                "supporting_event_ids": insight.get("supporting_event_ids", []) or [],
                "visualization_type": insight.get("visualization", "pitch_highlight"),
                "source_type": insight.get("source_type", "synthetic"),
                "confidence": insight.get("confidence"),
                "uncertain": insight.get("uncertain", False),
                "synthetic": insight.get("synthetic", True),
                "match_timestamp_seconds": insight.get("timestamp_seconds", 0),
                "metadata": {
                    "sample_size": insight.get("sample_size"),
                    "flags": insight.get("flags", []),
                    "jargon_level": insight.get("jargon_level", ""),
                },
            }
            overlays.append(overlay)

        # Sort by start_timestamp then priority
        overlays.sort(key=lambda o: (o["start_timestamp"], -o["display_priority"]))

        logger.info(f"Formatted {len(overlays)} overlays")
        return overlays

    def to_json(self, overlays: List[Dict[str, Any]]) -> str:
        """Serialize overlays as pretty-printed JSON."""
        import json
        return json.dumps(overlays, indent=2, ensure_ascii=False)

    def export_overlays(
        self,
        overlays: List[Dict[str, Any]],
        path: str,
    ) -> str:
        """Export overlays to a JSON file. Returns the path written."""
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(overlays, f, indent=2, ensure_ascii=False)
        logger.info(f"Exported {len(overlays)} overlays to {path}")
        return path

"""TACTIQ AI - Event Validation utilities.

Provides validated JSON Schema for event data, validation functions for
incoming events, and data-quality reporting.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from ..schemas import TacticalEvent
from .event_loader import validate_event, validate_event_sequence, EventValidationError

# ----------------------------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------------------------


def validate_events_bulk(
    events: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Validate a list of events and return valid ones + error messages.

    Args:
        events: List of event dictionaries

    Returns:
        (valid_events, error_messages) - valid events and any errors encountered.
    """
    valid = []
    errors = []

    for i, ev in enumerate(events):
        try:
            validated = validate_event(ev, i)
            valid.append(validated)
        except EventValidationError as e:
            errors.append(str(e))

    return valid, errors


def validate_model_input_sequence(
    event_sequence: List[Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    """Validate that a sequence is suitable for model inference.

    Args:
        event_sequence: List of events to validate

    Returns:
        (is_valid, error_message) - True if valid, False with message otherwise.
    """
    if not isinstance(event_sequence, (list, tuple)):
        return False, "event_sequence must be a list of event dicts"

    n = len(event_sequence)

    if n < 3:
        return False, f"need at least 3 events, got {n}"

    if n > 14:
        return False, (
            f"at most 14 events per sequence (model window); got {n}. "
            "Slice a window of the most relevant events first."
        )

    # Validate each event
    for i, ev in enumerate(event_sequence):
        if not isinstance(ev, dict):
            return False, f"event {i}: must be a dict"

        required = ["type", "x_start", "y_start", "x_end", "y_end"]
        for key in required:
            if key not in ev:
                return False, f"event {i}: missing required key '{key}'"

        if ev["type"] not in ["pass", "carry", "cross", "shot", "pressure",
                              "tackle", "recovery", "ball_loss", "opp_pass"]:
            return False, (
                f"event {i}: unknown type {ev['type']!r}; "
                f"allowed: pass, carry, cross, shot, pressure, tackle, "
                f"recovery, ball_loss, opp_pass"
            )

        for key in ["x_start", "y_start", "x_end", "y_end"]:
            try:
                val = float(ev[key])
            except (TypeError, ValueError):
                return False, f"event {i}: '{key}' must be a number"
            if not (0 <= val <= 100):
                return False, f"event {i}: '{key}'={val} outside allowed range [0, 100]"

    return True, None


# ----------------------------------------------------------------------------------------
# Data quality reporting
# ----------------------------------------------------------------------------------------


class DataQualityReport:
    """Reports on data quality metrics for a set of events."""

    def __init__(self, events: List[Dict[str, Any]], source_type: str):
        self.events = events
        self.source_type = source_type
        self._compute_metrics()

    def _compute_metrics(self) -> None:
        self.event_count = len(self.events)
        if self.event_count == 0:
            self.match_duration = 0.0
            self.player_count = 0
            self.missing_fields = {}
            self.available_event_types = []
            return

        timestamps = [e["timestamp_seconds"] for e in self.events]
        self.match_duration = max(timestamps) - min(timestamps) if timestamps else 0.0

        player_ids = set(e.get("player_id", "") for e in self.events if e.get("player_id"))
        self.player_count = len(player_ids)

        # Count missing fields
        field_counts: Dict[str, int] = {}
        for e in self.events:
            for key in e:
                field_counts[key] = field_counts.get(key, 0) + 1

        all_required = [
            "event_type", "team_id", "player_id", "start_x", "start_y",
            "end_x", "end_y", "timestamp_seconds", "match_id",
        ]
        self.missing_fields = {}
        for field in all_required:
            missing = self.event_count - field_counts.get(field, 0)
            if missing > 0:
                self.missing_fields[field] = missing

        self.available_event_types = sorted(set(
            e.get("event_type", "unknown") for e in self.events
        ))

        # Periods covered
        self.periods = sorted(set(e.get("period", 1) for e in self.events))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_count": self.event_count,
            "source_type": self.source_type,
            "match_duration_seconds": round(self.match_duration, 1),
            "match_duration_minutes": round(self.match_duration / 60.0, 1),
            "player_count": self.player_count,
            "periods": self.periods,
            "missing_fields": self.missing_fields,
            "available_event_types": self.available_event_types,
        }

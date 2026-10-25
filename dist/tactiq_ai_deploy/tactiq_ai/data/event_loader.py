"""TACTIQ AI - Event Loader and Validator.

Handles loading events from multiple sources (synthetic, CSV, JSON, StatsBomb),
validating them, and managing incremental processing with replay support.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..config import SOURCE_REAL, SOURCE_SYNTHETIC, SOURCE_UPLOADED
from ..schemas import TacticalEvent

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------
# Required fields validation
# ----------------------------------------------------------------------------------------

REQUIRED_EVENT_FIELDS = [
    "event_type",
    "team_id",
    "player_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "timestamp_seconds",
    "match_id",
]

OPTIONAL_EVENT_FIELDS = [
    "player_name",
    "period",
    "event_type",
    "outcome",
    "completed",
    "under_pressure",
    "turnover",
    "possession_id",
    "possession_before",
    "possession_after",
    "t_start_s",
    "duration_s",
    "source",
    "channel",
    "zone",
    "auxiliary_data",
]


class EventValidationError(Exception):
    """Raised when an event fails validation."""


class TimelineError(Exception):
    """Raised when event timestamps violate chronological order."""


# ----------------------------------------------------------------------------------------
# Event validation
# ----------------------------------------------------------------------------------------


def validate_event(event: Dict[str, Any], index: int = None) -> Dict[str, Any]:
    """Validate a single event dict and return a normalized version.

    Args:
        event: Event dictionary
        index: Index of the event in the sequence (for error messages)

    Returns:
        Normalized event dictionary with all required fields.

    Raises:
        EventValidationError: If required fields are missing or values are invalid.
    """
    prefix = f"event {index}: " if index is not None else ""

    if not isinstance(event, dict):
        raise EventValidationError(f"{prefix}must be a dict, got {type(event).__name__}")

    for field in REQUIRED_EVENT_FIELDS:
        if field not in event:
            raise EventValidationError(f"{prefix}missing required field '{field}'")

    if event["event_type"] not in [
        "pass", "carry", "cross", "shot", "pressure", "tackle", "recovery",
        "ball_loss", "opp_pass", "clearance", "interception", "dribble",
        "progressive_carry", "possession_change", "goal",
        "substitution", "card_yellow", "card_red", "offside",
    ]:
        raise EventValidationError(f"{prefix}unknown event_type '{event['event_type']}'")

    # Validate coordinates are numbers in valid ranges
    for coord in ["start_x", "start_y", "end_x", "end_y"]:
        try:
            val = float(event[coord])
        except (TypeError, ValueError):
            raise EventValidationError(f"{prefix}'{coord}' must be a number, got {event[coord]!r}")
        if not (-5 < val < 130):  # Allow some tolerance for off-pitch events
            raise EventValidationError(f"{prefix}'{coord}'={val} is outside valid pitch range")

    # Validate timestamp
    try:
        ts = float(event["timestamp_seconds"])
    except (TypeError, ValueError):
        raise EventValidationError(f"{prefix}'timestamp_seconds' must be a number")
    if ts < 0:
        raise EventValidationError(f"{prefix}'timestamp_seconds' must be non-negative")

    # Validate team_id
    if event["team_id"] not in ("home", "away"):
        raise EventValidationError(
            f"{prefix}'team_id' must be 'home' or 'away', got '{event['team_id']}'"
        )

    # Set defaults for optional fields
    normalized = dict(event)
    normalized.setdefault("player_name", event.get("player_id", "Unknown"))
    normalized.setdefault("period", 1)
    normalized.setdefault("outcome", None)
    normalized.setdefault("completed", None)
    normalized.setdefault("under_pressure", False)
    normalized.setdefault("turnover", False)
    normalized.setdefault("possession_id", None)
    normalized.setdefault("source", SOURCE_SYNTHETIC)

    normalized["under_pressure"] = bool(normalized["under_pressure"])
    normalized["turnover"] = bool(normalized["turnover"])
    normalized["timestamp_seconds"] = float(normalized["timestamp_seconds"])

    # Determine source if not provided
    if "source" not in event:
        normalized["source"] = SOURCE_SYNTHETIC

    return normalized


def validate_event_sequence(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate a sequence of events and ensure chronological ordering.

    Args:
        events: List of event dictionaries

    Returns:
        List of validated, normalized event dictionaries

    Raises:
        TimelineError: If events are not in chronological order
        EventValidationError: If any event is invalid
    """
    validated = []
    prev_ts = -1e9

    for i, event in enumerate(events):
        ev = validate_event(event, i)
        if ev["timestamp_seconds"] < prev_ts:
            raise TimelineError(
                f"event {i}: timestamp {ev['timestamp_seconds']} is out of order "
                f"(previous: {prev_ts})"
            )
        validated.append(ev)
        prev_ts = ev["timestamp_seconds"]

    return validated


# ----------------------------------------------------------------------------------------
# Loader functions
# ----------------------------------------------------------------------------------------


def load_events_from_dicts(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Load events from a list of dictionaries.

    Each event is validated and normalized. Source is set to 'uploaded'.
    """
    normalized_events = []
    for i, ev in enumerate(events):
        if "source" not in ev:
            ev["source"] = SOURCE_UPLOADED
        normalized_events.append(ev)

    return validate_event_sequence(normalized_events)


def load_events_from_json(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load events from a JSON file.

    Supports both:
    - A list of event dicts: [{"event_type": ... }, ...]
    - A dict with an "events" key: {"events": [...], "metadata": {...}}
    """
    path = Path(path)
    with open(path, "r") as f:
        data = json.load(f)

    if isinstance(data, dict) and "events" in data:
        events = data["events"]
        source = data.get("metadata", {}).get("source", SOURCE_UPLOADED)
    elif isinstance(data, list):
        events = data
        source = SOURCE_UPLOADED
    else:
        raise EventValidationError("JSON file must be a list of events or a dict with 'events' key")

    # Set source for each event
    for ev in events:
        if "source" not in ev:
            ev["source"] = source

    return validate_event_sequence(events)


def load_events_from_csv(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load events from a CSV file.

    Expected columns: event_type, team_id, player_id, start_x, start_y, end_x, end_y,
    timestamp_seconds, and optionally: player_name, period, outcome, completed,
    under_pressure, turnover, possession_id, duration_s, source
    """
    path = Path(path)
    events = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)

        required_csv = ["event_type", "team_id", "player_id", "start_x", "start_y",
                        "end_x", "end_y", "timestamp_seconds"]
        for col in required_csv:
            if col not in reader.fieldnames:
                raise EventValidationError(f"CSV missing required column: '{col}'")

        for row in reader:
            ev = {
                "event_type": row["event_type"],
                "team_id": row["team_id"],
                "player_id": row["player_id"],
                "player_name": row.get("player_name", row["player_id"]),
                "start_x": float(row["start_x"]),
                "start_y": float(row["start_y"]),
                "end_x": float(row["end_x"]),
                "end_y": float(row["end_y"]),
                "timestamp_seconds": float(row["timestamp_seconds"]),
                "period": int(row["period"]) if row.get("period") else 1,
                "outcome": row.get("outcome"),
                "completed": row.get("completed", "").lower() in ("true", "1", "yes"),
                "under_pressure": row.get("under_pressure", "").lower() in ("true", "1", "yes"),
                "turnover": row.get("turnover", "").lower() in ("true", "1", "yes"),
                "possession_id": row.get("possession_id"),
                "duration_s": float(row["duration_s"]) if row.get("duration_s") else None,
                "source": row.get("source", SOURCE_UPLOADED),
            }
            events.append(ev)

    return validate_event_sequence(events)


# ----------------------------------------------------------------------------------------
# Incremental processing engine
# ----------------------------------------------------------------------------------------


class MatchProcessor:
    """Processes events incrementally, simulating a live match.

    Supports:
    - Start / pause / resume / reset
    - Replay based on timestamps
    - Sequential event emission
    """

    def __init__(self, events: List[Dict[str, Any]], match_id: str = "live_match"):
        """Initialize the processor with a list of events.

        Events must be pre-validated and chronologically ordered.
        """
        self.match_id = match_id
        self._events = list(events)
        self._index = 0
        self._paused = False
        self._ended = False
        self._speed = 1.0  # simulation speed multiplier
        self._processed_events: List[Dict[str, Any]] = []

    @property
    def events(self) -> List[Dict[str, Any]]:
        return list(self._events)

    @property
    def processed_events(self) -> List[Dict[str, Any]]:
        return list(self._processed_events)

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_ended(self) -> bool:
        return self._ended

    @property
    def current_index(self) -> int:
        return self._index

    @property
    def n_processed(self) -> int:
        return len(self._processed_events)

    @property
    def n_total(self) -> int:
        return len(self._events)

    @property
    def speed(self) -> float:
        return self._speed

    def start(self, events: Optional[List[Dict[str, Any]]] = None) -> None:
        """Start or restart processing with given events (or existing ones)."""
        if events is not None:
            self._events = list(events)
        self._index = 0
        self._processed_events = []
        self._paused = False
        self._ended = False
        self._speed = 1.0

    def pause(self) -> None:
        """Pause incremental processing."""
        self._paused = True

    def resume(self) -> None:
        """Resume incremental processing."""
        self._paused = False

    def reset(self) -> None:
        """Reset the processor to initial state."""
        self._index = 0
        self._processed_events = []
        self._paused = False
        self._ended = False

    def set_speed(self, speed: float) -> None:
        """Set simulation speed multiplier."""
        self._speed = max(0.1, speed)

    def next_event(self) -> Optional[Dict[str, Any]]:
        """Process and return the next event. Returns None if ended or paused."""
        if self._paused or self._ended or self._index >= len(self._events):
            if self._index >= len(self._events):
                self._ended = True
            return None

        event = self._events[self._index]
        self._processed_events.append(dict(event))
        self._index += 1

        if self._index >= len(self._events):
            self._ended = True

        return event

    def next_events(self, n: int = 1) -> List[Dict[str, Any]]:
        """Process and return up to n events."""
        result = []
        for _ in range(n):
            ev = self.next_event()
            if ev is None:
                break
            result.append(ev)
        return result

    def replay_to_timestamp(self, timestamp: float) -> List[Dict[str, Any]]:
        """Replay events up to a given timestamp (inclusive).

        This resets the processor and replays events up to the specified timestamp.
        """
        self.reset()
        replayed = []
        for ev in self._events:
            if ev["timestamp_seconds"] <= timestamp:
                self._processed_events.append(dict(ev))
                self._index += 1
                replayed.append(dict(ev))
            else:
                break
        return replayed

    def seek(self, timestamp: float) -> None:
        """Seek to a specific timestamp without replaying."""
        new_index = 0
        for i, ev in enumerate(self._events):
            if ev["timestamp_seconds"] <= timestamp:
                new_index = i + 1
            else:
                break

        self._index = new_index
        self._processed_events = [dict(ev) for ev in self._events[:new_index]]
        self._ended = new_index >= len(self._events)

    def get_state(self) -> Dict[str, Any]:
        """Return current processor state."""
        return {
            "match_id": self.match_id,
            "current_index": self._index,
            "n_total": self.n_total,
            "n_processed": self.n_processed,
            "is_paused": self._paused,
            "is_ended": self._ended,
            "speed": self._speed,
        }

    def get_events_in_window(
        self,
        start_ts: float,
        end_ts: float,
        team_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get events within a timestamp window, optionally filtered by team."""
        result = []
        for ev in self._events:
            if start_ts <= ev["timestamp_seconds"] <= end_ts:
                if team_id is None or ev.get("team_id") == team_id:
                    result.append(ev)
        return result

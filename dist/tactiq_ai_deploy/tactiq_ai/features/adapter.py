"""TACTIQ AI - Feature Adapter.

Converts rich football events (with models) into the input format expected by
tactiq_inference.py (9 event types, 0-100 coords, possession tracking).

Key requirements per user instruction:
- Model only sees 14 events; extract the most relevant window.
- Rich events (receptions, clearances, cards) need mapping into the 9-type schema.
- Events like opp_pass must carry the focal team's frame so attacking-direction
  handling is correct — flip away team's coordinates so they always attack toward x=100.
- Coordinates rescale from pitch metres to 0-100 frame; default pitch 105x68.
- Keep tactiq_inference.py in project; use load_predictor() — don't re-implement preprocessing.
- Don't reorder classes — order is in tactiq_label_mapping.json.
- All outputs labelled as synthetic.
- Use the uncertain label (other_or_uncertain) when low confidence (<0.60).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..schemas import (
    ETYPES, ETYPE_TO_ID, map_event_type_to_model,
    rescale_coordinates_to_100x100, flip_coordinates_for_away_team,
    PITCH_L, PITCH_W, T_MAX, MIN_EVENTS,
)
from ..data.synthetic_generator import MatchState, PlayerState

logger = logging.getLogger(__name__)

# Mapping of rich event type -> model internal type ID.
# This is shared across both the generator and inference agent.
RICH_TO_MODEL_TYPE = {
    "pass": "pass",
    "carry": "carry",
    "cross": "cross",
    "shot": "shot",
    "pressure": "pressure",
    "tackle": "tackle",
    "recovery": "recovery",
    "ball_loss": "ball_loss",
    "opp_pass": "opp_pass",
    "clearance": "recovery",
    "interception": "tackle",
    "dribble": "carry",
    "progressive_carry": "carry",
    "possession_change": "ball_loss",
    "goal": "shot",
    "substitution": "opp_pass",
    "card_yellow": "opp_pass",
    "card_red": "opp_pass",
    "offside": "opp_pass",
}


def get_model_type(event_type: str) -> int:
    """Return model type ID for a rich event type."""
    mapped = RICH_TO_MODEL_TYPE.get(event_type)
    if mapped is None:
        raise ValueError(f"Unknown rich event type '{event_type}'")
    return ETYPE_TO_ID[mapped]


def events_to_model_dict(events: List[Dict[str, Any]], team_id: str) -> List[dict]:
    """Convert rich events to the model's expected dict format.

    Args:
        events: Rich event dicts with all fields
        team_id: 'home' or 'away'

    Returns:
        List of dicts in model format:
        {"type", "x_start", "y_start", "x_end", "y_end", "duration_s",
         "completed", "under_pressure", "turnover", "possession_before",
         "possession_after", "t_start_s"}

    Coordinate handling:
    - Rescale coordinates from metres to 0-100
    - Away team events are flipped so they always attack toward x=100
    """
    model_events = []
    for ev in events:
        # Map event type
        model_type = get_model_type(ev.get("event_type", "pass"))

        # Normalize coordinates
        x0, y0 = float(ev.get("start_x", 0)), float(ev.get("start_y", 0))
        x1, y1 = float(ev.get("end_x", x0 if model_type == "shot" else 0)), float(ev.get("end_y", y0))

        if team_id == "away":
            x0, y0 = flip_coordinates_for_away_team(x0, y0)
            x1, y1 = flip_coordinates_for_away_team(x1, y1)

        # Rescale to 0-100 frame
        scored_x0, scored_y0 = rescale_coordinates_to_100x100(x0, y0)
        scored_x1, scored_y1 = rescale_coordinates_to_100x100(x1, y1)

        completed = ev.get("completed", True)
        under_pressure = bool(ev.get("under_pressure", False))
        turnover = bool(ev.get("turnover", False))

        model_dict = {
            "type": model_type,
            "x_start": round(scored_x0, 4),
            "y_start": round(scored_y0, 4),
            "x_end": round(scored_x1, 4),
            "y_end": round(scored_y1, 4),
            "duration_s": float(ev.get("duration_s", 1.5)),
            "completed": 1 if completed else 0,
            "under_pressure": 1 if under_pressure else 0,
            "turnover": 1 if turnover else 0,
            "possession_before": ev.get("auxiliary_data", {}).get("possession_before", 0),
            "possession_after": ev.get("auxiliary_data", {}).get("possession_after", 1),
            "t_start_s": float(ev.get("t_start_s", ev.get("timestamp_seconds", 0))),
        }
        model_events.append(model_dict)

    return model_events


def extract_model_window(
    model_events: List[dict],
    n_events: Optional[int] = None,
    team_id: str = "home",
) -> List[dict]:
    """Extract the most relevant window of up to 14 model events.

    Strategy: prefer windows starting in defensive third (x <= 33) since these
    are most distinctive of build-up patterns. If fewer than 3 events, return
    all valid ones. Also limit total windows processed to avoid redundant work.

    Returns the window to pass to the classifier.
    """
    if not model_events:
        return []

    n_events = n_events or len(model_events)

    # Strategy 1: Windows starting in defensive third
    windows = []
    seen = set()
    for start in range(0, n_events - 2, 3):  # stride 3 for sliding windows
        window = model_events[start:start + T_MAX]
        if len(window) >= MIN_EVENTS and len(window) <= T_MAX:
            # Check if first event starts in defensive third
            if window[0].get("x_start", 0) < 40:  # defensive third
                key = tuple((w["type"], round(w["x_start"], 1)) for w in window)
                if key not in seen:
                    seen.add(key)
                    windows.append(window)

    # Strategy 2: First n events if few
    if len(windows) == 0 or len(windows) < 2:
        for start in range(min(n_events, max(1, (n_events // 4))), n_events):
            window = model_events[start:start + T_MAX]
            if len(window) >= MIN_EVENTS and len(window) <= T_MAX:
                windows.append(window)

    # Deduplicate by signature
    seen_keys = set()
    unique_windows = []
    for w in windows[:4]:  # Limit to 4 windows
        key = tuple((e["type"], round(e["x_start"], 1), round(e["y_start"], 1)) for e in w)
        if key not in seen_keys:
            seen_keys.add(key)
            unique_windows.append(w)

    logger.info(f"Extracted {len(unique_windows)} model windows from {n_events} events")
    return unique_windows[0] if unique_windows else model_events[:T_MAX]


def compute_pass_distance(x0: float, y0: float, x1: float, y1: float) -> float:
    """Compute pass distance in metres from 0-100 coordinates."""
    scaled_x0 = x0 * PITCH_L / 100.0
    scaled_y0 = y0 * PITCH_W / 100.0
    scaled_x1 = x1 * PITCH_L / 100.0
    scaled_y1 = y1 * PITCH_W / 100.0
    return ((scaled_x1 - scaled_x0)**2 + (scaled_y1 - scaled_y0)**2)**0.5


def classify_sequence_window(
    home_events: List[Dict[str, Any]],
    away_events: List[Dict[str, Any]],
    predictor=None,
) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Classify both teams' sequences and return predictions.

    Returns (home_pred, away_pred) or (None, None) if model unavailable.
    """
    from ..features.inference import InferenceAgent

    agent = predictor or InferenceAgent()
    try:
        home_model = events_to_model_dict(home_events, "home")
        away_model = events_to_model_dict(away_events, "away")

        home_window = extract_model_window(home_model)
        away_window = extract_model_window(away_model)

        home_pred = agent.classify_sequence(home_window) if home_window else None
        away_pred = agent.classify_sequence(away_window) if away_window else None

        return home_pred, away_pred
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return None, None

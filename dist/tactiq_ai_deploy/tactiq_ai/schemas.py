"""TACTIQ AI - Event Schemas and Constants.

Defines the event schema used for match events, the model's expected input format,
and the adapter mappings between rich event types and the 14-event model window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ----------------------------------------------------------------------------------------
# Model configuration - from tactiq_inference.py
# ----------------------------------------------------------------------------------------

T_MAX = 14  # max events per sequence (post-padded with zeros)
MIN_EVENTS = 3

ETYPES = [
    "pass",
    "carry",
    "cross",
    "shot",
    "pressure",
    "tackle",
    "recovery",
    "ball_loss",
    "opp_pass",
]

ETYPE_TO_ID = {n: i for i, n in enumerate(ETYPES)}

RAW_COLS = [
    "etype",
    "x0",
    "y0",
    "x1",
    "y1",
    "t_start",
    "dur",
    "completed",
    "under_pressure",
    "turnover",
    "poss_after",
    "prev_poss",
]

RC = {n: i for i, n in enumerate(RAW_COLS)}
N_RAW = len(RAW_COLS)

CHANNELS = ["left_wide", "left_halfspace", "central", "right_halfspace", "right_wide"]
CHANNEL_EDGES = [20.0, 38.0, 62.0, 80.0]  # on the 0-100 y axis

ZONES = ["defensive_third", "middle_third", "attacking_third"]
ZONE_EDGES = [33.3333, 66.6667]  # on the 0-100 x axis

# ----------------------------------------------------------------------------------------
# Adapter: Rich event types -> model's 9 event types
# ----------------------------------------------------------------------------------------


def map_event_type_to_model(event_type: str) -> int:
    """Map a rich event type to the model's internal type ID.

    The model was trained on 9 event types: pass, carry, cross, shot, pressure,
    tackle, recovery, ball_loss, opp_pass. Richer events (clearances, interceptions,
    cards, substitutions) are mapped here.
    """
    mapping = {
        "pass": "pass",
        "carry": "carry",
        "cross": "cross",
        "shot": "shot",
        "pressure": "pressure",
        "tackle": "tackle",
        "recovery": "recovery",
        "ball_loss": "ball_loss",
        "opp_pass": "opp_pass",
        # Richer events that need adapter mapping
        "clearance": "recovery",  # treated as recovery
        "interception": "tackle",  # treated as tackle
        "dribble": "carry",
        "progressive_carry": "carry",
        "possession_change": "ball_loss",
        "goal": "shot",
        # Events that always map to opp_pass in the attacking team's frame
        "substitution": "opp_pass",
        "card_yellow": "opp_pass",
        "card_red": "opp_pass",
        "offside": "opp_pass",
    }
    mapped = mapping.get(event_type)
    if mapped is None:
        raise ValueError(f"Unknown event type '{event_type}'; cannot map to model schema")
    return ETYPE_TO_ID[mapped]


# ----------------------------------------------------------------------------------------
# Pitch coordinate convention
# ----------------------------------------------------------------------------------------

PITCH_L = 105.0  # metres
PITCH_W = 68.0  # metres

# Coordinate frame: attacking-team frame, x 0..100, y 0..100
# If events use StatsBomb 120x80, they get rescaled to 100x100
# For the away team, coordinates are flipped so they always attack toward x=100

# Channel classification on y-axis (0-100)
CHANNEL_EDGES_Y = [20.0, 38.0, 62.0, 80.0]
# Zone classification on x-axis (0-100)
ZONE_EDGES_X = [33.3333, 66.6667]


def _get_zone(x_100: float) -> str:
    """Classify x coordinate into pitch zone."""
    if x_100 < 33.3333:
        return "defensive_third"
    elif x_100 < 66.6667:
        return "middle_third"
    else:
        return "attacking_third"


def _get_channel(y_100: float) -> str:
    """Classify y coordinate into pitch channel."""
    if y_100 < 20.0:
        return "left_wide"
    elif y_100 < 38.0:
        return "left_halfspace"
    elif y_100 < 62.0:
        return "central"
    elif y_100 < 80.0:
        return "right_halfspace"
    else:
        return "right_wide"


def rescale_coordinates_to_100x100(
    x: float,
    y: float,
    pitch_length: float = PITCH_L,
    pitch_width: float = PITCH_W,
) -> tuple[float, float]:
    """Rescale pitch coordinates to the 0-100 x 0-100 model frame.

    Args:
        x: x coordinate (metres, 0 = own goal line, pitch_length = opponent goal line)
        y: y coordinate (metres, 0 = left touchline, pitch_width = right touchline)
        pitch_length: total pitch length in metres (default 105)
        pitch_width: total pitch width in metres (default 68)

    Returns:
        (x_scaled, y_scaled) in 0-100 coordinates
    """
    x_scaled = x * 100.0 / pitch_length
    y_scaled = y * 100.0 / pitch_width
    return x_scaled, y_scaled


def flip_coordinates_for_away_team(
    x: float,
    y: float,
) -> tuple[float, float]:
    """Flip coordinates so the away team always attacks toward x=100.

    In the attacking-team frame, x=0 is the owning team's goal line and x=100 is
    the opponent's goal line. When the away team has possession, their events need
    to be mirrored so they also use the x=0..100 frame consistently.
    """
    # Mirror x: new_x = 100 - old_x
    # y stays the same (just rescaled)
    x_flipped = 100.0 - x
    return x_flipped, y


# ----------------------------------------------------------------------------------------
# Event dataclasses
# ----------------------------------------------------------------------------------------


@dataclass
class TacticalEvent:
    """A single football event with tactical metadata.

    This is the rich event schema used throughout the application. Events get
    adapted/adapted-to the model's 9-type schema before model inference.
    """

    # Core identification
    match_id: str
    timestamp_seconds: float
    period: int
    team_id: str  # "home" or "away"
    player_id: str
    player_name: str
    event_type: str  # Rich type, e.g. "pass", "carry", "clearance", etc.

    # Spatial coordinates (in metres, original pitch coordinates)
    start_x: float
    start_y: float
    end_x: float
    end_y: float

    # Event outcomes
    outcome: Optional[str] = None  # e.g. "complete", "incomplete", "on_target", "saved"
    completed: Optional[bool] = None
    under_pressure: bool = False
    turnover: bool = False

    # Possession and sequence
    possession_id: Optional[str] = None
    source: str = "synthetic"  # "synthetic" or "real" or "uploaded"

    # Temporal
    t_start_s: Optional[float] = None
    duration_s: Optional[float] = None

    # Additional metadata
    channel: Optional[str] = None  # pre-computed: left_wide, left_halfspace, etc.
    zone: Optional[str] = None  # pre-computed: defensive_third, middle_third, attacking_third
    auxiliary_data: Dict[str, Any] = field(default_factory=dict)

    def to_model_dict(self) -> dict:
        """Convert this rich event to the model's dict format.

        Returns a dict with keys matching the model's validate_sequence expected inputs.
        All spatial coords are rescaled to the 0-100 attacking-team frame.
        """
        # Rescale coordinates to 0-100 attacking-team frame
        x0, y0 = rescale_coordinates_to_100x100(self.start_x, self.start_y)
        x1, y1 = rescale_coordinates_to_100x100(self.end_x, self.end_y)

        # Flip if this is the away team (so they always attack toward x=100)
        if self.team_id == "away":
            x0, y0 = flip_coordinates_for_away_team(x0, y0)
            x1, y1 = flip_coordinates_for_away_team(x1, y1)

        # Map event type to model's 9-type schema
        model_type = map_event_type_to_model(self.event_type)

        # Fill in defaults
        if self.completed is None:
            # Simple heuristic: if outcome contains "complete" or is None for non-shots
            if self.outcome and "complete" in str(self.outcome).lower():
                self.completed = True
            elif self.event_type in ("shot",):
                self.completed = True  # shot = on target by default in absence of data
            else:
                self.completed = True  # default assumption

        result = {
            "type": model_type,
            "x_start": round(x0, 4),
            "y_start": round(y0, 4),
            "x_end": round(x1, 4),
            "y_end": round(y1, 4),
            "duration_s": self.duration_s if self.duration_s is not None else 1.5,
            "completed": 1 if self.completed else 0,
            "under_pressure": 1 if self.under_pressure else 0,
            "turnover": 1 if self.turnover else 0,
            "possession_before": self.auxiliary_data.get("possession_before", 0),
            "possession_after": self.auxiliary_data.get("possession_after", 1),
            "t_start_s": self.t_start_s if self.t_start_s is not None else self.timestamp_seconds,
        }
        return result


# ----------------------------------------------------------------------------------------
# Rich event generation (synthetic)
# ----------------------------------------------------------------------------------------


def generate_synthetic_event(
    match_id: str,
    timestamp: float,
    period: int,
    team_id: str,
    player_id: str,
    player_name: str,
    event_style: str = "build_up",
    **overrides,
) -> TacticalEvent:
    """Generate a synthetic tactical event matching one of several styles.

    Styles:
      - "build_up": possession-based progression passes
      - "direct": long forward passes
      - "wide": balls into wide channels
      - "press": pressure events
      - "counter": quick transition events
      - "shot": shot on goal
      - "tackle": defensive tackle
      - "recovery": possession regain

    Returns:
        A TacticalEvent instance with realistic synthetic data.
    """
    import random
    import math

    # Style-specific parameters
    styles = {
        "build_up": {
            "event_type": "pass",
            "x0_range": (0, 40),  # starts in defensive/middle third
            "y0_range": (20, 80),
            "x1_range": (25, 70),  # progresses toward middle
            "y1_range": (20, 80),
            "duration": (1.0, 3.0),
            "completed": True,
            "under_pressure": True,
        },
        "direct": {
            "event_type": "pass",
            "x0_range": (0, 30),
            "y0_range": (40, 60),
            "x1_range": (60, 95),  # long forward progression
            "y1_range": (35, 65),
            "duration": (2.0, 4.0),
            "completed": True,
            "under_pressure": False,
        },
        "wide": {
            "event_type": "pass",
            "x0_range": (0, 50),
            "y0_range": (5, 15),  # wide starting area
            "x1_range": (30, 75),
            "y1_range": (85, 95),  # ends very wide
            "duration": (1.0, 2.0),
            "completed": True,
            "under_pressure": False,
        },
        "press": {
            "event_type": "pressure",
            "x0_range": (50, 100),  # in opponent half
            "y0_range": (20, 80),
            "x1_range": (50, 100),
            "y1_range": (20, 80),
            "duration": (0.5, 2.0),
            "completed": False,  # pressure often leads to turnover
            "under_pressure": True,
            "turnover": True,
        },
        "counter": {
            "event_type": "carry",
            "x0_range": (0, 30),  # starts in own half/midfield
            "y0_range": (30, 70),
            "x1_range": (50, 85),  # quick forward progression
            "y1_range": (30, 70),
            "duration": (3.0, 6.0),
            "completed": True,
            "under_pressure": True,
            "turnover": False,
        },
        "shot": {
            "event_type": "shot",
            "x0_range": (25, 55),
            "y0_range": (20, 80),
            "x1_range": (None, None),  # shot doesn't have meaningful end_x/y for model
            "y1_range": (None, None),
            "duration": (0.5, 1.0),
            "completed": True,  # shot outcome
            "under_pressure": True,
        },
        "tackle": {
            "event_type": "tackle",
            "x0_range": (50, 100),  # in opponent half
            "y0_range": (20, 80),
            "x1_range": (50, 100),
            "y1_range": (20, 80),
            "duration": (0.5, 1.5),
            "completed": True,  # won the ball
            "under_pressure": True,
            "turnover": True,
        },
        "recovery": {
            "event_type": "recovery",
            "x0_range": (50, 100),  # in opponent half
            "y0_range": (20, 80),
            "x1_range": (50, 100),
            "y1_range": (20, 80),
            "duration": (1.0, 3.0),
            "completed": True,  # regained possession
            "under_pressure": False,
            "turnover": True,
        },
    }

    style = styles.get(event_style, styles["build_up"])

    # Apply style defaults, then override with any provided overrides
    params = {}
    for k, v in style.items():
        params[k] = v
    params.update(overrides)

    # Generate coordinates
    x0 = random.uniform(*params["x0_range"])
    y0 = random.uniform(*params["y0_range"])
    x1 = random.uniform(*params["x1_range"]) if params["x1_range"][0] is not None else x0
    y1 = random.uniform(*params["y1_range"]) if params["y1_range"][0] is not None else y0

    # Determine outcome
    completed = params["completed"]
    under_pressure = params["under_pressure"]
    turnover = params.get("turnover", False)

    # Build auxiliary data
    auxiliary = {}
    if event_style == "counter":
        # Counterattack: possession regained -> forward progression
        auxiliary["possession_before"] = 0  # opponent had possession
        auxiliary["possession_after"] = 1  # now our team has it
    elif event_style == "recovery":
        auxiliary["possession_before"] = 0
        auxiliary["possession_after"] = 1
    elif event_style == "press":
        auxiliary["possession_before"] = 1  # still own possession under pressure
        auxiliary["possession_after"] = 1 - int(turnover)

    return TacticalEvent(
        match_id=match_id,
        timestamp_seconds=timestamp,
        period=period,
        team_id=team_id,
        player_id=player_id,
        player_name=player_name,
        event_type=params["event_type"],
        start_x=x0,
        start_y=y0,
        end_x=x1,
        end_y=y1,
        completed=completed,
        under_pressure=under_pressure,
        turnover=turnover,
        possession_id=f"possession_{match_id}_{int(timestamp)}",
        source="synthetic",
        t_start_s=timestamp,
        duration_s=random.uniform(*params["duration"]) if isinstance(params["duration"], tuple) else params["duration"],
        auxiliary_data=auxiliary,
    )


# ----------------------------------------------------------------------------------------
# Pre-defined synthetic match sequences
# ----------------------------------------------------------------------------------------


def generate_possession_build_up_sequence(
    match_id: str = "demo_001",
    team_id: str = "home",
    player_ids: Optional[List[str]] = None,
) -> List[TacticalEvent]:
    """Generate a possession-based build-up sequence (controlled_buildup style).

    Simulates a team building up from defence through midfield into the attacking third.
    Uses 7-11 events, typical of a controlled build-up pattern.
    """
    if player_ids is None:
        player_ids = [f"player_{i:02d}" for i in range(1, 12)]

    events = []
    base_ts = 15.0
    players = list(player_ids)

    # Start from defensive third - goalkeeper/distributor
    events.append(generate_synthetic_event(
        match_id=match_id,
        timestamp=base_ts,
        period=1,
        team_id=team_id,
        player_id=players[0],
        player_name="Goalkeeper",
        event_style="build_up",
    ))
    base_ts += 2.5

    # Defender
    events.append(generate_synthetic_event(
        match_id=match_id,
        timestamp=base_ts,
        period=1,
        team_id=team_id,
        player_id=players[1],
        player_name="Centre-Back",
        event_style="build_up",
    ))
    base_ts += 2.0

    # Midfielder - first progression
    events.append(generate_synthetic_event(
        match_id=match_id,
        timestamp=base_ts,
        period=1,
        team_id=team_id,
        player_id=players[2],
        player_name="Defensive Mid",
        event_style="build_up",
    ))
    base_ts += 2.0

    # Wing build
    events.append(generate_synthetic_event(
        match_id=match_id,
        timestamp=base_ts,
        period=1,
        team_id=team_id,
        player_id=players[3],
        player_name="Left Winger",
        event_style="wide",
    ))
    base_ts += 2.5

    # Midfield progression
    events.append(generate_synthetic_event(
        match_id=match_id,
        timestamp=base_ts,
        period=1,
        team_id=team_id,
        player_id=players[4],
        player_name="Attacking Mid",
        event_style="build_up",
    ))
    base_ts += 2.0

    # Final third entry
    events.append(generate_synthetic_event(
        match_id=match_id,
        timestamp=base_ts,
        period=1,
        team_id=team_id,
        player_id=players[5],
        player_name="Right Winger",
        event_style="build_up",
    ))
    base_ts += 2.5

    # Shot attempt
    events.append(generate_synthetic_event(
        match_id=match_id,
        timestamp=base_ts,
        period=1,
        team_id=team_id,
        player_id=players[6],
        player_name="Striker",
        event_style="shot",
    ))

    return events


def generate_direct_progression_sequence(
    match_id: str = "demo_001",
    team_id: str = "home",
    player_ids: Optional[List[str]] = None,
) -> List[TacticalEvent]:
    """Generate a direct progression sequence (direct_progression style).

    Simulates a team playing direct balls forward, bypassing midfield.
    Uses fewer events with long forward passes.
    """
    if player_ids is None:
        player_ids = [f"player_{i:02d}" for i in range(1, 11)]

    events = []
    base_ts = 20.0
    players = list(player_ids)

    # Start with a long ball to striker
    events.append(generate_synthetic_event(
        match_id=match_id,
        timestamp=base_ts,
        period=1,
        team_id=team_id,
        player_id=players[0],
        player_name="Right Back",
        event_style="direct",
    ))
    base_ts += 3.0

    # Another direct ball
    events.append(generate_synthetic_event(
        match_id=match_id,
        timestamp=base_ts,
        period=1,
        team_id=team_id,
        player_id=players[1],
        player_name="Centre-Back",
        event_style="direct",
    ))
    base_ts += 2.5

    # Midfielder carries forward
    events.append(generate_synthetic_event(
        match_id=match_id,
        timestamp=base_ts,
        period=1,
        team_id=team_id,
        player_id=players[2],
        player_name="Defensive Mid",
        event_style="counter",
    ))
    base_ts += 4.0

    # Shot
    events.append(generate_synthetic_event(
        match_id=match_id,
        timestamp=base_ts,
        period=1,
        team_id=team_id,
        player_id=players[3],
        player_name="Striker",
        event_style="shot",
    ))

    return events


def generate_high_press_sequence(
    match_id: str = "demo_001",
    team_id: str = "home",
    player_ids: Optional[List[str]] = None,
) -> List[TacticalEvent]:
    """Generate a high press sequence (high_press_turnover style).

    Simulates aggressive pressing in the opponent half leading to turnovers.
    """
    if player_ids is None:
        player_ids = [f"player_{i:02d}" for i in range(1, 12)]

    events = []
    base_ts = 25.0
    players = list(player_ids)

    # First press event
    events.append(generate_synthetic_event(
        match_id=match_id,
        timestamp=base_ts,
        period=1,
        team_id=team_id,
        player_id=players[0],
        player_name="Striker",
        event_style="press",
    ))
    base_ts += 1.5

    # Second press
    events.append(generate_synthetic_event(
        match_id=match_id,
        timestamp=base_ts,
        period=1,
        team_id=team_id,
        player_id=players[1],
        player_name="Left Winger",
        event_style="press",
    ))
    base_ts += 1.5

    # Third press leading to tackle/turnover
    events.append(generate_synthetic_event(
        match_id=match_id,
        timestamp=base_ts,
        period=1,
        team_id=team_id,
        player_id=players[2],
        player_name="Right Midfielder",
        event_style="tackle",
    ))
    base_ts += 2.0

    # Quick counter after turnover
    events.append(generate_synthetic_event(
        match_id=match_id,
        timestamp=base_ts,
        period=1,
        team_id=team_id,
        player_id=players[3],
        player_name="Counter-Attacker",
        event_style="counter",
    ))
    base_ts += 3.5

    # Shot on counter
    events.append(generate_synthetic_event(
        match_id=match_id,
        timestamp=base_ts,
        period=1,
        team_id=team_id,
        player_id=players[4],
        player_name="Striker",
        event_style="shot",
    ))

    return events


# ----------------------------------------------------------------------------------------
# Export helpers
# ----------------------------------------------------------------------------------------


__all__ = [
    "TacticalEvent",
    "map_event_type_to_model",
    "rescale_coordinates_to_100x100",
    "flip_coordinates_for_away_team",
    "generate_synthetic_event",
    "generate_possession_build_up_sequence",
    "generate_direct_progression_sequence",
    "generate_high_press_sequence",
    "ETYPES",
    "ETYPE_TO_ID",
    "T_MAX",
    "MIN_EVENTS",
    "CHANNEL_EDGES_Y",
    "ZONE_EDGES_X",
]
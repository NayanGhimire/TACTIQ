"""TACTIQ AI - Synthetic Match Event Generator.

Generates realistic synthetic football match events simulating live matches
with contrasting tactical identities: possession-based build-up, direct play,
high pressing, compact defending, and counterattacking.

All events are explicitly tagged as source='synthetic'.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import PITCH_LENGTH_M, PITCH_WIDTH_M, SOURCE_SYNTHETIC
from ..schemas import TacticalEvent

# Re-export from schemas
__all__ = [
    "generate_full_synthetic_match",
    "generate_tactical_sequence",
    "generate_single_match_scenario",
    "tactical_identities",
    "TACTICAL_IDENTITIES",
    "TEAM_NAMES",
]

# ----------------------------------------------------------------------------------------
# Team definitions
# ----------------------------------------------------------------------------------------

TEAM_NAMES = [
    ("Arsenal", "arsenal"),
    ("Manchester City", "mancity"),
    ("Liverpool", "liverpool"),
    ("Real Madrid", "realmadrid"),
    ("Barcelona", "barcelona"),
    ("Bayern Munich", "bayern"),
    ("PSG", "psg"),
    ("Inter Milan", "inter"),
    ("Chelsea", "chelsea"),
    ("Manchester United", "united"),
]


def get_team_players(team_name: str, n: int = 11) -> List[Dict[str, str]]:
    """Generate n player dictionaries for a team."""
    positions = [
        ("Goalkeeper", "GK"),
        ("Right-Back", "RB"),
        ("Right-Centre-Back", "RCB"),
        ("Left-Centre-Back", "LCB"),
        ("Left-Back", "LB"),
        ("Right-Back-Mid", "RM"),
        ("Right-Centre-Mid", "RCM"),
        ("Left-Centre-Mid", "LCM"),
        ("Right-Winger", "RW"),
        ("Left-Winger", "LW"),
        ("Striker", "ST"),
    ]
    players = []
    for i in range(n):
        pos_name, pos_abbr = positions[i % len(positions)]
        players.append({
            "player_id": f"p{team_name[:3].lower()}_{i:02d}",
            "player_name": f"{pos_name} {i+1}",
            "position": pos_abbr,
            "team": team_name,
        })
    return players


# ----------------------------------------------------------------------------------------
# Tactical identities
# ----------------------------------------------------------------------------------------


@dataclass
class TacticalIdentity:
    """Definition of a synthetic tactical identity."""
    name: str
    description: str
    # Build-up style
    build_up_style: str  # "short", "direct", "mixed"
    build_up_pass_length_mean: float  # meters
    build_up_completion_rate: float
    # Progression style
    progression_style: str  # "central", "wide", "mixed"
    final_third_style: str  # "patient", "direct", "crosses"
    # Defensive style
    defensive_line: str  # "high", "mid", "low"
    pressing_intensity: str  # "high", "medium", "low"
    defensive_width: str  # "wide", "narrow"
    # Transition style
    transition_style: str  # "counter", "immediate_press", "retain"
    avg_possession_passes: int
    avg_counter_speed: float  # m/s estimated


TACTICAL_IDENTITIES: Dict[str, TacticalIdentity] = {
    "possession_buildup": TacticalIdentity(
        name="Possession-Based Build-Up",
        description="Short passing, patient progression, central buildup through midfield",
        build_up_style="short",
        build_up_pass_length_mean=12.0,
        build_up_completion_rate=0.92,
        progression_style="central",
        final_third_style="patient",
        defensive_line="high",
        pressing_intensity="medium",
        defensive_width="narrow",
        transition_style="retain",
        avg_possession_passes=8,
        avg_counter_speed=2.5,
    ),
    "direct_play": TacticalIdentity(
        name="Direct Play",
        description="Long balls bypassing midfield, quick transitions to a target striker",
        build_up_style="direct",
        build_up_pass_length_mean=30.0,
        build_up_completion_rate=0.70,
        progression_style="central",
        final_third_style="direct",
        defensive_line="mid",
        pressing_intensity="low",
        defensive_width="narrow",
        transition_style="counter",
        avg_possession_passes=4,
        avg_counter_speed=5.5,
    ),
    "high_press": TacticalIdentity(
        name="High Press",
        description="Aggressive pressing in opponent's half, quick turnovers and counters",
        build_up_style="short",
        build_up_pass_length_mean=15.0,
        build_up_completion_rate=0.80,
        progression_style="wide",
        final_third_style="direct",
        defensive_line="high",
        pressing_intensity="high",
        defensive_width="wide",
        transition_style="immediate_press",
        avg_possession_passes=6,
        avg_counter_speed=4.5,
    ),
    "compact_defense": TacticalIdentity(
        name="Compact Defending",
        description="Defensive shape stays compact, absorbs pressure and hits on the counter",
        build_up_style="mixed",
        build_up_pass_length_mean=18.0,
        build_up_completion_rate=0.88,
        progression_style="central",
        final_third_style="direct",
        defensive_line="low",
        pressing_intensity="low",
        defensive_width="narrow",
        transition_style="counter",
        avg_possession_passes=3,
        avg_counter_speed=5.0,
    ),
    "counterattacking": TacticalIdentity(
        name="Counter-Attacking",
        description="Defensive posture, quick transitions and vertical runs when possession is won",
        build_up_style="short",
        build_up_pass_length_mean=10.0,
        build_up_completion_rate=0.85,
        progression_style="wide",
        final_third_style="direct",
        defensive_line="low",
        pressing_intensity="medium",
        defensive_width="wide",
        transition_style="counter",
        avg_possession_passes=5,
        avg_counter_speed=6.0,
    ),
}


def tactical_identities() -> Dict[str, TacticalIdentity]:
    """Return the dict of all tactical identities."""
    return TACTICAL_IDENTITIES


# ----------------------------------------------------------------------------------------
# Coordinate helpers
# ----------------------------------------------------------------------------------------


def _to_metres(x_100: float, y_100: float) -> Tuple[float, float]:
    """Convert 0-100 coordinates to pitch metres."""
    x_m = x_100 * PITCH_LENGTH_M / 100.0
    y_m = y_100 * PITCH_WIDTH_M / 100.0
    return x_m, y_m


def _in_defensive_third(x_100: float) -> bool:
    return x_100 < 33.3


def _in_middle_third(x_100: float) -> bool:
    return 33.3 <= x_100 < 66.7


def _in_attacking_third(x_100: float) -> bool:
    return x_100 >= 66.7


def _get_channel(y_100: float) -> str:
    if y_100 < 20:
        return "left_wide"
    elif y_100 < 38:
        return "left_halfspace"
    elif y_100 < 62:
        return "central"
    elif y_100 < 80:
        return "right_halfspace"
    else:
        return "right_wide"


def _get_zone(x_100: float) -> str:
    if x_100 < 33.3:
        return "defensive_third"
    elif x_100 < 66.7:
        return "middle_third"
    else:
        return "attacking_third"


# ----------------------------------------------------------------------------------------
# Event generation engine
# ----------------------------------------------------------------------------------------


@dataclass
class PlayerState:
    """Track a player's position and involvement."""
    player_id: str
    player_name: str
    position: str
    team: str
    x: float = 50.0
    y: float = 50.0
    touches: int = 0
    passes: int = 0
    pressures: int = 0
    tackles: int = 0
    shots: int = 0
    goals: int = 0


@dataclass
class MatchState:
    """Track overall match state."""
    match_id: str
    home_team: str
    away_team: str
    home_score: int = 0
    away_score: int = 0
    period: int = 1
    clock_seconds: float = 0.0
    current_possession_team: str = "home"
    possession_id: str = "poss_001"
    players: Dict[str, PlayerState] = None
    events: List[dict] = None
    event_counter: int = 0

    def __post_init__(self):
        if self.players is None:
            self.players = {}
        if self.events is None:
            self.events = []

    def elapsed_minutes(self) -> float:
        return self.clock_seconds / 60.0


def _make_event(
    match: MatchState,
    event_type: str,
    player: PlayerState,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    duration: float,
    completed: bool,
    under_pressure: bool,
    turnover: bool,
    outcome: Optional[str] = None,
    **extra,
) -> dict:
    """Create a single event dict (model-100-frame coordinates)."""
    match.event_counter += 1
    ts = match.clock_seconds + duration / 2.0

    event = {
        "event_id": f"event_{match.match_id}_{match.event_counter:05d}",
        "match_id": match.match_id,
        "timestamp_seconds": round(ts, 3),
        "period": match.period,
        "team_id": player.team,
        "player_id": player.player_id,
        "player_name": player.player_name,
        "position": player.position,
        "event_type": event_type,
        "start_x": round(x0, 4),
        "start_y": round(y0, 4),
        "end_x": round(x1, 4),
        "end_y": round(y1, 4),
        "x_start": round(x0, 4),  # alias for model adapter
        "y_start": round(y0, 4),
        "x_end": round(x1, 4),
        "y_end": round(y1, 4),
        "duration_s": round(duration, 2),
        "outcome": outcome if outcome else ("complete" if completed else "incomplete"),
        "completed": completed,
        "under_pressure": under_pressure,
        "turnover": turnover,
        "possession_id": match.possession_id,
        "possession_before": 1 if match.current_possession_team == player.team else 0,
        "possession_after": 1 if match.current_possession_team == player.team else 0,
        "channel": _get_channel(y1),
        "zone": _get_zone(x1),
        "source": SOURCE_SYNTHETIC,
        **extra,
    }

    # Update player state
    player.touches += 1
    player.x, player.y = x1, y1

    if event_type == "pass" and completed:
        player.passes += 1

    if match.current_possession_team == player.team:
        if event_type == "pass":
            player.passes += 1

    if turnover and match.current_possession_team == player.team:
        match.current_possession_team = "away" if player.team == "home" else "home"
        match.possession_id = f"poss_{match.match_id}_{match.event_counter:03d}"

    match.events.append(event)
    return event


def _generate_pass(
    match: MatchState,
    player: PlayerState,
    identity: TacticalIdentity,
    x0: float = None,
    y0: float = None,
) -> dict:
    """Generate a pass event based on the team's tactical identity."""
    if x0 is None:
        x0 = player.x
    if y0 is None:
        y0 = player.y

    # Determine pass target based on identity
    if identity.build_up_style == "short":
        length = random.gauss(identity.build_up_pass_length_mean, 5.0)
    elif identity.build_up_style == "direct":
        length = random.gauss(identity.build_up_pass_length_mean, 8.0)
    else:
        length = random.gauss(18.0, 7.0)

    length = max(3.0, min(60.0, length))

    # Direction: attacking team should pass forward
    if player.team == "home":
        x_dir = 1  # home attacks toward opponent goal at x=100
    else:
        x_dir = -1  # away team already flipped, so forward is still +x in attacking frame

    # Convert to model frame (always attacking toward x=100)
    # For home team, forward = +x
    # For away team: their events are flipped when loaded, so their forward direction in
    # their own frame is toward their opponent's goal (original x=0 direction),
    # but after flipping it becomes +x. So in the raw data we generate, away team
    # should have forward direction as -x.
    if player.team == "home":
        angle = random.uniform(-0.5, 0.5)  # slight variation
        x1 = x0 + length * math.cos(angle)
        y1 = y0 + length * math.sin(angle)
    else:
        # Away team: in raw coords, they attack toward x=0
        # So in raw space, forward is -x
        angle = random.uniform(-0.5, 0.5)
        x1 = x0 - length * math.cos(angle)
        y1 = y0 + length * math.sin(angle)

    # Boundary checks
    x1 = max(0.01, min(99.99, x1))
    y1 = max(0.01, min(99.99, y1))

    # Completion rate
    completed = random.random() < identity.build_up_completion_rate
    duration = max(0.5, length / 20.0 + random.gauss(0, 0.3))

    return _make_event(
        match, "pass", player, x0, y0, x1, y1, duration, completed,
        under_pressure=random.random() < 0.3,
        turnover=not completed and random.random() < 0.8,
        outcome="complete" if completed else "incomplete",
    )


def _generate_carry(
    match: MatchState,
    player: PlayerState,
    identity: TacticalIdentity,
) -> dict:
    """Generate a carry event."""
    x0, y0 = player.x, player.y

    if identity.transition_style == "counter":
        length = random.gauss(20.0, 5.0)
    else:
        length = random.gauss(6.0, 3.0)
    length = max(1.0, min(30.0, length))

    if player.team == "home":
        angle = random.uniform(-0.3, 0.3)
        x1 = x0 + length * math.cos(angle)
        y1 = y0 + length * math.sin(angle)
    else:
        angle = random.uniform(-0.3, 0.3)
        x1 = x0 - length * math.cos(angle)
        y1 = y0 + length * math.sin(angle)

    x1 = max(0.01, min(99.99, x1))
    y1 = max(0.01, min(99.99, y1))

    duration = max(1.0, length / 5.0 + random.gauss(0, 0.3))

    return _make_event(
        match, "carry", player, x0, y0, x1, y1, duration, True,
        under_pressure=random.random() < 0.4,
        turnover=False,
    )


def _generate_cross(
    match: MatchState,
    player: PlayerState,
    identity: TacticalIdentity,
) -> dict:
    """Generate a cross event from wide area."""
    x0, y0 = player.x, player.y

    # Cross from wide area
    if y0 < 50:
        y1 = random.uniform(75, 95)  # cross to far post
    else:
        y1 = random.uniform(5, 25)  # cross to near post

    x1 = random.uniform(80, 95)
    duration = random.uniform(0.8, 1.5)
    completed = random.random() < 0.25  # crosses are hard to complete

    return _make_event(
        match, "cross", player, x0, y0, x1, y1, duration, completed,
        under_pressure=True,
        turnover=not completed,
    )


def _generate_shot(
    match: MatchState,
    player: PlayerState,
    identity: TacticalIdentity,
) -> dict:
    """Generate a shot event."""
    x0, y0 = player.x, player.y

    # Shot should be in attacking third or just outside box
    if player.team == "home":
        x1 = 101.0  # off the pitch - goal
        on_target = random.random() < 0.35
        goal = on_target and random.random() < 0.40
    else:
        x1 = -1.0  # away team shooting at x=0 (their opponent's goal)
        on_target = random.random() < 0.35
        goal = on_target and random.random() < 0.40

    if goal:
        outcome = "goal"
        match.home_score += 1 if player.team == "home" else 0
        match.away_score += 0 if player.team == "home" else 1
    elif on_target:
        outcome = "on_target"
    else:
        outcome = "off_target"

    duration = random.uniform(0.3, 0.8)

    event = _make_event(
        match, "shot", player, x0, y0, x1, y0, duration, on_target,
        under_pressure=True,
        turnover=True,
        outcome=outcome,
        goal=goal,
    )

    if goal:
        player.goals += 1

    return event


def _generate_pressure(
    match: MatchState,
    player: PlayerState,
    identity: TacticalIdentity,
) -> dict:
    """Generate a pressure event."""
    x0, y0 = player.x, player.y

    # Pressure in attacking or middle third
    duration = random.uniform(1.0, 3.0)
    turn_event = random.random() < 0.4

    return _make_event(
        match, "pressure", player, x0, y0, x0, y0, duration, False,
        under_pressure=True,
        turnover=turn_event,
        outcome="pressure_lost" if turn_event else "pressure_maintained",
    )


def _generate_tackle(
    match: MatchState,
    player: PlayerState,
    identity: TacticalIdentity,
) -> dict:
    """Generate a tackle event."""
    x0, y0 = player.x, player.y

    won = random.random() < 0.65
    duration = random.uniform(0.8, 2.0)

    event = _make_event(
        match, "tackle", player, x0, y0, x0, y0, duration, won,
        under_pressure=True,
        turnover=won,
        outcome="won" if won else "lost",
    )

    if won:
        # Turnover to tackling team's possession
        match.current_possession_team = player.team
        match.possession_id = f"poss_{match.match_id}_{match.event_counter:03d}"

    return event


def _generate_recovery(
    match: MatchState,
    player: PlayerState,
    identity: TacticalIdentity,
) -> dict:
    """Generate a recovery event (loose ball or interception)."""
    x0, y0 = player.x, player.y

    duration = random.uniform(1.0, 3.0)

    event = _make_event(
        match, "recovery", player, x0, y0, x0, y0, duration, True,
        under_pressure=False,
        turnover=True,
        outcome="recovery",
    )

    # Recovery = regain possession
    match.current_possession_team = player.team
    match.possession_id = f"poss_{match.match_id}_{match.event_counter:03d}"

    return event


def _generate_ball_loss(
    match: MatchState,
    player: PlayerState,
    identity: TacticalIdentity,
) -> dict:
    """Generate a ball loss event."""
    x0, y0 = player.x, player.y

    duration = random.uniform(0.5, 2.0)

    event = _make_event(
        match, "ball_loss", player, x0, y0, x0, y0, duration, False,
        under_pressure=True,
        turnover=True,
        outcome="ball_loss",
    )

    # Ball loss = lose possession
    match.current_possession_team = "away" if player.team == "home" else "home"
    match.possession_id = f"poss_{match.match_id}_{match.event_counter:03d}"

    return event


# ----------------------------------------------------------------------------------------
# Possession sequence generation
# ----------------------------------------------------------------------------------------


def _get_active_team_identity(
    match: MatchState,
    identities: Dict[str, TacticalIdentity],
) -> TacticalIdentity:
    """Get the tactical identity for the team currently in possession."""
    team_key = "home" if match.current_possession_team == "home" else "away"
    if team_key not in identities:
        return identities.get("possession_buildup")
    return identities[team_key]


def _generate_possession_sequence(
    match: MatchState,
    identity: TacticalIdentity,
    owning_team: str,
    length_range: Tuple[int, int] = (5, 12),
) -> List[dict]:
    """Generate a sequence of events for one possession."""
    events = []
    seq_length = random.randint(*length_range)

    # Get players for the owning team
    owning_players = [p for p in match.players.values() if p.team == owning_team]
    if not owning_players:
        return events

    current_player = random.choice(owning_players)

    for i in range(seq_length):
        match.clock_seconds += random.uniform(0.5, 3.0)

        # Decide what happens
        x_zone = _get_zone(current_player.x)
        is_attacking_third = _get_zone(current_player.x) == "attacking_third"
        is_defensive_third = _get_zone(current_player.x) == "defensive_third"

        # Probability of different event types
        roll = random.random()

        if player_in_possession := True:
            if is_attacking_third and roll < 0.15:
                # Shot attempt
                ev = _generate_shot(match, current_player, identity)
                events.append(ev)
                break  # possession likely ends after shot
            elif is_attacking_third and roll < 0.25 and identity.final_third_style in ("direct", "crosses"):
                # Cross from wide
                if _get_channel(current_player.y) in ("left_wide", "right_wide"):
                    ev = _generate_cross(match, current_player, identity)
                    events.append(ev)
                    current_player = random.choice(owning_players)
                else:
                    ev = _generate_pass(match, current_player, identity)
                    events.append(ev)
                    if not ev["turnover"]:
                        current_player = owning_players[event_counter % len(owning_players)]
                    else:
                        break
            elif roll < 0.1 and identity.pressing_intensity == "high":
                # Pressure leading to turnover
                ev = _generate_pressure(match, current_player, identity)
                events.append(ev)
                if ev["turnover"]:
                    break
            else:
                # Regular pass or carry
                if roll < 0.15:
                    ev = _generate_carry(match, current_player, identity)
                else:
                    ev = _generate_pass(match, current_player, identity)

                events.append(ev)
                if ev["turnover"]:
                    break
                else:
                    current_player = owning_players[len(events) % len(owning_players)]

    return events


# We need math module
import math


def generate_full_synthetic_match(
    home_team: str = "Arsenal",
    away_team: str = "Liverpool",
    home_identity: str = "possession_buildup",
    away_identity: str = "counterattacking",
    duration_minutes: int = 90,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate a complete synthetic football match.

    Args:
        home_team: Home team name
        away_team: Away team name
        home_identity: Tactical identity key for home team
        away_identity: Tactical identity key for away team
        duration_minutes: Match duration in minutes
        seed: Random seed for reproducibility

    Returns:
        Dictionary with match metadata and events list
    """
    random.seed(seed)

    match_id = f"{home_team.lower().replace(' ', '_')}_vs_{away_team.lower().replace(' ', '_')}_{seed}"

    # Set up match state
    match = MatchState(
        match_id=match_id,
        home_team=home_team,
        away_team=away_team,
        period=1,
    )

    # Set up players
    home_players = get_team_players(home_team)
    away_players = get_team_players(away_team)
    for p in home_players:
        match.players[p["player_id"]] = PlayerState(
            player_id=p["player_id"],
            player_name=p["player_name"],
            position=p["position"],
            team="home",
            x=50.0,
            y=50.0,
        )
    for p in away_players:
        match.players[p["player_id"]] = PlayerState(
            player_id=p["player_id"],
            player_name=p["player_name"],
            position=p["position"],
            team="away",
            x=50.0,
            y=50.0,
        )

    # Set initial positions
    for pid in [p["player_id"] for p in home_players]:
        p = match.players[pid]
        p.x = random.uniform(50, 80)
        p.y = random.uniform(20, 80)

    # Half-time handling
    total_seconds = duration_minutes * 60
    half_time = duration_minutes * 30 if duration_minutes == 90 else total_seconds / 2

    # Generate events
    match.clock_seconds = 0.0
    match.current_possession_team = "home"
    match.possession_id = "poss_001"

    home_identity_obj = TACTICAL_IDENTITIES.get(home_identity, TACTICAL_IDENTITIES["possession_buildup"])
    away_identity_obj = TACTICAL_IDENTITIES.get(away_identity, TACTICAL_IDENTITIES["counterattacking"])

    while match.clock_seconds < total_seconds:
        # Switch to second half
        if match.clock_seconds >= half_time and match.period == 1:
            match.period = 2
            # Reset positions
            for p in match.players.values():
                p.x = random.uniform(20, 80)
                p.y = random.uniform(20, 80)

        match.clock_seconds += random.uniform(1.0, 5.0)

        # Get the identity for the team in possession
        team_in_possession = match.current_possession_team
        if team_in_possession == "home":
            identity = home_identity_obj
        else:
            identity = away_identity_obj

        # Generate a possession sequence
        seq = _generate_possession_sequence(
            match, identity, team_in_possession,
            length_range=(3, identity.avg_possession_passes + 3),
        )

    return {
        "match_id": match_id,
        "home_team": home_team,
        "away_team": away_team,
        "home_identity": home_identity,
        "away_identity": away_identity,
        "home_score": match.home_score,
        "away_score": match.away_score,
        "duration_minutes": duration_minutes,
        "total_events": len(match.events),
        "events": match.events,
        "metadata": {
            "generated_at": "synthetic",
            "seed": seed,
            "source": SOURCE_SYNTHETIC,
        }
    }


def generate_single_match_scenario(
    scenario: str = "possession_vs_counter",
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate a single match based on a named scenario.

    Scenarios:
      - "possession_vs_counter": possession team vs counter-attacking team
      - "press_vs_compact": high press team vs compact defending team
      - "direct_vs_direct": two direct-play teams (end-to-end game)
      - "wide_vs_central": wide-attacking team vs central-progressing team
    """
    scenarios = {
        "possession_vs_counter": {
            "home_team": "Manchester City",
            "away_team": "Real Madrid",
            "home_identity": "possession_buildup",
            "away_identity": "counterattacking",
            "duration_minutes": 90,
        },
        "press_vs_compact": {
            "home_team": "Liverpool",
            "away_team": "Atletico Madrid",
            "home_identity": "high_press",
            "away_identity": "compact_defense",
            "duration_minutes": 90,
        },
        "direct_vs_direct": {
            "home_team": "Manchester United",
            "away_team": "Chelsea",
            "home_identity": "direct_play",
            "away_identity": "direct_play",
            "duration_minutes": 90,
        },
        "wide_vs_central": {
            "home_team": "Barcelona",
            "away_team": "Inter Milan",
            "home_identity": "high_press",
            "away_identity": "compact_defense",
            "duration_minutes": 90,
        },
    }

    config = scenarios.get(scenario, scenarios["possession_vs_counter"])
    return generate_full_synthetic_match(
        home_team=config["home_team"],
        away_team=config["away_team"],
        home_identity=config["home_identity"],
        away_identity=config["away_identity"],
        duration_minutes=config["duration_minutes"],
        seed=seed,
    )


def generate_tactical_sequence(
    identity_name: str,
    team_id: str = "home",
    match_id: str = "seq_demo",
    n_events: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate a focused tactical sequence for model testing.

    This creates events that match the model's training rule-labels,
    so the classifier should produce a confident prediction.
    """
    random.seed(seed)
    identity = TACTICAL_IDENTITIES.get(identity_name)
    if identity is None:
        raise ValueError(f"Unknown identity '{identity_name}'. Available: {list(TACTICAL_IDENTITIES.keys())}")

    match = MatchState(
        match_id=match_id,
        home_team="TACTIQ Home",
        away_team="TACTIQ Away",
    )

    # Create players
    players = []
    for i in range(n_events):
        pid = f"p{i:02d}"
        match.players[pid] = PlayerState(
            player_id=pid,
            player_name=f"Player {i+1}",
            position="ST",
            team=team_id,
            x=random.uniform(20, 80),
            y=random.uniform(20, 80),
        )
        players.append(match.players[pid])

    match.current_possession_team = team_id
    match.clock_seconds = 60.0

    # Generate events matching the rule-label pattern
    events = []
    current_player = players[0]
    match.players[current_player.player_id] = current_player

    for i in range(n_events):
        match.clock_seconds += random.uniform(1.0, 3.0)

        etype = identity.build_up_style
        if i == n_events - 1:
            etype = "shot" if random.random() < 0.5 else "pass"

        if etype == "shot":
            ev = _generate_shot(match, current_player, identity)
        else:
            ev = _generate_pass(match, current_player, identity)

        events.append(ev)
        if ev.get("turnover", False):
            break
        current_player = players[(i + 1) % len(players)]

    return {
        "match_id": match_id,
        "events": events,
        "identity": identity_name,
        "n_events": len(events),
        "source": SOURCE_SYNTHETIC,
    }

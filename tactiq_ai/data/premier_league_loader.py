"""TACTIQ AI - Real Premier League Match Data Loader.

Ingests official StatsBomb Open Data for English Premier League matches.
Converts real match events into normalized TACTIQ AI schemas:
- Converts 120x80 StatsBomb pitch frame to standard 105x68m coordinates
- Extracts authentic Premier League player names, formations, and tactical events
- Maps passes, carries, shots, pressures, tackles, recoveries, interceptions, turnovers
- Guarantees chronological timestamp ordering and strict data integrity
- Only Premier League teams: Arsenal, Man City, Man United, Liverpool, Chelsea, Tottenham, Leicester
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..config import PROJECT_ROOT

logger = logging.getLogger(__name__)

PL_DATA_DIR = PROJECT_ROOT / "tactiq_ai" / "data" / "premier_league_matches"
PL_DATA_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_REAL_PL = "premier_league_statsbomb"

# Real Marquee Premier League Clashes (StatsBomb Open Data IDs - Season 2015/2016)
PREMIER_LEAGUE_MATCHES = {
    "arsenal_vs_mancity": {
        "match_id": 3754296,
        "title": "Arsenal 2 - 1 Manchester City (2015-12-21)",
        "home_team": "Arsenal",
        "away_team": "Manchester City",
        "home_score": 2,
        "away_score": 1,
        "date": "2015-12-21",
        "stadium": "Emirates Stadium, London",
        "description": "Mesut Ozil double assist masterclass; Walcott & Giroud goals vs Yaya Toure curler.",
    },
    "arsenal_vs_manunited": {
        "match_id": 3754171,
        "title": "Arsenal 3 - 0 Manchester United (2015-10-04)",
        "home_team": "Arsenal",
        "away_team": "Manchester United",
        "home_score": 3,
        "away_score": 0,
        "date": "2015-10-04",
        "stadium": "Emirates Stadium, London",
        "description": "Blistering first 20 minutes with Alexis Sanchez brace and Mesut Ozil strike.",
    },
    "mancity_vs_liverpool": {
        "match_id": 3754208,
        "title": "Manchester City 1 - 4 Liverpool (2015-11-21)",
        "home_team": "Manchester City",
        "away_team": "Liverpool",
        "home_score": 1,
        "away_score": 4,
        "date": "2015-11-21",
        "stadium": "Etihad Stadium, Manchester",
        "description": "Jurgen Klopp heavy metal counter-press masterclass; Firmino & Coutinho brilliance.",
    },
    "leicester_vs_manunited": {
        "match_id": 3754186,
        "title": "Leicester City 1 - 1 Manchester United (2015-11-28)",
        "home_team": "Leicester City",
        "away_team": "Manchester United",
        "home_score": 1,
        "away_score": 1,
        "date": "2015-11-28",
        "stadium": "King Power Stadium, Leicester",
        "description": "Historic Premier League moment: Jamie Vardy breaks record scoring in 11 straight games.",
    },
    "chelsea_vs_liverpool": {
        "match_id": 3754098,
        "title": "Chelsea 1 - 3 Liverpool (2015-10-31)",
        "home_team": "Chelsea",
        "away_team": "Liverpool",
        "home_score": 1,
        "away_score": 3,
        "date": "2015-10-31",
        "stadium": "Stamford Bridge, London",
        "description": "Philippe Coutinho double sparks Liverpool comeback at Stamford Bridge.",
    },
    "leicester_vs_arsenal": {
        "match_id": 3754174,
        "title": "Leicester City 2 - 5 Arsenal (2015-09-26)",
        "home_team": "Leicester City",
        "away_team": "Arsenal",
        "home_score": 2,
        "away_score": 5,
        "date": "2015-09-26",
        "stadium": "King Power Stadium, Leicester",
        "description": "End-to-end classic: Alexis Sanchez hat-trick vs Jamie Vardy double.",
    },
    "tottenham_vs_mancity": {
        "match_id": 3754197,
        "title": "Tottenham Hotspur 4 - 1 Manchester City (2015-09-26)",
        "home_team": "Tottenham Hotspur",
        "away_team": "Manchester City",
        "home_score": 4,
        "away_score": 1,
        "date": "2015-09-26",
        "stadium": "White Hart Lane, London",
        "description": "Mauricio Pochettino high-intensity press dismantles City; Kane & Dier on target.",
    },
    "chelsea_vs_arsenal": {
        "match_id": 3754217,
        "title": "Chelsea 2 - 0 Arsenal (2015-09-19)",
        "home_team": "Chelsea",
        "away_team": "Arsenal",
        "home_score": 2,
        "away_score": 0,
        "date": "2015-09-19",
        "stadium": "Stamford Bridge, London",
        "description": "Fierce London Derby: Zouma header and Chambers deflection seal Chelsea win.",
    },
}


def _convert_statsbomb_coords(sb_loc: Optional[List[float]]) -> Tuple[float, float]:
    """Convert StatsBomb 120x80 coordinate frame to 105x68m football pitch frame."""
    if not sb_loc or len(sb_loc) < 2:
        return (52.5, 34.0)
    x = max(0.0, min(120.0, float(sb_loc[0])))
    y = max(0.0, min(80.0, float(sb_loc[1])))
    m_x = round((x / 120.0) * 105.0, 2)
    m_y = round((y / 80.0) * 68.0, 2)
    return (m_x, m_y)


def _parse_statsbomb_event(sb_ev: Dict[str, Any], home_team_name: str, idx: int, match_id: str = "pl_match") -> Optional[Dict[str, Any]]:
    """Convert a single raw StatsBomb event into a normalized TACTIQ TacticalEvent."""
    ev_type_raw = sb_ev.get("type", {}).get("name", "")
    team_name = sb_ev.get("team", {}).get("name", "")
    team_id = "home" if team_name == home_team_name else "away"

    player_name = sb_ev.get("player", {}).get("name", "Unknown Player")
    player_id = f"pl_{sb_ev.get('player', {}).get('id', idx)}"
    pos_name = sb_ev.get("position", {}).get("name", "Player")

    period = int(sb_ev.get("period", 1))
    minute = int(sb_ev.get("minute", 0))
    second = int(sb_ev.get("second", 0))

    # Calculate global match timestamp (seconds)
    ts_seconds = minute * 60.0 + second + (idx * 0.001)

    location = sb_ev.get("location")
    start_x, start_y = _convert_statsbomb_coords(location)
    end_x, end_y = start_x, start_y

    tactiq_type = None
    outcome = "completed"
    completed = True
    under_pressure = bool(sb_ev.get("under_pressure", False))
    turnover = False

    # 1. Passes
    if ev_type_raw == "Pass":
        pass_data = sb_ev.get("pass", {})
        end_loc = pass_data.get("end_location")
        if end_loc:
            end_x, end_y = _convert_statsbomb_coords(end_loc)
        pass_outcome = pass_data.get("outcome", {}).get("name")
        if pass_outcome:
            completed = False
            outcome = pass_outcome.lower()
        else:
            completed = True
            outcome = "completed"

        # Detect crosses
        if pass_data.get("cross", False):
            tactiq_type = "cross"
        else:
            tactiq_type = "pass"

    # 2. Carries
    elif ev_type_raw == "Carry":
        carry_data = sb_ev.get("carry", {})
        end_loc = carry_data.get("end_location")
        if end_loc:
            end_x, end_y = _convert_statsbomb_coords(end_loc)
        tactiq_type = "carry"
        outcome = "completed"
        completed = True

    # 3. Shots
    elif ev_type_raw == "Shot":
        shot_data = sb_ev.get("shot", {})
        end_loc = shot_data.get("end_location")
        if end_loc:
            end_x, end_y = _convert_statsbomb_coords(end_loc)
        shot_outcome = shot_data.get("outcome", {}).get("name", "Off Target")
        if shot_outcome == "Goal":
            outcome = "goal"
            completed = True
        elif shot_outcome in ("Saved", "Saved to Post"):
            outcome = "saved"
            completed = True
        elif shot_outcome == "Blocked":
            outcome = "blocked"
            completed = False
        elif shot_outcome in ("Off T", "Wayward"):
            outcome = "off_target"
            completed = False
        else:
            outcome = "on_target"
            completed = True
        tactiq_type = "shot"

    # 4. Pressures
    elif ev_type_raw == "Pressure":
        tactiq_type = "pressure"
        completed = True
        outcome = "applied"

    # 5. Duels / Tackles
    elif ev_type_raw == "Duel":
        tactiq_type = "tackle"
        duel_type = sb_ev.get("duel", {}).get("type", {}).get("name", "")
        duel_outcome = sb_ev.get("duel", {}).get("outcome", {}).get("name", "")
        completed = "Won" in duel_outcome or "Success" in duel_outcome
        outcome = "won" if completed else "lost"

    # 6. Ball Recoveries
    elif ev_type_raw == "Ball Recovery":
        tactiq_type = "recovery"
        outcome = "won"
        completed = True

    # 7. Interceptions
    elif ev_type_raw == "Interception":
        tactiq_type = "interception"
        outcome = "intercepted"
        completed = True

    # 8. Clearances
    elif ev_type_raw == "Clearance":
        tactiq_type = "clearance"
        outcome = "cleared"
        completed = True

    # 9. Dispossessed / Miscontrols (Turnovers)
    elif ev_type_raw in ("Dispossessed", "Miscontrol"):
        tactiq_type = "ball_loss"
        outcome = "turnover"
        completed = False
        turnover = True

    if not tactiq_type:
        return None

    return {
        "event_id": f"pl_{sb_ev.get('id', idx)[:12]}",
        "match_id": str(match_id),
        "timestamp_seconds": round(ts_seconds, 2),
        "period": period,
        "team_id": team_id,
        "player_id": player_id,
        "player_name": player_name,
        "position": pos_name,
        "event_type": tactiq_type,
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y,
        "completed": completed,
        "outcome": outcome,
        "under_pressure": under_pressure,
        "turnover": turnover,
        "source": SOURCE_REAL_PL,
    }


def download_and_cache_pl_match(match_key: str) -> Dict[str, Any]:
    """Download, convert, and cache a real Premier League match dataset."""
    info = PREMIER_LEAGUE_MATCHES.get(match_key)
    if not info:
        raise ValueError(f"Unknown Premier League match key: {match_key}")

    match_id = info["match_id"]
    local_cache = PL_DATA_DIR / f"{match_key}_{match_id}.json"

    if local_cache.exists():
        with open(local_cache, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Ensure match_id exists on all events
            m_id = str(data.get("match_id", match_id))
            for ev in data.get("events", []):
                ev["match_id"] = m_id
            return data

    logger.info(f"Downloading real Premier League match data: {info['title']}...")
    url = f"https://raw.githubusercontent.com/hudl/open-data/master/data/events/{match_id}.json"

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        raw_events = json.loads(resp.read().decode("utf-8"))

    # Convert all valid events
    parsed_events = []
    for idx, sb_ev in enumerate(raw_events):
        parsed = _parse_statsbomb_event(sb_ev, home_team_name=info["home_team"], idx=idx, match_id=str(match_id))
        if parsed is not None:
            parsed_events.append(parsed)

    # Sort strictly by timestamp
    parsed_events.sort(key=lambda e: float(e["timestamp_seconds"]))
    prev_ts = -1.0
    for ev in parsed_events:
        ts = float(ev["timestamp_seconds"])
        if ts <= prev_ts:
            ts = prev_ts + 0.05
            ev["timestamp_seconds"] = round(ts, 2)
        prev_ts = ts

    match_package = {
        "match_id": str(match_id),
        "match_key": match_key,
        "title": info["title"],
        "home_team": info["home_team"],
        "away_team": info["away_team"],
        "home_score": info["home_score"],
        "away_score": info["away_score"],
        "date": info["date"],
        "stadium": info["stadium"],
        "description": info["description"],
        "source": SOURCE_REAL_PL,
        "competition": "Premier League",
        "total_events": len(parsed_events),
        "events": parsed_events,
    }

    with open(local_cache, "w", encoding="utf-8") as f:
        json.dump(match_package, f, indent=2)

    logger.info(f"Cached {len(parsed_events)} real Premier League events to {local_cache.name}")
    return match_package


def get_available_pl_matches() -> List[Tuple[str, str]]:
    """Return dropdown choices of all available real Premier League matches."""
    return [(info["title"], k) for k, info in PREMIER_LEAGUE_MATCHES.items()]


def load_real_pl_match(match_key: str = "arsenal_vs_mancity") -> Dict[str, Any]:
    """Load a real Premier League match package."""
    return download_and_cache_pl_match(match_key)

"""TACTIQ AI - Football Pitch Visualization.

Plots interactive football pitch with event markers, directional pass arrows,
possession indicators, progressive routes, carries, crosses, shots, recoveries,
and tactical zones using Plotly.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import plotly.graph_objects as go

logger = logging.getLogger(__name__)

PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# Pitch thirds boundaries (in meters)
THIRD_DEF_MAX = 35.0
THIRD_MID_MAX = 70.0

# Channel boundaries (in meters)
CHANNEL_Y = {
    "left_wide": (0.0, 13.6),
    "left_halfspace": (13.6, 25.84),
    "central": (25.84, 42.16),
    "right_halfspace": (42.16, 54.4),
    "right_wide": (54.4, 68.0),
}

# Tactical colors
COLOR_HOME = "#1f77b4"         # Deep cyan-blue
COLOR_AWAY = "#e74c3c"         # Crisp crimson
COLOR_PROGRESSIVE = "#ffd700"  # Vibrant gold
COLOR_PASS_COMPLETE = "#3498db"# Sky blue
COLOR_PASS_INCOMPLETE = "#7f8c8d" # Muted slate
COLOR_CARRY = "#2ecc71"        # Emerald green
COLOR_CROSS = "#e67e22"        # Amber
COLOR_SHOT_GOAL = "#f1c40f"    # Gold star
COLOR_SHOT_TARGET = "#e67e22"  # Bright orange
COLOR_SHOT_OFF = "#95a5a6"     # Grey
COLOR_PRESSURE = "#9b59b6"     # Purple
COLOR_TACKLE = "#c0392b"       # Deep red
COLOR_RECOVERY = "#1abc9c"     # Turquoise
COLOR_TURNOVER = "#d35400"     # Rust


def _rescale_coord(x: float, y: float, is_normalized_100: bool) -> Tuple[float, float]:
    """Ensure coordinates are in pitch meters (0-105, 0-68)."""
    if is_normalized_100:
        return (x * PITCH_LENGTH_M / 100.0, y * PITCH_WIDTH_M / 100.0)
    # If coordinates already look like 0-100 but length > 100, clamp/keep
    if x <= 100.0 and y <= 100.0 and (x > 105.0 or y > 68.0):
        return (x * PITCH_LENGTH_M / 100.0, y * PITCH_WIDTH_M / 100.0)
    return (x, y)


def is_progressive_pass(start_x: float, start_y: float, end_x: float, end_y: float, team_id: str = "home") -> bool:
    """Determine if a pass is progressive according to standard tactical metrics.

    A progressive pass moves the ball:
    - at least 15m towards opponent goal if in defensive half
    - at least 10m towards opponent goal if starting in attacking half
    - into the opponent penalty box
    """
    dx = end_x - start_x
    if team_id == "away":
        dx = -dx  # Away attacks left unless flipped

    # Check entry into box
    if (team_id == "home" and end_x >= 88.5 and 13.84 <= end_y <= 54.16) or \
       (team_id == "away" and end_x <= 16.5 and 13.84 <= end_y <= 54.16):
        return True

    # Distance check
    if start_x < (PITCH_LENGTH_M / 2):
        return dx >= 15.0
    return dx >= 10.0


def create_pitch_figure(
    title: str = "TACTIQ AI — Interactive Tactical Pitch",
    pitch_length_m: float = PITCH_LENGTH_M,
    pitch_width_m: float = PITCH_WIDTH_M,
    show_zones: bool = True,
    show_channels: bool = False,
    home_team_name: str = "Home",
    away_team_name: str = "Away",
    show_attack_direction: bool = True,
    figsize: Tuple[int, int] = (10, 7),
) -> go.Figure:
    """Create a pristine, high-contrast football pitch figure with professional markings.

    Args:
        title: Title banner
        pitch_length_m: Pitch length in meters (default 105m)
        pitch_width_m: Pitch width in meters (default 68m)
        show_zones: Whether to display dashed vertical third dividers
        show_channels: Whether to display tactical five-channel dividers
        home_team_name: Name of home team (attacks right by default)
        away_team_name: Name of away team (attacks left by default)
        show_attack_direction: Whether to render direction indicators
        figsize: Canvas dimensions (width, height in hundreds of px)

    Returns:
        Plotly Figure with pitch layout and grass markings
    """
    fig = go.Figure()

    # 1. Pitch turf: Alternating grass cut stripes (10 stripes across length)
    n_stripes = 10
    stripe_w = pitch_length_m / n_stripes
    turf_dark = "#1c4a27"
    turf_light = "#22592f"

    for i in range(n_stripes):
        fill = turf_light if i % 2 == 0 else turf_dark
        fig.add_shape(
            type="rect",
            x0=i * stripe_w,
            y0=0,
            x1=(i + 1) * stripe_w,
            y1=pitch_width_m,
            fillcolor=fill,
            line=dict(width=0),
            layer="below",
        )

    # 2. Outer boundary line
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=pitch_length_m,
        y1=pitch_width_m,
        line=dict(color="rgba(255, 255, 255, 0.85)", width=2.5),
        layer="below",
    )

    # 3. Halfway line
    half_x = pitch_length_m / 2.0
    half_y = pitch_width_m / 2.0
    fig.add_shape(
        type="line",
        x0=half_x,
        y0=0,
        x1=half_x,
        y1=pitch_width_m,
        line=dict(color="rgba(255, 255, 255, 0.85)", width=2),
        layer="below",
    )

    # 4. Center circle (radius 9.15m) and center spot
    circle_r = 9.15
    fig.add_shape(
        type="circle",
        x0=half_x - circle_r,
        y0=half_y - circle_r,
        x1=half_x + circle_r,
        y1=half_y + circle_r,
        line=dict(color="rgba(255, 255, 255, 0.85)", width=2),
        layer="below",
    )
    fig.add_shape(
        type="circle",
        x0=half_x - 0.4,
        y0=half_y - 0.4,
        x1=half_x + 0.4,
        y1=half_y + 0.4,
        fillcolor="white",
        line=dict(color="white", width=1),
        layer="below",
    )

    # 5. Penalty boxes (16.5m depth, 40.32m width)
    box_w = 16.5
    box_h = 40.32
    box_y0 = (pitch_width_m - box_h) / 2.0
    box_y1 = box_y0 + box_h

    # Left penalty box
    fig.add_shape(
        type="rect",
        x0=0,
        y0=box_y0,
        x1=box_w,
        y1=box_y1,
        line=dict(color="rgba(255, 255, 255, 0.85)", width=2),
        layer="below",
    )

    # Right penalty box
    fig.add_shape(
        type="rect",
        x0=pitch_length_m - box_w,
        y0=box_y0,
        x1=pitch_length_m,
        y1=box_y1,
        line=dict(color="rgba(255, 255, 255, 0.85)", width=2),
        layer="below",
    )

    # 6. Goal boxes (5.5m depth, 18.32m width)
    gbox_w = 5.5
    gbox_h = 18.32
    gbox_y0 = (pitch_width_m - gbox_h) / 2.0
    gbox_y1 = gbox_y0 + gbox_h

    fig.add_shape(
        type="rect",
        x0=0,
        y0=gbox_y0,
        x1=gbox_w,
        y1=gbox_y1,
        line=dict(color="rgba(255, 255, 255, 0.85)", width=1.5),
        layer="below",
    )
    fig.add_shape(
        type="rect",
        x0=pitch_length_m - gbox_w,
        y0=gbox_y0,
        x1=pitch_length_m,
        y1=gbox_y1,
        line=dict(color="rgba(255, 255, 255, 0.85)", width=1.5),
        layer="below",
    )

    # 7. Penalty spots (11.0m from goal line)
    for pen_x in [11.0, pitch_length_m - 11.0]:
        fig.add_shape(
            type="circle",
            x0=pen_x - 0.35,
            y0=half_y - 0.35,
            x1=pen_x + 0.35,
            y1=half_y + 0.35,
            fillcolor="white",
            line=dict(color="white", width=1),
            layer="below",
        )

    # 8. Goals (outside boundary: 2.0m depth, 7.32m wide)
    goal_h = 7.32
    goal_y0 = (pitch_width_m - goal_h) / 2.0
    goal_y1 = goal_y0 + goal_h

    fig.add_shape(
        type="rect",
        x0=-2.0,
        y0=goal_y0,
        x1=0,
        y1=goal_y1,
        line=dict(color="white", width=2),
        fillcolor="rgba(255, 255, 255, 0.15)",
        layer="below",
    )
    fig.add_shape(
        type="rect",
        x0=pitch_length_m,
        y0=goal_y0,
        x1=pitch_length_m + 2.0,
        y1=goal_y1,
        line=dict(color="white", width=2),
        fillcolor="rgba(255, 255, 255, 0.15)",
        layer="below",
    )

    # 9. Tactical Thirds (dashed yellow dividers)
    if show_zones:
        for t_x in [THIRD_DEF_MAX, THIRD_MID_MAX]:
            fig.add_shape(
                type="line",
                x0=t_x,
                y0=0,
                x1=t_x,
                y1=pitch_width_m,
                line=dict(color="rgba(241, 196, 15, 0.4)", width=1.5, dash="dash"),
                layer="below",
            )
        # Zone labels along bottom
        fig.add_annotation(
            x=THIRD_DEF_MAX / 2, y=2.0, text="DEFENSIVE THIRD",
            showarrow=False, font=dict(color="rgba(255, 255, 255, 0.4)", size=9)
        )
        fig.add_annotation(
            x=(THIRD_DEF_MAX + THIRD_MID_MAX) / 2, y=2.0, text="MIDDLE THIRD",
            showarrow=False, font=dict(color="rgba(255, 255, 255, 0.4)", size=9)
        )
        fig.add_annotation(
            x=(THIRD_MID_MAX + pitch_length_m) / 2, y=2.0, text="ATTACKING THIRD",
            showarrow=False, font=dict(color="rgba(255, 255, 255, 0.4)", size=9)
        )

    # 10. Tactical Channels (5 vertical lanes)
    if show_channels:
        for ch_name, (y0, y1) in CHANNEL_Y.items():
            if y1 < pitch_width_m:
                fig.add_shape(
                    type="line",
                    x0=0,
                    y0=y1,
                    x1=pitch_length_m,
                    y1=y1,
                    line=dict(color="rgba(52, 152, 219, 0.35)", width=1, dash="dot"),
                    layer="below",
                )

    # 11. Attacking direction indicators
    if show_attack_direction:
        # Home attacks right
        fig.add_annotation(
            x=25,
            y=pitch_width_m - 2.5,
            text=f"<b>{home_team_name}</b> Attacks ➔",
            showarrow=False,
            font=dict(color="#3498db", size=11),
            bgcolor="rgba(26, 38, 57, 0.75)",
            bordercolor="#3498db",
            borderwidth=1,
            borderpad=4,
        )
        # Away attacks left
        fig.add_annotation(
            x=pitch_length_m - 25,
            y=pitch_width_m - 2.5,
            text=f"⮜ <b>{away_team_name}</b> Attacks",
            showarrow=False,
            font=dict(color="#e74c3c", size=11),
            bgcolor="rgba(44, 24, 24, 0.75)",
            bordercolor="#e74c3c",
            borderwidth=1,
            borderpad=4,
        )

    # Layout configuration
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            x=0.5,
            font=dict(color="white", size=15),
        ),
        xaxis=dict(
            range=[-4, pitch_length_m + 4],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            range=[-3, pitch_width_m + 4],
            showgrid=False,
            showticklabels=False,
            zeroline=False,
        ),
        plot_bgcolor="#16391e",
        paper_bgcolor="#141e17",
        font=dict(color="white"),
        width=figsize[0] * 100,
        height=figsize[1] * 100,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(size=10, color="#ecf0f1"),
            bgcolor="rgba(20, 30, 23, 0.8)",
            bordercolor="#34495e",
            borderwidth=1,
        ),
        margin=dict(l=20, r=20, t=50, b=50),
    )

    return fig


def add_pitch_players(
    fig: go.Figure,
    home_players: List[Dict[str, Any]],
    away_players: List[Dict[str, Any]],
    x_range: Tuple[float, float] = None,
    y_range: Tuple[float, float] = None,
) -> go.Figure:
    """Add player markers to pitch figure when tracking data is available.

    Never invents positions: only plots players with explicit 'x' and 'y' coordinates.
    """
    player_size = 13

    if home_players:
        valid_home = [p for p in home_players if "x" in p and "y" in p]
        if valid_home:
            hx = [p["x"] * PITCH_LENGTH_M / 100.0 if p["x"] <= 100 and p["x"] > 105 else p["x"] for p in valid_home]
            hy = [p["y"] * PITCH_WIDTH_M / 100.0 if p["y"] <= 100 and p["y"] > 68 else p["y"] for p in valid_home]
            names = [p.get("name", p.get("player_name", "Home")) for p in valid_home]
            positions = [p.get("position", "") for p in valid_home]
            hover_text = [f"<b>{n}</b> ({pos})<br>Team: Home" for n, pos in zip(names, positions)]

            fig.add_trace(
                go.Scatter(
                    x=hx,
                    y=hy,
                    mode="markers+text",
                    marker=dict(
                        size=player_size,
                        color=COLOR_HOME,
                        line=dict(width=2, color="white"),
                        symbol="circle",
                    ),
                    text=positions,
                    textposition="middle center",
                    textfont=dict(color="white", size=8, family="Arial Black"),
                    hovertext=hover_text,
                    hoverinfo="text",
                    name="Home Squad",
                    showlegend=True,
                )
            )

    if away_players:
        valid_away = [p for p in away_players if "x" in p and "y" in p]
        if valid_away:
            ax = [p["x"] * PITCH_LENGTH_M / 100.0 if p["x"] <= 100 and p["x"] > 105 else p["x"] for p in valid_away]
            ay = [p["y"] * PITCH_WIDTH_M / 100.0 if p["y"] <= 100 and p["y"] > 68 else p["y"] for p in valid_away]
            names = [p.get("name", p.get("player_name", "Away")) for p in valid_away]
            positions = [p.get("position", "") for p in valid_away]
            hover_text = [f"<b>{n}</b> ({pos})<br>Team: Away" for n, pos in zip(names, positions)]

            fig.add_trace(
                go.Scatter(
                    x=ax,
                    y=ay,
                    mode="markers+text",
                    marker=dict(
                        size=player_size,
                        color=COLOR_AWAY,
                        line=dict(width=2, color="white"),
                        symbol="circle",
                    ),
                    text=positions,
                    textposition="middle center",
                    textfont=dict(color="white", size=8, family="Arial Black"),
                    hovertext=hover_text,
                    hoverinfo="text",
                    name="Away Squad",
                    showlegend=True,
                )
            )

    return fig


def add_directional_pass_arrows(
    fig: go.Figure,
    passes: List[Dict[str, Any]],
    team_id: Optional[str] = None,
    highlight_ids: Optional[List[str]] = None,
    max_arrows: int = 50,
) -> go.Figure:
    """Draw true directional pass arrows for completed and progressive passes.

    Uses Plotly annotations for clean arrowheads and Scatter traces for hoverability and customdata.
    """
    highlight_ids = highlight_ids or []
    filtered = [p for p in passes if p.get("event_type") == "pass"]
    if team_id:
        filtered = [p for p in filtered if p.get("team_id") == team_id]

    # Focus on the most recent passes up to max_arrows to maintain pitch clarity
    sample = filtered[-max_arrows:] if len(filtered) > max_arrows else filtered

    # Separate passes into regular completed, progressive, and incomplete
    for p in sample:
        x0 = float(p.get("start_x", 50))
        y0 = float(p.get("start_y", 34))
        x1 = float(p.get("end_x", x0))
        y1 = float(p.get("end_y", y0))

        # Check normalization (if coords are 0-100 scale, adapt to pitch dimensions)
        if x0 <= 100 and x1 <= 100 and y0 <= 100 and y1 <= 100 and (x0 > 85 or y0 > 60):
            x0 = x0 * PITCH_LENGTH_M / 100.0
            x1 = x1 * PITCH_LENGTH_M / 100.0
            y0 = y0 * PITCH_WIDTH_M / 100.0
            y1 = y1 * PITCH_WIDTH_M / 100.0

        ev_id = p.get("event_id", "")
        is_highlighted = ev_id in highlight_ids
        is_complete = p.get("completed", True) is not False
        p_team = p.get("team_id", "home")
        prog = is_progressive_pass(x0, y0, x1, y1, team_id=p_team)

        if is_highlighted:
            arrow_color = "#f39c12"  # Bright amber glow
            arrow_width = 3.5
            opacity = 1.0
        elif prog and is_complete:
            arrow_color = COLOR_PROGRESSIVE
            arrow_width = 2.8
            opacity = 0.95
        elif is_complete:
            arrow_color = COLOR_HOME if p_team == "home" else COLOR_AWAY
            arrow_width = 1.8
            opacity = 0.75
        else:
            arrow_color = COLOR_PASS_INCOMPLETE
            arrow_width = 1.2
            opacity = 0.55

        # Add directional vector arrow
        fig.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2 if is_complete else 1,
            arrowsize=1.2,
            arrowwidth=arrow_width,
            arrowcolor=arrow_color,
            opacity=opacity,
        )

    return fig


def add_pass_trajectories(
    fig: go.Figure,
    home_passes: List[Dict[str, Any]],
    away_passes: List[Dict[str, Any]],
) -> go.Figure:
    """Add pass trajectory lines and directional arrows to the pitch figure."""
    if home_passes:
        add_directional_pass_arrows(fig, home_passes, team_id="home")
    if away_passes:
        add_directional_pass_arrows(fig, away_passes, team_id="away")
    return fig


def add_event_markers(
    fig: go.Figure,
    events: List[Dict[str, Any]],
    event_type: str = "pass",
    color: str = "#ffff00",
    size: int = 9,
) -> go.Figure:
    """Add event markers with rich hover details and clickable customdata."""
    if not events:
        return fig

    filtered = [e for e in events if e.get("event_type") == event_type]
    if not filtered:
        return fig

    x_coords = []
    y_coords = []
    custom_data = []
    hover_texts = []

    for ev in filtered:
        x = float(ev.get("start_x", ev.get("x", 50)))
        y = float(ev.get("start_y", ev.get("y", 34)))
        if x <= 100 and y <= 100 and (x > 85 or y > 60):
            x = x * PITCH_LENGTH_M / 100.0
            y = y * PITCH_WIDTH_M / 100.0

        x_coords.append(x)
        y_coords.append(y)

        ev_id = ev.get("event_id", "?")
        ts = ev.get("timestamp_seconds", 0.0)
        team = ev.get("team_id", "home")
        player = ev.get("player_name", ev.get("player_id", "Unknown"))
        outcome = ev.get("outcome", "completed" if ev.get("completed", True) else "failed")
        period = ev.get("period", 1)
        zone = ev.get("zone", "")

        custom_data.append([ev_id, ts, player, team, event_type, outcome, period, zone])
        hover_texts.append(
            f"<b>{event_type.upper()}</b> [{ev_id}]<br>"
            f"Time: {ts:.1f}s ({ts/60:.1f}m) | Period: {period}<br>"
            f"Player: {player} ({team.title()})<br>"
            f"Outcome: {outcome}<br>"
            f"Coord: ({x:.1f}m, {y:.1f}m)"
        )

    symbols = {
        "pass": "circle",
        "shot": "triangle-down",
        "carry": "square",
        "cross": "diamond",
        "pressure": "hexagon",
        "tackle": "x",
        "recovery": "star-triangle-up",
        "turnover": "bowtie",
    }
    sym = symbols.get(event_type, "circle")

    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="markers",
            marker=dict(
                size=size,
                color=color,
                line=dict(width=1.5, color="white"),
                symbol=sym,
            ),
            customdata=custom_data,
            hovertext=hover_texts,
            hoverinfo="text",
            name=f"{event_type.title()} ({len(filtered)})",
            showlegend=True,
        )
    )

    return fig


def add_zone_highlight(
    fig: go.Figure,
    zone: str = "defensive_third",
    opacity: float = 0.15,
    color: str = "yellow",
) -> go.Figure:
    """Highlight a specific pitch zone (defensive/middle/attacking third)."""
    if zone in ("defensive_third", "def"):
        x0, x1 = 0.0, THIRD_DEF_MAX
    elif zone in ("middle_third", "mid"):
        x0, x1 = THIRD_DEF_MAX, THIRD_MID_MAX
    elif zone in ("attacking_third", "att"):
        x0, x1 = THIRD_MID_MAX, PITCH_LENGTH_M
    else:
        return fig

    fig.add_shape(
        type="rect",
        x0=x0,
        y0=0,
        x1=x1,
        y1=PITCH_WIDTH_M,
        fillcolor=color,
        opacity=opacity,
        line=dict(color=color, width=1.5, dash="dot"),
        layer="below",
    )
    return fig


def create_shot_map(
    shots: List[Dict[str, Any]],
    home_team_attacking_right: bool = True,
    title: str = "Shot Map & Goal Locations",
) -> go.Figure:
    """Create a high-impact shot map visualization with expected outcome glyphs."""
    fig = create_pitch_figure(
        title=title,
        show_zones=False,
        show_channels=False,
        show_attack_direction=True,
    )

    if not shots:
        fig.add_annotation(
            x=PITCH_LENGTH_M / 2, y=PITCH_WIDTH_M / 2,
            text="No Shots Recorded in Sample",
            showarrow=False, font=dict(color="rgba(255,255,255,0.6)", size=14)
        )
        return fig

    # Group shots by outcome
    for outcome_label, marker_sym, color, base_size in [
        ("goal", "star", COLOR_SHOT_GOAL, 18),
        ("on_target", "triangle-down", COLOR_SHOT_TARGET, 14),
        ("off_target", "x", COLOR_SHOT_OFF, 10),
        ("saved", "diamond", "#3498db", 12),
        ("blocked", "square", "#e67e22", 10),
    ]:
        matching = [s for s in shots if s.get("outcome") == outcome_label]
        if not matching:
            continue

        xs, ys, texts, cdata = [], [], [], []
        for s in matching:
            x = float(s.get("start_x", s.get("x", s.get("x_start", 50))))
            y = float(s.get("start_y", s.get("y", s.get("y_start", 34))))
            if x <= 100 and y <= 100 and (x > 85 or y > 60):
                x = x * PITCH_LENGTH_M / 100.0
                y = y * PITCH_WIDTH_M / 100.0

            xs.append(x)
            ys.append(y)
            player = s.get("player_name", s.get("player_id", "Unknown"))
            team = s.get("team_id", "home")
            ts = s.get("timestamp_seconds", 0.0)
            ev_id = s.get("event_id", "")
            cdata.append([ev_id, ts, player, team, "shot", outcome_label])
            texts.append(
                f"<b>SHOT: {outcome_label.upper()}</b><br>"
                f"Player: {player} ({team.title()})<br>"
                f"Time: {ts:.1f}s<br>"
                f"Event ID: {ev_id}"
            )

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(
                    size=base_size,
                    color=color,
                    symbol=marker_sym,
                    line=dict(width=2, color="white"),
                ),
                customdata=cdata,
                hovertext=texts,
                hoverinfo="text",
                name=f"Shot: {outcome_label.replace('_', ' ').title()} ({len(matching)})",
                showlegend=True,
            )
        )

    return fig


def add_possession_arc(
    fig: go.Figure,
    events: List[Dict[str, Any]],
    team_id: str = "home",
    color: Optional[str] = None,
) -> go.Figure:
    """Connect possession events in temporal sequence to visualize build-up routing."""
    filtered = [
        e for e in events
        if e.get("team_id") == team_id and e.get("event_type") in ("pass", "carry", "cross", "shot")
    ]
    if not filtered:
        return fig

    c = color or (COLOR_HOME if team_id == "home" else COLOR_AWAY)
    x_positions = []
    y_positions = []
    labels = []

    for ev in filtered:
        x = float(ev.get("end_x", ev.get("start_x", 50)))
        y = float(ev.get("end_y", ev.get("start_y", 34)))
        if x <= 100 and y <= 100 and (x > 85 or y > 60):
            x = x * PITCH_LENGTH_M / 100.0
            y = y * PITCH_WIDTH_M / 100.0
        x_positions.append(x)
        y_positions.append(y)
        labels.append(f"{ev.get('event_type')}: {ev.get('player_name', '')}")

    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=y_positions,
            mode="lines+markers",
            line=dict(color=c, width=2.5, dash="dash"),
            marker=dict(size=6, color=c, symbol="circle"),
            hovertext=labels,
            hoverinfo="text",
            name=f"Sequence Flow — {team_id.title()}",
            showlegend=True,
        )
    )

    return fig


def render_tactical_pitch(
    events: List[Dict[str, Any]],
    team_filter: str = "All",
    player_filter: str = "All",
    event_type_filter: str = "All",
    period_filter: Union[str, int] = "All",
    highlight_event_ids: Optional[List[str]] = None,
    home_team_name: str = "Manchester City",
    away_team_name: str = "Liverpool",
    show_arrows: bool = True,
    show_carries: bool = True,
    show_defensive: bool = True,
    show_channels: bool = False,
    flip_away_coordinates: bool = False,
) -> go.Figure:
    """Comprehensive visual-first football pitch renderer.

    Implements:
    - Directional pass arrows (progressive highlighted in gold)
    - Carries (dashed with square glyphs)
    - Crosses (distinct trajectories)
    - Shots (color-coded by outcome)
    - Defensive events (pressures, tackles, recoveries, turnovers)
    - Team attacking direction indicators
    - Coordinate scaling and optional half-specific coordinate flipping
    - Interactive filters for team, player, event type, and match period
    - Clickable customdata for every event linked to underlying data
    - Never invents player positions: renders exact event coordinates only
    """
    highlight_event_ids = highlight_event_ids or []

    # 1. Base pitch figure
    fig = create_pitch_figure(
        title=f"Tactical Pitch: {home_team_name} vs {away_team_name}",
        home_team_name=home_team_name,
        away_team_name=away_team_name,
        show_zones=True,
        show_channels=show_channels,
        show_attack_direction=True,
    )

    if not events:
        fig.add_annotation(
            x=PITCH_LENGTH_M / 2, y=PITCH_WIDTH_M / 2,
            text="No Match Events Available",
            showarrow=False, font=dict(color="rgba(255,255,255,0.7)", size=15)
        )
        return fig

    # 2. Filter events
    filtered = list(events)

    # Team filter
    if team_filter.lower() in ("home", "away"):
        filtered = [e for e in filtered if e.get("team_id") == team_filter.lower()]

    # Player filter
    if player_filter != "All" and player_filter:
        filtered = [
            e for e in filtered
            if e.get("player_name") == player_filter or e.get("player_id") == player_filter
        ]

    # Period filter
    if str(period_filter).lower() != "all":
        try:
            target_period = int(period_filter)
            filtered = [e for e in filtered if e.get("period", 1) == target_period]
        except (ValueError, TypeError):
            pass

    # Event type filter
    if event_type_filter != "All" and event_type_filter:
        et_lower = event_type_filter.lower()
        if et_lower == "passes":
            filtered = [e for e in filtered if e.get("event_type") == "pass"]
        elif et_lower == "carries":
            filtered = [e for e in filtered if e.get("event_type") == "carry"]
        elif et_lower == "shots":
            filtered = [e for e in filtered if e.get("event_type") == "shot"]
        elif et_lower == "defensive":
            filtered = [e for e in filtered if e.get("event_type") in ("pressure", "tackle", "recovery", "interception")]
        elif et_lower == "turnovers":
            filtered = [e for e in filtered if e.get("turnover", False) or e.get("event_type") == "ball_loss"]
        else:
            filtered = [e for e in filtered if e.get("event_type") == et_lower]

    # 3. Coordinate standardization and optional flip for away team
    processed_events = []
    for ev in filtered:
        ev_copy = dict(ev)
        sx = float(ev.get("start_x", ev.get("x", 50)))
        sy = float(ev.get("start_y", ev.get("y", 34)))
        ex = float(ev.get("end_x", sx))
        ey = float(ev.get("end_y", sy))

        # Check if normalized 0-100
        if sx <= 100 and sy <= 100 and ex <= 100 and ey <= 100 and (sx > 85 or sy > 60):
            sx = sx * PITCH_LENGTH_M / 100.0
            ex = ex * PITCH_LENGTH_M / 100.0
            sy = sy * PITCH_WIDTH_M / 100.0
            ey = ey * PITCH_WIDTH_M / 100.0

        if flip_away_coordinates and ev.get("team_id") == "away":
            sx = PITCH_LENGTH_M - sx
            ex = PITCH_LENGTH_M - ex
            sy = PITCH_WIDTH_M - sy
            ey = PITCH_WIDTH_M - ey

        ev_copy["start_x"] = sx
        ev_copy["start_y"] = sy
        ev_copy["end_x"] = ex
        ev_copy["end_y"] = ey
        processed_events.append(ev_copy)

    # 4. Draw Directional Pass Arrows
    if show_arrows:
        pass_events = [e for e in processed_events if e.get("event_type") == "pass"]
        add_directional_pass_arrows(
            fig,
            pass_events,
            team_id=None,
            highlight_ids=highlight_event_ids,
            max_arrows=45,
        )

    # 5. Draw Carries
    if show_carries:
        carry_events = [e for e in processed_events if e.get("event_type") == "carry"]
        for c in carry_events[-30:]:
            team = c.get("team_id", "home")
            color = COLOR_CARRY
            fig.add_trace(
                go.Scatter(
                    x=[c["start_x"], c["end_x"]],
                    y=[c["start_y"], c["end_y"]],
                    mode="lines+markers",
                    line=dict(color=color, width=2.5, dash="dot"),
                    marker=dict(size=6, symbol="square", color=color),
                    hovertext=f"Carry: {c.get('player_name', '')} ({c.get('event_id', '')})",
                    hoverinfo="text",
                    showlegend=False,
                )
            )

    # 6. Draw Shots
    shot_events = [e for e in processed_events if e.get("event_type") == "shot"]
    if shot_events:
        for s in shot_events:
            outcome = s.get("outcome", "off_target")
            color = COLOR_SHOT_GOAL if outcome == "goal" else (COLOR_SHOT_TARGET if outcome == "on_target" else COLOR_SHOT_OFF)
            sym = "star" if outcome == "goal" else ("triangle-down" if outcome == "on_target" else "x")
            size = 18 if outcome == "goal" else 13
            ev_id = s.get("event_id", "")
            fig.add_trace(
                go.Scatter(
                    x=[s["start_x"]],
                    y=[s["start_y"]],
                    mode="markers",
                    marker=dict(size=size, color=color, symbol=sym, line=dict(color="white", width=2)),
                    customdata=[[ev_id, s.get("timestamp_seconds", 0), s.get("player_name", ""), s.get("team_id", ""), "shot", outcome]],
                    hovertext=f"<b>SHOT [{outcome.upper()}]</b>: {s.get('player_name', '')} ({ev_id})",
                    hoverinfo="text",
                    name=f"Goal ({s.get('player_name', '')})" if outcome == "goal" else "Shot Attempt",
                    showlegend=False,
                )
            )

    # 7. Draw Defensive Actions (Pressures, Tackles, Recoveries, Turnovers)
    if show_defensive:
        for etype, color, sym in [
            ("pressure", COLOR_PRESSURE, "circle-open"),
            ("tackle", COLOR_TACKLE, "x"),
            ("recovery", COLOR_RECOVERY, "star-triangle-up"),
        ]:
            matching = [e for e in processed_events if e.get("event_type") == etype]
            if matching:
                add_event_markers(fig, matching, event_type=etype, color=color, size=10)

        # Explicit turnovers
        turnovers = [e for e in processed_events if e.get("turnover", False) or e.get("event_type") == "ball_loss"]
        if turnovers:
            tx = [e["start_x"] for e in turnovers]
            ty = [e["start_y"] for e in turnovers]
            fig.add_trace(
                go.Scatter(
                    x=tx,
                    y=ty,
                    mode="markers",
                    marker=dict(size=12, color=COLOR_TURNOVER, symbol="bowtie", line=dict(width=1.5, color="white")),
                    name=f"Turnovers ({len(turnovers)})",
                    hovertext=[f"Turnover: {e.get('player_name', '')} ({e.get('event_id', '')})" for e in turnovers],
                    hoverinfo="text",
                    showlegend=True,
                )
            )

    # 8. Render Highlighted Events with pulsating glow
    if highlight_event_ids:
        highlighted = [e for e in processed_events if e.get("event_id") in highlight_event_ids]
        if highlighted:
            hx = [e["start_x"] for e in highlighted]
            hy = [e["start_y"] for e in highlighted]
            fig.add_trace(
                go.Scatter(
                    x=hx,
                    y=hy,
                    mode="markers",
                    marker=dict(
                        size=20,
                        color="rgba(255, 215, 0, 0.4)",
                        symbol="circle",
                        line=dict(color="#f1c40f", width=3),
                    ),
                    name="Selected Event Focus",
                    hoverinfo="skip",
                    showlegend=True,
                )
            )

    return fig


__all__ = [
    "create_pitch_figure",
    "add_pitch_players",
    "add_directional_pass_arrows",
    "add_pass_trajectories",
    "add_event_markers",
    "add_zone_highlight",
    "create_shot_map",
    "add_possession_arc",
    "render_tactical_pitch",
    "is_progressive_pass",
    "PITCH_LENGTH_M",
    "PITCH_WIDTH_M",
]
"""TACTIQ AI - Match Timeline and Event Inspection Visualization.

Creates interactive timeline of match events with filtering,
zoom, period dividers, and comprehensive event inspection panels.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# Tactical theme palette
DARK_BG = "#121820"
PANEL_BG = "#182230"
TEXT_COLOR = "#ecf0f1"

# Event type colours
EVENT_COLOURS = {
    "pass": "#3498db",
    "carry": "#2ecc71",
    "cross": "#e67e22",
    "shot": "#e74c3c",
    "pressure": "#9b59b6",
    "tackle": "#c0392b",
    "recovery": "#1abc9c",
    "interception": "#34495e",
    "clearance": "#95a5a6",
    "ball_loss": "#f39c12",
}


def create_match_timeline(
    events: List[Dict[str, Any]],
    home_team: str = "Home",
    away_team: str = "Away",
    period_breaks: Optional[List[float]] = None,
    highlight_event_ids: Optional[List[str]] = None,
    show_teams: bool = True,
    width: Optional[int] = None,
    height: int = 350,
) -> go.Figure:
    """Create an interactive match timeline with event markers.

    Args:
        events: List of match events
        home_team: Home team name
        away_team: Away team name
        period_breaks: Timestamps where period breaks occur (default: [2700.0] for 45 min)
        highlight_event_ids: Event IDs to highlight with glowing markers
        show_teams: Whether to color events by team
        width: Optional pixel width
        height: Pixel height
    """
    sorted_events = sorted(events, key=lambda e: float(e.get("timestamp_seconds", 0)))
    highlight_event_ids = highlight_event_ids or []
    breaks = period_breaks if period_breaks is not None else [2700.0]

    fig = go.Figure()

    if not sorted_events:
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="No timeline events available",
            showarrow=False, font=dict(color="white", size=14)
        )
        fig.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG)
        return fig

    # Period break markers (e.g. Halftime at 45m / 2700s)
    for pb in breaks:
        fig.add_vline(
            x=pb,
            line_dash="dash",
            line_color="rgba(255, 255, 255, 0.4)",
            line_width=2,
            annotation_text="HALF TIME (45')",
            annotation_position="top",
            annotation_font=dict(color="#f1c40f", size=10),
        )

    # Separate events by team: Home (y=+1), Neutral (y=0), Away (y=-1)
    home_evs = [e for e in sorted_events if e.get("team_id") == "home"]
    away_evs = [e for e in sorted_events if e.get("team_id") == "away"]

    for team_label, ev_list, y_pos, base_color in [
        (home_team, home_evs, 1.0, "#3498db"),
        (away_team, away_evs, -1.0, "#e74c3c"),
    ]:
        if not ev_list:
            continue

        # Non-shot events
        normal_evs = [e for e in ev_list if e.get("event_type") != "shot"]
        shot_evs = [e for e in ev_list if e.get("event_type") == "shot"]

        if normal_evs:
            times = [float(e.get("timestamp_seconds", 0)) for e in normal_evs]
            y_vals = [y_pos] * len(normal_evs)
            sizes = [14 if e.get("event_id") in highlight_event_ids else 8 for e in normal_evs]
            colors = [
                "#f39c12" if e.get("event_id") in highlight_event_ids else EVENT_COLOURS.get(e.get("event_type", ""), base_color)
                for e in normal_evs
            ]
            hover_text = [
                f"<b>{e.get('event_type', '').upper()}</b> [{e.get('event_id', '')}]<br>"
                f"Time: {float(e.get('timestamp_seconds', 0)):.1f}s ({float(e.get('timestamp_seconds', 0))/60:.1f}')<br>"
                f"Player: {e.get('player_name', e.get('player_id', 'Unknown'))}<br>"
                f"Outcome: {e.get('outcome', '')}"
                for e in normal_evs
            ]
            cdata = [[e.get("event_id"), e.get("timestamp_seconds"), e.get("player_name")] for e in normal_evs]

            fig.add_trace(
                go.Scatter(
                    x=times,
                    y=y_vals,
                    mode="markers",
                    marker=dict(
                        size=sizes,
                        color=colors,
                        symbol="circle",
                        line=dict(width=1, color="white"),
                    ),
                    customdata=cdata,
                    hovertext=hover_text,
                    hoverinfo="text",
                    name=f"{team_label} Events",
                    showlegend=True,
                )
            )

        # Shots (rendered with star/triangle flags)
        if shot_evs:
            s_times = [float(e.get("timestamp_seconds", 0)) for e in shot_evs]
            s_y = [y_pos * 1.3] * len(shot_evs)
            s_colors = ["#f1c40f" if e.get("outcome") == "goal" else base_color for e in shot_evs]
            s_symbols = ["star" if e.get("outcome") == "goal" else "triangle-down" for e in shot_evs]
            s_hover = [
                f"<b>SHOT: {e.get('outcome', '').upper()}</b><br>"
                f"Player: {e.get('player_name', e.get('player_id', ''))}<br>"
                f"Time: {float(e.get('timestamp_seconds', 0))/60:.1f}'"
                for e in shot_evs
            ]

            fig.add_trace(
                go.Scatter(
                    x=s_times,
                    y=s_y,
                    mode="markers",
                    marker=dict(
                        size=16,
                        color=s_colors,
                        symbol=s_symbols,
                        line=dict(width=2, color="white"),
                    ),
                    hovertext=s_hover,
                    hoverinfo="text",
                    name=f"{team_label} Shots",
                    showlegend=True,
                )
            )

    max_t = max([float(e.get("timestamp_seconds", 0)) for e in sorted_events], default=5400.0)

    fig.update_layout(
        title=dict(
            text=f"Match Timeline & Event Stream — {home_team} vs {away_team}",
            font=dict(color="white", size=14),
        ),
        xaxis=dict(
            title="Match Time (Seconds / Minutes)",
            range=[-60, max(max_t + 120, 5400)],
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.1)",
            zeroline=False,
        ),
        yaxis=dict(
            range=[-2.2, 2.2],
            tickvals=[-1.0, 1.0],
            ticktext=[f"⮜ {away_team}", f"{home_team} ➔"],
            showgrid=False,
            zeroline=True,
            zerolinecolor="rgba(255, 255, 255, 0.2)",
        ),
        plot_bgcolor=PANEL_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR),
        height=height,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
    )

    return fig


def create_event_inspection_panel(
    events: List[Dict[str, Any]],
    event_ids: Optional[List[str]] = None,
    width: Optional[int] = None,
    height: int = 340,
) -> go.Figure:
    """Create a tactical inspection panel showing event location and spatial vector."""
    fig = go.Figure()

    # Mini pitch silhouette
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=105, y1=68,
        line=dict(color="white", width=2),
        fillcolor="#193322",
        layer="below",
    )
    fig.add_shape(
        type="line",
        x0=52.5, y0=0, x1=52.5, y1=68,
        line=dict(color="rgba(255,255,255,0.4)", width=1.5, dash="dash"),
        layer="below",
    )
    fig.add_shape(
        type="circle",
        x0=52.5 - 9.15, y0=34 - 9.15, x1=52.5 + 9.15, y1=34 + 9.15,
        line=dict(color="rgba(255,255,255,0.4)", width=1.5),
        layer="below",
    )

    selected = []
    if event_ids:
        selected = [e for e in events if e.get("event_id") in event_ids]
    if not selected and events:
        selected = events[:1]

    if not selected:
        fig.add_annotation(
            x=52.5, y=34, text="Select an event from the dropdown to inspect",
            showarrow=False, font=dict(color="white", size=13)
        )
        fig.update_layout(
            paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
            xaxis=dict(range=[-2, 107], visible=False),
            yaxis=dict(range=[-2, 70], visible=False),
            height=height,
        )
        return fig

    for ev in selected:
        sx = float(ev.get("start_x", ev.get("x", 50)))
        sy = float(ev.get("start_y", ev.get("y", 34)))
        ex = float(ev.get("end_x", sx))
        ey = float(ev.get("end_y", sy))

        if sx <= 100 and sy <= 100 and (sx > 85 or sy > 60):
            sx = sx * 105.0 / 100.0
            ex = ex * 105.0 / 100.0
            sy = sy * 68.0 / 100.0
            ey = ey * 68.0 / 100.0

        ev_id = ev.get("event_id", "")
        etype = ev.get("event_type", "event")
        player = ev.get("player_name", ev.get("player_id", "Player"))
        team = ev.get("team_id", "home")
        color = "#3498db" if team == "home" else "#e74c3c"

        # Directional vector arrow
        fig.add_annotation(
            x=ex, y=ey, ax=sx, ay=sy,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.5,
            arrowwidth=3, arrowcolor="#f1c40f",
        )

        # Start marker
        fig.add_trace(
            go.Scatter(
                x=[sx], y=[sy],
                mode="markers+text",
                marker=dict(size=14, color=color, symbol="circle", line=dict(color="white", width=2)),
                text=[f"Start ({sx:.1f}, {sy:.1f})"],
                textposition="bottom center",
                textfont=dict(color="white", size=10),
                name="Start Coord",
            )
        )

        # End marker
        if (ex, ey) != (sx, sy):
            fig.add_trace(
                go.Scatter(
                    x=[ex], y=[ey],
                    mode="markers+text",
                    marker=dict(size=14, color="#f1c40f", symbol="star", line=dict(color="white", width=2)),
                    text=[f"End ({ex:.1f}, {ey:.1f})"],
                    textposition="top center",
                    textfont=dict(color="white", size=10),
                    name="End Coord",
                )
            )

    info_title = f"Event Inspection: {selected[0].get('event_type', '').upper()} [{selected[0].get('event_id', '')}]"

    fig.update_layout(
        title=dict(text=info_title, font=dict(color="white", size=13)),
        xaxis=dict(range=[-3, 108], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[-3, 71], showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor="#142017",
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR),
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True,
    )

    return fig


def create_insight_timeline(
    insights: List[Dict[str, Any]],
    width: Optional[int] = None,
    height: int = 240,
) -> go.Figure:
    """Create a timeline displaying detected key tactical moments and insights."""
    fig = go.Figure()

    colours = {
        "build_up": "#3498db",
        "attack": "#e74c3c",
        "defense": "#9b59b6",
        "transition": "#1abc9c",
        "player": "#e67e22",
        "model": "#f1c40f",
        "moment": "#95a5a6",
    }

    if not insights:
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="No tactical insights recorded yet",
            showarrow=False, font=dict(color="white", size=12)
        )
        fig.update_layout(paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG, height=height)
        return fig

    for ins in insights:
        ts = float(ins.get("timestamp_seconds", 0.0))
        cat = ins.get("category", "moment")
        headline = ins.get("headline", ins.get("observation", "Tactical insight"))
        ev_ids = ins.get("supporting_event_ids", [])
        c = colours.get(cat, "#3498db")

        fig.add_trace(
            go.Scatter(
                x=[ts],
                y=[0],
                mode="markers+text",
                marker=dict(size=14, color=c, symbol="diamond", line=dict(color="white", width=1.5)),
                text=[cat.title()[:6]],
                textposition="top center",
                textfont=dict(color="white", size=9),
                hovertext=f"<b>{cat.upper()}: {headline}</b><br>Time: {ts:.1f}s<br>Supporting Events: {len(ev_ids)}",
                hoverinfo="text",
                name=cat.title(),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=dict(text="Tactical Insights & Turning Points", font=dict(color="white", size=13)),
        xaxis=dict(
            title="Time (seconds)",
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
        ),
        yaxis=dict(visible=False, range=[-0.5, 0.5]),
        plot_bgcolor=PANEL_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR),
        height=height,
        margin=dict(l=30, r=30, t=40, b=30),
    )

    return fig


__all__ = [
    "create_match_timeline",
    "create_event_inspection_panel",
    "create_insight_timeline",
]
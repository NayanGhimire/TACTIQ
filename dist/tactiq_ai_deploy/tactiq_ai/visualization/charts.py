"""TACTIQ AI - Chart Visualizations.

Plotly charts for tactical intelligence:
- Passing networks and progression routes
- Build-up flow and midfield entries
- Attacking channels, crosses, and shot distributions
- Defensive behavior, pressure zones, and recovery maps
- Transitions and turnover analysis
- Player radar metrics and connection charts
- Tactical DNA and Keras model prediction probabilities
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import plotly.graph_objects as go
import plotly.express as px

logger = logging.getLogger(__name__)

# Dark tactical theme palette
DARK_BG = "#121820"
PANEL_BG = "#182230"
TEXT_COLOR = "#ecf0f1"
ACCENT_BLUE = "#3498db"
ACCENT_RED = "#e74c3c"
ACCENT_GOLD = "#f1c40f"
ACCENT_GREEN = "#2ecc71"
ACCENT_PURPLE = "#9b59b6"


def passing_network_chart(
    events: List[Dict[str, Any]],
    team_id: str = "home",
    min_passes: int = 1,
    team_name: Optional[str] = None,
) -> go.Figure:
    """Create a tactical passing network showing pass volume between pitch zones.

    Nodes represent average positions of players or zone centroids;
    edges represent volume of successful passes with proportional thickness.
    """
    fig = go.Figure()

    # Dark tactical pitch background silhouette
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=105, y1=68,
        line=dict(color="rgba(255,255,255,0.3)", width=2),
        fillcolor="#193322",
        layer="below",
    )
    # Halfway line and center circle
    fig.add_shape(
        type="line",
        x0=52.5, y0=0, x1=52.5, y1=68,
        line=dict(color="rgba(255,255,255,0.25)", width=1.5, dash="dash"),
        layer="below",
    )
    fig.add_shape(
        type="circle",
        x0=52.5 - 9.15, y0=34 - 9.15, x1=52.5 + 9.15, y1=34 + 9.15,
        line=dict(color="rgba(255,255,255,0.25)", width=1.5),
        layer="below",
    )

    passes = [e for e in events if e.get("event_type") == "pass" and e.get("team_id", "home") == team_id]

    if not passes:
        fig.add_annotation(
            x=52.5, y=34, text=f"No pass data for {team_id}",
            showarrow=False, font=dict(color="white", size=13)
        )
        return fig

    # Compute player average locations or zone links
    player_positions: Dict[str, List[Tuple[float, float]]] = {}
    for p in passes:
        player = p.get("player_name") or p.get("player_id", "Unknown")
        sx = float(p.get("start_x", 50))
        sy = float(p.get("start_y", 34))
        if sx <= 100 and sy <= 100 and (sx > 85 or sy > 60):
            sx = sx * 105.0 / 100.0
            sy = sy * 68.0 / 100.0
        player_positions.setdefault(player, []).append((sx, sy))

    # Calculate average node positions for active players
    player_nodes = {}
    for player, coords in player_positions.items():
        if len(coords) >= 1:
            avg_x = sum(c[0] for c in coords) / len(coords)
            avg_y = sum(c[1] for c in coords) / len(coords)
            player_nodes[player] = (avg_x, avg_y, len(coords))

    # Aggregated pass connections
    pair_counts: Dict[Tuple[str, str], int] = {}
    for i in range(len(passes) - 1):
        p1 = passes[i]
        p2 = passes[i + 1]
        t1 = p1.get("timestamp_seconds", 0)
        t2 = p2.get("timestamp_seconds", 0)
        if 0 <= (t2 - t1) <= 6.0:  # consecutive pass sequence
            pl1 = p1.get("player_name") or p1.get("player_id", "P1")
            pl2 = p2.get("player_name") or p2.get("player_id", "P2")
            if pl1 != pl2 and pl1 in player_nodes and pl2 in player_nodes:
                pair = tuple(sorted([pl1, pl2]))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    # Draw connection edges
    edge_color = "rgba(52, 152, 219, 0.7)" if team_id == "home" else "rgba(231, 76, 60, 0.7)"
    for (pl1, pl2), count in pair_counts.items():
        if count < min_passes:
            continue
        x0, y0, _ = player_nodes[pl1]
        x1, y1, _ = player_nodes[pl2]
        w = min(6.0, 1.2 + count * 0.8)

        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=edge_color, width=w),
                hoverinfo="text",
                hovertext=f"Connection: {pl1} ↔ {pl2} ({count} passes)",
                showlegend=False,
            )
        )

    # Draw player nodes
    node_xs = [node[0] for node in player_nodes.values()]
    node_ys = [node[1] for node in player_nodes.values()]
    node_sizes = [min(28, 12 + node[2] * 0.6) for node in player_nodes.values()]
    node_labels = list(player_nodes.keys())
    node_color = ACCENT_BLUE if team_id == "home" else ACCENT_RED

    fig.add_trace(
        go.Scatter(
            x=node_xs,
            y=node_ys,
            mode="markers+text",
            marker=dict(
                size=node_sizes,
                color=node_color,
                line=dict(width=2, color="white"),
            ),
            text=[f"{n[:8]}" for n in node_labels],
            textposition="top center",
            textfont=dict(color="white", size=9),
            hovertext=[f"<b>{n}</b><br>Involvement: {node[2]} passes" for n, node in player_nodes.items()],
            hoverinfo="text",
            name=f"{team_name or team_id.title()} Passers",
            showlegend=True,
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Passing Network & Spatial Combinations — {team_name or team_id.title()}",
            font=dict(color="white", size=14),
        ),
        xaxis=dict(range=[-2, 107], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[-2, 70], showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor="#142017",
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR),
        height=450,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


def buildup_progression_chart(
    buildup_data: Dict[str, Any],
    team_name: str = "Team",
) -> go.Figure:
    """Sankey or categorized bar chart displaying progression from Defensive to Attacking Thirds."""
    fig = go.Figure()

    starts = buildup_data.get("defensive_third_starts", 0)
    crossings = buildup_data.get("third_crossings", 0)
    passes = buildup_data.get("total_passes", 0)
    completed = buildup_data.get("completed_passes", 0)
    pressure = buildup_data.get("pressure_build_up", 0)
    turnovers = buildup_data.get("turnovers_in_build_up", 0)

    categories = [
        "Def 3rd Starts",
        "Total Passes",
        "Completed Passes",
        "Under Pressure",
        "Third Crossings",
        "Build-up Turnovers",
    ]
    values = [starts, passes, completed, pressure, crossings, turnovers]
    colors = ["#3498db", "#2980b9", "#2ecc71", "#9b59b6", "#f1c40f", "#e74c3c"]

    fig.add_trace(
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=values,
            textposition="auto",
            textfont=dict(color="white", size=12),
        )
    )

    fig.update_layout(
        title=f"Build-Up Progression Breakdown — {team_name}",
        xaxis_title="Build-Up Stage",
        yaxis_title="Count",
        plot_bgcolor=PANEL_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR),
        height=380,
        margin=dict(l=40, r=20, t=50, b=50),
    )
    return fig


def channel_distribution_chart(
    channel_data: Dict[str, int],
    title: str = "Attacking Channel Utilization",
) -> go.Figure:
    """Bar chart comparing wing vs central channel progression routes."""
    labels = ["Left Wing", "Left Halfspace", "Central", "Right Halfspace", "Right Wing"]
    raw_keys = ["left_wide", "left_halfspace", "central", "right_halfspace", "right_wide"]
    values = [channel_data.get(k, 0) for k in raw_keys]

    colors = ["#3498db", "#4ecdc4", "#f1c40f", "#4ecdc4", "#3498db"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker=dict(color=colors, line=dict(color="white", width=1)),
                text=[f"{v} ({v/max(sum(values), 1):.0%})" for v in values],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title="Pitch Channel",
        yaxis_title="Event Involvements",
        plot_bgcolor=PANEL_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR),
        height=380,
        margin=dict(l=40, r=20, t=50, b=50),
    )
    return fig


def defensive_actions_chart(
    defence_data: Dict[str, Any],
    title: str = "Defensive Actions & Pressing Distribution",
) -> go.Figure:
    """Comparative bar chart of pressures, tackles, recoveries across pitch zones."""
    zone_dist = defence_data.get("zone_distribution", {})
    pressures = defence_data.get("pressures", 0)
    tackles = defence_data.get("tackles", 0)
    tackle_win_rate = defence_data.get("tackle_success", 0.0)
    pressure_rate = defence_data.get("pressure_success", 0.0)

    fig = go.Figure()

    if zone_dist:
        zones = [z.replace("_", " ").title() for z in zone_dist.keys()]
        z_vals = list(zone_dist.values())
        fig.add_trace(
            go.Bar(
                name="Zone Activity",
                x=zones,
                y=z_vals,
                marker_color="#e74c3c",
                text=z_vals,
                textposition="auto",
            )
        )

    fig.update_layout(
        title=f"{title} (Tackle Win: {tackle_win_rate:.0%}, Press Success: {pressure_rate:.0%})",
        xaxis_title="Zone",
        yaxis_title="Defensive Actions",
        plot_bgcolor=PANEL_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR),
        height=380,
        margin=dict(l=40, r=20, t=50, b=50),
    )
    return fig


def transition_rhythm_chart(
    transition_data: Dict[str, Any],
    team_name: str = "Team",
) -> go.Figure:
    """Visualizes possession turnovers, counter-attacks, and tempo metrics."""
    metrics = [
        "Total Possessions",
        "Avg Duration (s)",
        "Counter-Presses",
        "Dangerous Turnovers",
        "Avg Time Between Events (s)",
    ]
    vals = [
        transition_data.get("possessions", 0),
        round(transition_data.get("avg_duration", 0), 1),
        transition_data.get("counter_presses", 0),
        transition_data.get("dangerous_turnovers", 0),
        round(transition_data.get("avg_time_between", 0), 1),
    ]

    colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#f1c40f"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=metrics,
                y=vals,
                marker_color=colors,
                text=vals,
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title=f"Transitions & Match Rhythm — {team_name}",
        plot_bgcolor=PANEL_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR),
        height=380,
        margin=dict(l=40, r=20, t=50, b=60),
    )
    return fig


def player_radar_chart(
    player_metrics: Dict[str, Any],
    player_name: str = "Player",
) -> go.Figure:
    """Radar/Spider chart of a player's tactical contribution across key attributes."""
    categories = [
        "Passing Vol",
        "Pass Accuracy",
        "Carries",
        "Carry Success",
        "Pressures",
        "Tackles",
        "Recoveries",
        "Shots",
    ]

    # Normalize metrics to 0-100 scale for radar visualization
    total_passes = player_metrics.get("total_passes", 0)
    pass_acc = player_metrics.get("pass_accuracy", 0.0) * 100
    carries = player_metrics.get("total_carries", 0)
    carry_acc = player_metrics.get("carry_success", 0.0) * 100
    pressures = player_metrics.get("pressing", 0)
    tackles = player_metrics.get("tackles", 0)
    recoveries = player_metrics.get("recoveries", 0)
    shots = player_metrics.get("total_shots", 0)

    # Scale values relative to typical match caps
    values = [
        min(100, total_passes * 2.0),
        min(100, pass_acc),
        min(100, carries * 5.0),
        min(100, carry_acc),
        min(100, pressures * 6.0),
        min(100, tackles * 15.0),
        min(100, recoveries * 10.0),
        min(100, shots * 20.0),
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(52, 152, 219, 0.4)",
            line=dict(color=ACCENT_BLUE, width=2.5),
            name=player_name,
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#7f8c8d"),
            angularaxis=dict(color=TEXT_COLOR),
            bgcolor=PANEL_BG,
        ),
        title=f"Tactical Profile Radar — {player_name}",
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR),
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        showlegend=False,
    )
    return fig


def tactical_dna_profile_chart(
    team_name: str,
    identity_name: str = "possession_buildup",
    possession_pct: float = 55.0,
    completion_rate: float = 0.86,
    pressing_intensity: str = "high",
    defensive_line: str = "high",
) -> go.Figure:
    """Generates a Tactical DNA radar chart for a team."""
    categories = [
        "Possession Control",
        "Pass Accuracy",
        "Pressing Intensity",
        "Directness",
        "Defensive Line Height",
        "Transition Speed",
    ]

    press_score = 85 if pressing_intensity == "high" else (60 if pressing_intensity == "medium" else 40)
    line_score = 85 if defensive_line == "high" else (60 if defensive_line == "mid" else 35)
    direct_score = 40 if "possession" in identity_name else (85 if "direct" in identity_name else 65)
    transition_score = 85 if "counter" in identity_name or "press" in identity_name else 60

    values = [
        possession_pct,
        completion_rate * 100,
        press_score,
        direct_score,
        line_score,
        transition_score,
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(46, 204, 113, 0.35)",
            line=dict(color="#2ecc71", width=2.5),
            name=f"{team_name} DNA",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#7f8c8d"),
            angularaxis=dict(color=TEXT_COLOR),
            bgcolor=PANEL_BG,
        ),
        title=f"Tactical DNA Profile — {team_name} ({identity_name.replace('_', ' ').title()})",
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR),
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
        showlegend=False,
    )
    return fig


def model_prediction_chart(
    class_probabilities: Dict[str, float],
    predicted_class: str,
    title: str = "Keras 1D-CNN+LSTM Model Inference",
) -> go.Figure:
    """Bar chart displaying softmax tactical style probability distribution from Keras model."""
    labels = list(class_probabilities.keys())
    values = list(class_probabilities.values())

    clean_labels = [l.replace("_", " ").title() for l in labels]
    colours = [ACCENT_GOLD if l == predicted_class else ACCENT_BLUE for l in labels]

    fig = go.Figure(
        data=[
            go.Bar(
                x=clean_labels,
                y=values,
                marker=dict(color=colours, line=dict(color="white", width=1)),
                text=[f"{v:.1%}" for v in values],
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title=f"{title} (Predicted: {predicted_class.replace('_', ' ').title()})",
        xaxis_title="Tactical Sequence Archetype",
        yaxis_title="Softmax Probability",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        plot_bgcolor=PANEL_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR),
        height=380,
        margin=dict(l=40, r=20, t=50, b=60),
    )
    return fig


def possession_timeline_chart(
    events: List[Dict[str, Any]],
    title: str = "Match Possession Rhythm & Momentum",
) -> go.Figure:
    """Smooth match timeline showing rolling control of possession."""
    fig = go.Figure()

    sorted_events = sorted(events, key=lambda e: float(e.get("timestamp_seconds", 0)))
    if not sorted_events:
        fig.update_layout(title=title + " (no events)", paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG)
        return fig

    times = []
    possessions = []
    current = "home"

    for ev in sorted_events:
        ts = float(ev.get("timestamp_seconds", 0))
        team = ev.get("team_id")
        if ev.get("turnover", False):
            current = "away" if team == "home" else "home"
        elif team in ("home", "away"):
            current = team

        times.append(ts)
        possessions.append(1 if current == "home" else 0)

    # Compute a 10-event rolling average for momentum
    rolling = []
    window = 10
    for i in range(len(possessions)):
        start_idx = max(0, i - window + 1)
        sub = possessions[start_idx : i + 1]
        rolling.append(sum(sub) / len(sub))

    fig.add_trace(
        go.Scatter(
            x=times,
            y=rolling,
            mode="lines",
            line=dict(color="#3498db", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(52, 152, 219, 0.25)",
            name="Home Momentum",
        )
    )

    fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,255,255,0.4)")

    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis=dict(
            range=[0, 1],
            tickvals=[0.1, 0.5, 0.9],
            ticktext=["Away Dominance", "Neutral / Balanced", "Home Dominance"],
        ),
        plot_bgcolor=PANEL_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR),
        height=300,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def player_metrics_chart(
    players_data: List[Dict[str, Any]],
    metric: str = "total_passes",
    title: Optional[str] = None,
) -> go.Figure:
    """Bar chart comparing all squad players on a selected metric."""
    if not players_data:
        fig = go.Figure()
        fig.update_layout(title=title or "Player Metrics", paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG)
        return fig

    names = [p.get("player_name", p.get("player_id", "?")) for p in players_data]
    values = []
    for p in players_data:
        data = p.get("data", p)
        val = data.get(metric, 0) if isinstance(data, dict) else 0
        values.append(val)

    fig = go.Figure(
        data=[
            go.Bar(
                x=names,
                y=values,
                marker_color="#2ecc71",
                text=values,
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title=title or f"Player Comparison — {metric.replace('_', ' ').title()}",
        xaxis_title="Player",
        yaxis_title=metric.replace("_", " ").title(),
        plot_bgcolor=PANEL_BG,
        paper_bgcolor=DARK_BG,
        font=dict(color=TEXT_COLOR),
        height=380,
        margin=dict(l=40, r=20, t=50, b=60),
    )
    return fig


__all__ = [
    "passing_network_chart",
    "buildup_progression_chart",
    "channel_distribution_chart",
    "defensive_actions_chart",
    "transition_rhythm_chart",
    "player_radar_chart",
    "tactical_dna_profile_chart",
    "model_prediction_chart",
    "possession_timeline_chart",
    "player_metrics_chart",
]
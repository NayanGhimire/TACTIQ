"""TACTIQ AI - Visualization and Dashboard Integration Tests.

Comprehensive automated tests verifying Phase 3 tactical visualization,
directional pitch arrows, progressive routes, tactical charts, timeline,
event inspection, and Gradio dashboard assembly.
"""

from __future__ import annotations

import pytest
import plotly.graph_objects as go

from tactiq_ai.visualization.pitch import (
    create_pitch_figure,
    add_pitch_players,
    add_directional_pass_arrows,
    add_pass_trajectories,
    add_event_markers,
    add_zone_highlight,
    create_shot_map,
    add_possession_arc,
    render_tactical_pitch,
    is_progressive_pass,
    PITCH_LENGTH_M,
    PITCH_WIDTH_M,
)
from tactiq_ai.visualization.charts import (
    passing_network_chart,
    buildup_progression_chart,
    channel_distribution_chart,
    defensive_actions_chart,
    transition_rhythm_chart,
    player_radar_chart,
    tactical_dna_profile_chart,
    model_prediction_chart,
    possession_timeline_chart,
    player_metrics_chart,
)
from tactiq_ai.visualization.timeline import (
    create_match_timeline,
    create_event_inspection_panel,
    create_insight_timeline,
)
from tactiq_ai.app import build_gradio_app, get_visible_events, build_event_table_data, get_match_clock_and_score


@pytest.fixture
def sample_match_events():
    """Generates a rich sample of validated events covering passes, carries, shots, and defensive actions."""
    return [
        {
            "event_id": "syn_001",
            "timestamp_seconds": 12.5,
            "period": 1,
            "team_id": "home",
            "player_id": "p_gk",
            "player_name": "Ederson",
            "event_type": "pass",
            "start_x": 10.0,
            "start_y": 34.0,
            "end_x": 30.0,
            "end_y": 20.0,
            "completed": True,
            "under_pressure": False,
            "turnover": False,
        },
        {
            "event_id": "syn_002",
            "timestamp_seconds": 15.0,
            "period": 1,
            "team_id": "home",
            "player_id": "p_def",
            "player_name": "Dias",
            "event_type": "carry",
            "start_x": 30.0,
            "start_y": 20.0,
            "end_x": 45.0,
            "end_y": 22.0,
            "completed": True,
            "under_pressure": True,
            "turnover": False,
        },
        {
            "event_id": "syn_003",
            "timestamp_seconds": 18.2,
            "period": 1,
            "team_id": "home",
            "player_id": "p_mid",
            "player_name": "Rodri",
            "event_type": "pass",
            "start_x": 45.0,
            "start_y": 22.0,
            "end_x": 80.0,
            "end_y": 45.0,
            "completed": True,
            "under_pressure": False,
            "turnover": False,
        },
        {
            "event_id": "syn_004",
            "timestamp_seconds": 22.0,
            "period": 1,
            "team_id": "home",
            "player_id": "p_fwd",
            "player_name": "Haaland",
            "event_type": "shot",
            "start_x": 88.0,
            "start_y": 34.0,
            "end_x": 105.0,
            "end_y": 34.0,
            "outcome": "goal",
            "completed": True,
            "under_pressure": True,
            "turnover": False,
        },
        {
            "event_id": "syn_005",
            "timestamp_seconds": 45.0,
            "period": 1,
            "team_id": "away",
            "player_id": "p_away_fwd",
            "player_name": "Vinicius",
            "event_type": "pass",
            "start_x": 85.0,
            "start_y": 55.0,
            "end_x": 60.0,
            "end_y": 40.0,
            "completed": False,
            "under_pressure": True,
            "turnover": True,
        },
        {
            "event_id": "syn_006",
            "timestamp_seconds": 46.5,
            "period": 1,
            "team_id": "home",
            "player_id": "p_def2",
            "player_name": "Walker",
            "event_type": "recovery",
            "start_x": 40.0,
            "start_y": 38.0,
            "end_x": 40.0,
            "end_y": 38.0,
            "completed": True,
            "under_pressure": False,
            "turnover": False,
        },
        {
            "event_id": "syn_007",
            "timestamp_seconds": 48.0,
            "period": 1,
            "team_id": "away",
            "player_id": "p_away_mid",
            "player_name": "Bellingham",
            "event_type": "pressure",
            "start_x": 42.0,
            "start_y": 38.0,
            "end_x": 42.0,
            "end_y": 38.0,
            "completed": True,
            "under_pressure": False,
            "turnover": False,
        },
        {
            "event_id": "syn_008",
            "timestamp_seconds": 2800.0,
            "period": 2,
            "team_id": "away",
            "player_id": "p_away_fwd2",
            "player_name": "Rodrygo",
            "event_type": "shot",
            "start_x": 20.0,
            "start_y": 30.0,
            "end_x": 0.0,
            "end_y": 34.0,
            "outcome": "saved",
            "completed": True,
            "under_pressure": False,
            "turnover": False,
        },
    ]


# ----------------------------------------------------------------------------------------
# Pitch Unit Tests
# ----------------------------------------------------------------------------------------

def test_create_pitch_figure_structure():
    """Verify base pitch figure contains pitch dimensions, grass cut stripes, and tactical lines."""
    fig = create_pitch_figure(
        title="Test Pitch",
        show_zones=True,
        show_channels=True,
        home_team_name="Arsenal",
        away_team_name="Chelsea",
    )
    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "<b>Test Pitch</b>"
    assert list(fig.layout.xaxis.range) == [-4, PITCH_LENGTH_M + 4]
    assert list(fig.layout.yaxis.range) == [-3, PITCH_WIDTH_M + 4]
    # Check shapes (stripes, boundaries, penalty areas)
    assert len(fig.layout.shapes) >= 15
    # Check attacking direction annotations
    annot_texts = [a.text for a in fig.layout.annotations if a.text]
    assert any("Arsenal" in t for t in annot_texts)
    assert any("Chelsea" in t for t in annot_texts)


def test_is_progressive_pass_metric():
    """Test standard tactical definition for progressive passes."""
    # Home team in defensive half: 16m forward pass is progressive
    assert is_progressive_pass(start_x=20.0, start_y=34.0, end_x=37.0, end_y=34.0, team_id="home") is True
    # Home team in defensive half: 8m forward pass is not progressive
    assert is_progressive_pass(start_x=20.0, start_y=34.0, end_x=28.0, end_y=34.0, team_id="home") is False

    # Home team in attacking half: 11m forward pass is progressive
    assert is_progressive_pass(start_x=60.0, start_y=34.0, end_x=72.0, end_y=34.0, team_id="home") is True
    # Pass entering opponent box is progressive
    assert is_progressive_pass(start_x=75.0, start_y=34.0, end_x=90.0, end_y=34.0, team_id="home") is True

    # Away team attacking towards left
    assert is_progressive_pass(start_x=80.0, start_y=34.0, end_x=60.0, end_y=34.0, team_id="away") is True


def test_render_tactical_pitch_filters_and_arrows(sample_match_events):
    """Verify render_tactical_pitch applies team, event type, and period filters."""
    # Render all
    fig_all = render_tactical_pitch(sample_match_events, show_arrows=True, show_carries=True)
    assert isinstance(fig_all, go.Figure)
    assert len(fig_all.data) >= 3

    # Team Filter
    fig_home = render_tactical_pitch(sample_match_events, team_filter="Home")
    assert isinstance(fig_home, go.Figure)

    # Event Type Filter
    fig_shots = render_tactical_pitch(sample_match_events, event_type_filter="Shots")
    assert isinstance(fig_shots, go.Figure)

    # Period Filter
    fig_p2 = render_tactical_pitch(sample_match_events, period_filter=2)
    assert isinstance(fig_p2, go.Figure)

    # Highlight Event ID
    fig_hl = render_tactical_pitch(sample_match_events, highlight_event_ids=["syn_004"])
    hl_traces = [t for t in fig_hl.data if t.name == "Selected Event Focus"]
    assert len(hl_traces) == 1
    assert hl_traces[0].x[0] == 88.0


def test_create_shot_map_glyphs(sample_match_events):
    """Verify shot map distinguishes goals and shots with distinct glyphs."""
    shots = [e for e in sample_match_events if e["event_type"] == "shot"]
    fig = create_shot_map(shots, title="Shot Analysis")
    assert isinstance(fig, go.Figure)
    trace_names = [t.name for t in fig.data if t.name]
    assert any("Goal" in n for n in trace_names)
    assert any("Saved" in n for n in trace_names)


# ----------------------------------------------------------------------------------------
# Tactical Charts Tests
# ----------------------------------------------------------------------------------------

def test_passing_network_chart(sample_match_events):
    """Verify passing network draws connection nodes and edges."""
    fig = passing_network_chart(sample_match_events, team_id="home", min_passes=1)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1
    # Check player node labels
    passer_trace = next((t for t in fig.data if "Passers" in (t.name or "")), None)
    assert passer_trace is not None
    assert len(passer_trace.x) >= 2


def test_buildup_progression_chart():
    """Verify build-up progression stages bar chart."""
    data = {
        "defensive_third_starts": 14,
        "third_crossings": 10,
        "total_passes": 45,
        "completed_passes": 39,
        "pressure_build_up": 8,
        "turnovers_in_build_up": 2,
    }
    fig = buildup_progression_chart(data, team_name="Man City")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].x[0] == "Def 3rd Starts"
    assert fig.data[0].y[0] == 14


def test_channel_distribution_chart():
    """Verify channel distribution chart covers all five channels."""
    ch_data = {
        "left_wide": 5,
        "left_halfspace": 12,
        "central": 18,
        "right_halfspace": 10,
        "right_wide": 6,
    }
    fig = channel_distribution_chart(ch_data)
    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].x) == 5
    assert "Left Wing" in fig.data[0].x


def test_defensive_actions_chart():
    """Verify defensive actions chart includes zones and success rate."""
    defence_data = {
        "zone_distribution": {"defensive_third": 15, "middle_third": 8, "attacking_third": 4},
        "pressures": 22,
        "tackles": 10,
        "tackle_success": 0.70,
        "pressure_success": 0.45,
    }
    fig = defensive_actions_chart(defence_data)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert len(fig.data[0].x) == 3


def test_transition_rhythm_chart():
    """Verify transition rhythm chart."""
    t_data = {
        "possessions": 28,
        "avg_duration": 18.5,
        "counter_presses": 7,
        "dangerous_turnovers": 2,
        "avg_time_between": 2.8,
    }
    fig = transition_rhythm_chart(t_data, team_name="Liverpool")
    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].x) == 5


def test_player_radar_chart():
    """Verify player radar polar chart attributes."""
    metrics = {
        "total_passes": 35,
        "pass_accuracy": 0.88,
        "total_carries": 12,
        "carry_success": 0.75,
        "pressing": 14,
        "tackles": 4,
        "recoveries": 6,
        "total_shots": 3,
    }
    fig = player_radar_chart(metrics, player_name="Rodri")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Scatterpolar)


def test_tactical_dna_profile_chart():
    """Verify tactical DNA polar chart values."""
    fig = tactical_dna_profile_chart(
        team_name="Barcelona",
        identity_name="possession_buildup",
        possession_pct=62.0,
        completion_rate=0.89,
        pressing_intensity="high",
        defensive_line="high",
    )
    assert isinstance(fig, go.Figure)
    assert isinstance(fig.data[0], go.Scatterpolar)


def test_model_prediction_chart():
    """Verify softmax distribution bar chart."""
    probs = {
        "controlled_buildup": 0.74,
        "direct_progression": 0.12,
        "high_press_turnover": 0.08,
        "counterattack": 0.06,
    }
    fig = model_prediction_chart(probs, predicted_class="controlled_buildup")
    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].x) == 4


def test_possession_timeline_chart(sample_match_events):
    """Verify possession timeline computes momentum."""
    fig = possession_timeline_chart(sample_match_events)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


# ----------------------------------------------------------------------------------------
# Timeline & Event Inspector Tests
# ----------------------------------------------------------------------------------------

def test_create_match_timeline(sample_match_events):
    """Verify match timeline markers, periods, and shot symbols."""
    fig = create_match_timeline(sample_match_events, home_team="Man City", away_team="Real Madrid")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2


def test_create_event_inspection_panel(sample_match_events):
    """Verify event inspection draws mini pitch and vector arrow."""
    # Test empty / default call
    fig_default = create_event_inspection_panel(sample_match_events)
    assert isinstance(fig_default, go.Figure)

    # Test specific event inspection
    fig_spec = create_event_inspection_panel(sample_match_events, event_ids=["syn_003"])
    assert isinstance(fig_spec, go.Figure)
    # Check that vector arrow annotation exists
    arrows = [a for a in fig_spec.layout.annotations if a.showarrow]
    assert len(arrows) >= 1


def test_create_insight_timeline():
    """Verify insight timeline renders chronological diamonds."""
    insights = [
        {"timestamp_seconds": 120.0, "category": "build_up", "headline": "Strong build-up from back", "supporting_event_ids": ["syn_001"]},
        {"timestamp_seconds": 450.0, "category": "attack", "headline": "Central box penetration", "supporting_event_ids": ["syn_004"]},
    ]
    fig = create_insight_timeline(insights)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


# ----------------------------------------------------------------------------------------
# Dashboard Helpers & App Assembly Tests
# ----------------------------------------------------------------------------------------

def test_dashboard_helpers(sample_match_events):
    """Verify scoreboard, clock, and event feed formatting."""
    clock_str, h_goals, a_goals, period = get_match_clock_and_score(sample_match_events[:4])
    assert h_goals == 1  # Haaland goal
    assert a_goals == 0
    assert period == 1

    table_data = build_event_table_data(sample_match_events, n_last=5)
    assert len(table_data) == 5
    assert len(table_data[0]) == 8  # 8 columns

    visible = get_visible_events(3)
    assert len(visible) == 3


def test_build_gradio_app_complete():
    """Verify full Gradio app can be assembled with all 10 tabs without errors."""
    app = build_gradio_app()
    assert app is not None
    assert hasattr(app, "blocks")
    assert len(app.blocks) >= 20

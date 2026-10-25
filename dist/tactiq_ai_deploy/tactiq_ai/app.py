"""TACTIQ AI - Main Gradio Dashboard Application.

Interactive visual-first football tactical intelligence platform:
1. Live Match (Replay Controls, Interactive Pitch, Event Feed, Evidence Link)
2. Tactical DNA (Team Profiles & Strategic Identities)
3. Build-up Analysis (Passing Networks, Midfield Entries & Progression Routes)
4. Attacking Patterns (Shot Maps & Wing vs Central Channel Progression)
5. Defensive Analysis (Pressures, Tackles, Recoveries, Turnovers by Zone)
6. Transitions & Match Control (Possession Rhythm & Counter Patterns)
7. Player Focus (Involvement, Passing Connections & Radar Profile)
8. Match Timeline (Multi-Team Event Stream & Turning Points)
9. AI Tactical Report (Keras Inference + Evidence Validation)
10. Viewer Preferences & Export (Personalization, Modes, Multilingual, JSON Export)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Support both package import and direct script execution
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import numpy as np
import plotly.graph_objects as go

try:
    from .config import (
        PROJECT_ROOT, MODEL_PATH, LABEL_MAPPING_PATH, FEATURE_CONFIG_PATH,
        LOW_CONFIDENCE_THRESHOLD, SOURCE_SYNTHETIC, APP_NAME
    )
    from .schemas import (
        ETYPES, ETYPE_TO_ID, T_MAX, MIN_EVENTS,
        TacticalEvent, rescale_coordinates_to_100x100, flip_coordinates_for_away_team,
    )
    from .data.synthetic_generator import (
        generate_single_match_scenario,
        generate_full_synthetic_match,
        tactical_identities,
    )
    from .data.event_loader import MatchProcessor, validate_event_sequence
    from .features.inference import InferenceAgent
    from .features.adapter import events_to_model_dict, extract_model_window
    from .analytics.buildup import calculate_build_up_segments
    from .analytics.attack import calculate_attack_patterns
    from .analytics.defence import calculate_defensive_behaviour
    from .analytics.transitions import calculate_transitions
    from .analytics.player_analysis import calculate_player_metrics_from_events
    from .agents.narrative_generator import NarrativeGeneratorAgent
    from .agents.evidence_validator import EvidenceValidatorAgent
    from .agents.personalization import PersonalizationAgent
    from .agents.orchestrator import TactiqOrchestrator, Insight
    from .visualization.pitch import (
        create_pitch_figure, render_tactical_pitch, create_shot_map,
        add_possession_arc, PITCH_LENGTH_M, PITCH_WIDTH_M,
    )
    from .visualization.charts import (
        passing_network_chart, buildup_progression_chart,
        channel_distribution_chart, defensive_actions_chart,
        transition_rhythm_chart, player_radar_chart,
        tactical_dna_profile_chart, model_prediction_chart,
        possession_timeline_chart, player_metrics_chart,
    )
    from .visualization.timeline import (
        create_match_timeline, create_event_inspection_panel, create_insight_timeline
    )
except ImportError:
    from config import (
        PROJECT_ROOT, MODEL_PATH, LABEL_MAPPING_PATH, FEATURE_CONFIG_PATH,
        LOW_CONFIDENCE_THRESHOLD, SOURCE_SYNTHETIC, APP_NAME
    )
    from schemas import (
        ETYPES, ETYPE_TO_ID, T_MAX, MIN_EVENTS,
        TacticalEvent, rescale_coordinates_to_100x100, flip_coordinates_for_away_team,
    )
    from data.synthetic_generator import (
        generate_single_match_scenario,
        generate_full_synthetic_match,
        tactical_identities,
    )
    from data.event_loader import MatchProcessor, validate_event_sequence
    from features.inference import InferenceAgent
    from features.adapter import events_to_model_dict, extract_model_window
    from analytics.buildup import calculate_build_up_segments
    from analytics.attack import calculate_attack_patterns
    from analytics.defence import calculate_defensive_behaviour
    from analytics.transitions import calculate_transitions
    from analytics.player_analysis import calculate_player_metrics_from_events
    from agents.narrative_generator import NarrativeGeneratorAgent
    from agents.evidence_validator import EvidenceValidatorAgent
    from agents.personalization import PersonalizationAgent
    from agents.orchestrator import TactiqOrchestrator, Insight
    from visualization.pitch import (
        create_pitch_figure, render_tactical_pitch, create_shot_map,
        add_possession_arc, PITCH_LENGTH_M, PITCH_WIDTH_M,
    )
    from visualization.charts import (
        passing_network_chart, buildup_progression_chart,
        channel_distribution_chart, defensive_actions_chart,
        transition_rhythm_chart, player_radar_chart,
        tactical_dna_profile_chart, model_prediction_chart,
        possession_timeline_chart, player_metrics_chart,
    )
    from visualization.timeline import (
        create_match_timeline, create_event_inspection_panel, create_insight_timeline
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------
# Global Application State
# ----------------------------------------------------------------------------------------

match_processor: Optional[MatchProcessor] = None
orchestrator: Optional[TactiqOrchestrator] = None
inference_agent: Optional[InferenceAgent] = None
current_scenario_data: Dict[str, Any] = {}
current_match_events: List[Dict[str, Any]] = []
current_insights: List[Insight] = []
current_step_index: int = 50


def load_model_agent() -> Optional[InferenceAgent]:
    """Load Keras inference model."""
    global inference_agent
    try:
        inference_agent = InferenceAgent()
        logger.info("Keras Inference Model successfully initialized.")
        return inference_agent
    except Exception as e:
        logger.warning(f"Keras model could not be loaded: {e}")
        return None


def init_scenario(scenario_key: str = "possession_vs_counter", seed: int = 42) -> Dict[str, Any]:
    """Generate and load a full synthetic match scenario."""
    global current_scenario_data, current_match_events, match_processor, orchestrator, current_insights, current_step_index
    current_scenario_data = generate_single_match_scenario(scenario_key, seed=seed)
    raw_events = list(current_scenario_data["events"])
    raw_events.sort(key=lambda e: float(e.get("timestamp_seconds", 0)))
    prev_ts = -1.0
    for ev in raw_events:
        ts = float(ev.get("timestamp_seconds", 0))
        if ts <= prev_ts:
            ts = prev_ts + 0.1
            ev["timestamp_seconds"] = ts
        prev_ts = ts
    current_match_events = raw_events
    current_scenario_data["events"] = current_match_events
    match_processor = MatchProcessor(current_match_events, match_id=current_scenario_data["match_id"])
    current_step_index = min(50, len(current_match_events))

    if orchestrator is None:
        orchestrator = TactiqOrchestrator()

    orch_res = orchestrator.run(current_match_events)
    current_insights = orch_res.insights
    return current_scenario_data


# Initialize default scenario on startup
init_scenario("possession_vs_counter", seed=42)
load_model_agent()


# ----------------------------------------------------------------------------------------
# Helper Functions for Gradio UI
# ----------------------------------------------------------------------------------------

def _get_insight_field(ins: Any, field_name: str, default: Any = None) -> Any:
    """Safely extract field from either an Insight dataclass or a personalized dict."""
    if isinstance(ins, dict):
        return ins.get(field_name, default)
    return getattr(ins, field_name, default)


def get_visible_events(step_idx: int) -> List[Dict[str, Any]]:
    """Return slice of match events up to current replay step index."""
    if not current_match_events:
        return []
    idx = max(1, min(step_idx, len(current_match_events)))
    return current_match_events[:idx]


def build_event_table_data(events: List[Dict[str, Any]], n_last: int = 15) -> List[List[Any]]:
    """Format recent events into rows for dataframe display."""
    subset = events[-n_last:] if len(events) > n_last else events
    rows = []
    for ev in reversed(subset):
        ts = float(ev.get("timestamp_seconds", 0))
        m = int(ts // 60)
        s = int(ts % 60)
        clock_str = f"{m:02d}:{s:02d}"
        ev_id = ev.get("event_id", "")
        team = ev.get("team_id", "home").title()
        player = ev.get("player_name", ev.get("player_id", "Unknown"))
        etype = ev.get("event_type", "").upper()
        outcome = ev.get("outcome", "OK")
        press = "YES" if ev.get("under_pressure") else "NO"
        turn = "YES" if ev.get("turnover") else "NO"
        rows.append([ev_id, clock_str, team, player, etype, outcome, press, turn])
    return rows


def get_match_clock_and_score(events: List[Dict[str, Any]]) -> Tuple[str, int, int, int]:
    """Compute live match clock and current score up to visible events."""
    if not events:
        return "00:00", 0, 0, 1
    last_ev = events[-1]
    ts = float(last_ev.get("timestamp_seconds", 0))
    m = int(ts // 60)
    s = int(ts % 60)
    period = 1 if ts <= 2700 else 2
    clock_str = f"{m:02d}:{s:02d}"

    home_goals = sum(1 for e in events if e.get("event_type") == "shot" and e.get("outcome") == "goal" and e.get("team_id") == "home")
    away_goals = sum(1 for e in events if e.get("event_type") == "shot" and e.get("outcome") == "goal" and e.get("team_id") == "away")
    return clock_str, home_goals, away_goals, period


# ----------------------------------------------------------------------------------------
# Gradio Dashboard Layout
# ----------------------------------------------------------------------------------------

def build_gradio_app() -> gr.Blocks:
    """Build the complete 10-section visual-first TACTIQ platform."""

    custom_css = """
    .tactiq-header {
        background: linear-gradient(135deg, #111e2e 0%, #1a2f4c 50%, #0d1b2a 100%);
        border: 1px solid #2d4263;
        border-radius: 10px;
        padding: 16px 24px;
        margin-bottom: 12px;
        color: white;
    }
    .tactiq-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: bold;
        letter-spacing: 0.5px;
        margin-right: 8px;
    }
    .badge-synthetic {
        background: rgba(243, 156, 18, 0.2);
        color: #f39c12;
        border: 1px solid #f39c12;
    }
    .badge-model {
        background: rgba(52, 152, 219, 0.2);
        color: #3498db;
        border: 1px solid #3498db;
    }
    .badge-engine {
        background: rgba(46, 204, 113, 0.2);
        color: #2ecc71;
        border: 1px solid #2ecc71;
    }
    .scoreboard-box {
        background: #141f2c;
        border: 1px solid #243447;
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    """

    with gr.Blocks(
        title="TACTIQ AI — Football Tactical Intelligence",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
        ),
        css=custom_css,
    ) as app:

        # Header banner with clear data provenance indicators
        gr.HTML(
            """
            <div class="tactiq-header">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h2 style="margin: 0; font-size: 24px; letter-spacing: 1px; color: #ecf0f1;">TACTIQ AI — Football Intelligence</h2>
                        <div style="font-size: 12px; color: #95a5a6; margin-top: 4px;">Visual Tactical Analysis Platform: Build-Up • Attacks • Pressing • Defending • Transitions</div>
                    </div>
                    <div>
                        <span class="tactiq-badge badge-synthetic">SYNTHETIC DATASET</span>
                        <span class="tactiq-badge badge-engine">MATCH ENGINE v2.0</span>
                        <span class="tactiq-badge badge-model">KERAS 1D-CNN+LSTM</span>
                    </div>
                </div>
            </div>
            """
        )

        # Shared Global States
        active_step_state = gr.State(value=50)
        highlighted_events_state = gr.State(value=[])
        viewer_mode_state = gr.State(value="casual_fan")

        # --------------------------------------------------------------------------------
        # TAB 1: Live Match Dashboard & Replay Controls
        # --------------------------------------------------------------------------------
        with gr.Tab("⚽ Live Match"):
            with gr.Row():
                # Replay Controls & Match Setup
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("#### 🎮 Match Replay Engine")
                        scenario_dropdown = gr.Dropdown(
                            choices=[
                                ("Man City vs Real Madrid (Possession vs Counter)", "possession_vs_counter"),
                                ("Liverpool vs Atletico (High Press vs Compact)", "press_vs_compact"),
                                ("Man United vs Chelsea (Direct vs Direct)", "direct_vs_direct"),
                                ("Arsenal vs Bayern (Wide Attack vs Central)", "wide_vs_central"),
                            ],
                            value="possession_vs_counter",
                            label="Tactical Matchup Scenario",
                        )
                        load_match_btn = gr.Button("Generate New Match", variant="primary", size="sm")

                        with gr.Row():
                            step_btn = gr.Button("Step +5 ⏩", size="sm")
                            step_one_btn = gr.Button("Step +1 ⏭️", size="sm")
                            play_all_btn = gr.Button("Full Match 🏁", size="sm")
                            reset_replay_btn = gr.Button("Reset 🔄", size="sm")

                        replay_slider = gr.Slider(
                            minimum=1,
                            maximum=len(current_match_events),
                            value=50,
                            step=1,
                            label="Replay Timeline Scrub (Event Index)",
                        )

                    # Live Scoreboard & Clock
                    with gr.Group():
                        gr.Markdown("#### ⏱️ Match Clock & Scoreboard")
                        with gr.Row():
                            clock_display = gr.Textbox(value="14:20", label="Clock", interactive=False)
                            period_display = gr.Textbox(value="1st Half", label="Period", interactive=False)
                        with gr.Row():
                            home_score_disp = gr.Number(value=0, label="Home Score", interactive=False)
                            away_score_disp = gr.Number(value=0, label="Away Score", interactive=False)

                    # Pitch Filter Controls
                    with gr.Accordion("🔍 Pitch Tactical Filters", open=True):
                        team_filter_dd = gr.Dropdown(choices=["All", "Home", "Away"], value="All", label="Filter Team")
                        event_type_filter_dd = gr.Dropdown(
                            choices=["All", "Passes", "Carries", "Shots", "Defensive", "Turnovers"],
                            value="All",
                            label="Event Category",
                        )
                        period_filter_dd = gr.Dropdown(choices=["All", "1", "2"], value="All", label="Match Period")
                        arrows_checkbox = gr.Checkbox(value=True, label="Show Directional Pass Arrows")
                        carries_checkbox = gr.Checkbox(value=True, label="Show Ball Carries")
                        defensive_checkbox = gr.Checkbox(value=True, label="Show Defensive Actions")
                        flip_away_checkbox = gr.Checkbox(value=False, label="Normalize Away Direction (Left➔Right)")

                # Center Column: Interactive Pitch Visualization
                with gr.Column(scale=2):
                    live_pitch_plot = gr.Plot(
                        value=lambda: render_tactical_pitch(
                            get_visible_events(50),
                            home_team_name=current_scenario_data.get("home_team", "Manchester City"),
                            away_team_name=current_scenario_data.get("away_team", "Real Madrid"),
                        ),
                        label="Interactive Tactical Football Pitch",
                    )

            # Live Event Feed Table & Inspector
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("#### 📋 Live Event Stream")
                    event_feed_table = gr.Dataframe(
                        headers=["ID", "Clock", "Team", "Player", "Action", "Outcome", "Pressured", "Turnover"],
                        value=lambda: build_event_table_data(get_visible_events(50)),
                        datatype=["str", "str", "str", "str", "str", "str", "str", "str"],
                        interactive=False,
                    )
                with gr.Column(scale=1):
                    gr.Markdown("#### 🔬 Event Vector Inspection")
                    event_select_dd = gr.Dropdown(
                        choices=[e.get("event_id") for e in current_match_events[:30]],
                        value=current_match_events[0].get("event_id") if current_match_events else None,
                        label="Inspect Specific Event",
                    )
                    event_inspection_plot = gr.Plot(
                        value=lambda: create_event_inspection_panel(get_visible_events(50)),
                        label="Spatial Coordinates & Trajectory",
                    )

        # --------------------------------------------------------------------------------
        # TAB 2: Tactical DNA & Team Identities
        # --------------------------------------------------------------------------------
        with gr.Tab("🧬 Tactical DNA"):
            gr.Markdown("### Team Strategic Profiles & System Comparison")
            with gr.Row():
                with gr.Column(scale=1):
                    home_dna_plot = gr.Plot(
                        value=lambda: tactical_dna_profile_chart(
                            current_scenario_data.get("home_team", "Home"),
                            current_scenario_data.get("home_identity", "possession_buildup"),
                            possession_pct=58.0,
                            completion_rate=0.88,
                            pressing_intensity="high",
                            defensive_line="high",
                        ),
                        label="Home Team Tactical DNA",
                    )
                with gr.Column(scale=1):
                    away_dna_plot = gr.Plot(
                        value=lambda: tactical_dna_profile_chart(
                            current_scenario_data.get("away_team", "Away"),
                            current_scenario_data.get("away_identity", "counterattacking"),
                            possession_pct=42.0,
                            completion_rate=0.79,
                            pressing_intensity="medium",
                            defensive_line="low",
                        ),
                        label="Away Team Tactical DNA",
                    )
            with gr.Accordion("Tactical Archetype Descriptions", open=True):
                dna_markdown_info = gr.Markdown(
                    value=lambda: f"""
                    **{current_scenario_data.get('home_team', 'Home Team')}**: {current_scenario_data.get('home_identity', 'possession_buildup').replace('_', ' ').title()}
                    - Controlled build-up from defensive third with patient tempo and high progression volume.
                    
                    **{current_scenario_data.get('away_team', 'Away Team')}**: {current_scenario_data.get('away_identity', 'counterattacking').replace('_', ' ').title()}
                    - Compact mid/low defensive block aiming for explosive vertical counter-attacks upon turnovers.
                    """
                )

        # --------------------------------------------------------------------------------
        # TAB 3: Build-up Analysis
        # --------------------------------------------------------------------------------
        with gr.Tab("🧱 Build-Up Analysis"):
            gr.Markdown("### Progression Routes, Passing Combinations & Midfield Entries")
            with gr.Row():
                with gr.Column(scale=1):
                    home_pass_net_plot = gr.Plot(
                        value=lambda: passing_network_chart(
                            current_match_events,
                            team_id="home",
                            team_name=current_scenario_data.get("home_team", "Home"),
                        ),
                        label="Home Passing Network",
                    )
                with gr.Column(scale=1):
                    away_pass_net_plot = gr.Plot(
                        value=lambda: passing_network_chart(
                            current_match_events,
                            team_id="away",
                            team_name=current_scenario_data.get("away_team", "Away"),
                        ),
                        label="Away Passing Network",
                    )
            with gr.Row():
                with gr.Column(scale=1):
                    home_buildup_chart = gr.Plot(
                        value=lambda: buildup_progression_chart(
                            calculate_build_up_segments(current_match_events, "home"),
                            team_name=current_scenario_data.get("home_team", "Home"),
                        ),
                        label="Home Build-Up Stages",
                    )
                with gr.Column(scale=1):
                    away_buildup_chart = gr.Plot(
                        value=lambda: buildup_progression_chart(
                            calculate_build_up_segments(current_match_events, "away"),
                            team_name=current_scenario_data.get("away_team", "Away"),
                        ),
                        label="Away Build-Up Stages",
                    )

        # --------------------------------------------------------------------------------
        # TAB 4: Attacking Patterns & Chance Creation
        # --------------------------------------------------------------------------------
        with gr.Tab("⚡ Attacking Patterns"):
            gr.Markdown("### Wing Progression, Central Penetration & Shot Locations")
            with gr.Row():
                with gr.Column(scale=1):
                    home_shot_map_plot = gr.Plot(
                        value=lambda: create_shot_map(
                            [e for e in current_match_events if e.get("event_type") == "shot" and e.get("team_id") == "home"],
                            title=f"Shot Map — {current_scenario_data.get('home_team', 'Home')}",
                        ),
                        label="Home Shots & Goals",
                    )
                with gr.Column(scale=1):
                    away_shot_map_plot = gr.Plot(
                        value=lambda: create_shot_map(
                            [e for e in current_match_events if e.get("event_type") == "shot" and e.get("team_id") == "away"],
                            title=f"Shot Map — {current_scenario_data.get('away_team', 'Away')}",
                        ),
                        label="Away Shots & Goals",
                    )
            with gr.Row():
                with gr.Column(scale=1):
                    home_channel_plot = gr.Plot(
                        value=lambda: channel_distribution_chart(
                            calculate_attack_patterns(current_match_events, "home").get("shots_by_channel", {}),
                            title=f"Attacking Channel Usage — {current_scenario_data.get('home_team', 'Home')}",
                        ),
                        label="Home Channel Breakdown",
                    )
                with gr.Column(scale=1):
                    away_channel_plot = gr.Plot(
                        value=lambda: channel_distribution_chart(
                            calculate_attack_patterns(current_match_events, "away").get("shots_by_channel", {}),
                            title=f"Attacking Channel Usage — {current_scenario_data.get('away_team', 'Away')}",
                        ),
                        label="Away Channel Breakdown",
                    )

        # --------------------------------------------------------------------------------
        # TAB 5: Defensive Analysis & Pressing
        # --------------------------------------------------------------------------------
        with gr.Tab("🛡️ Defensive Analysis"):
            gr.Markdown("### Pressing Intensity, Tackles, Recoveries & Pitch Zones")
            with gr.Row():
                with gr.Column(scale=1):
                    home_defence_plot = gr.Plot(
                        value=lambda: defensive_actions_chart(
                            calculate_defensive_behaviour(current_match_events, "home"),
                            title=f"Defensive Actions by Zone — {current_scenario_data.get('home_team', 'Home')}",
                        ),
                        label="Home Defensive Footprint",
                    )
                with gr.Column(scale=1):
                    away_defence_plot = gr.Plot(
                        value=lambda: defensive_actions_chart(
                            calculate_defensive_behaviour(current_match_events, "away"),
                            title=f"Defensive Actions by Zone — {current_scenario_data.get('away_team', 'Away')}",
                        ),
                        label="Away Defensive Footprint",
                    )

        # --------------------------------------------------------------------------------
        # TAB 6: Transitions & Match Control
        # --------------------------------------------------------------------------------
        with gr.Tab("🔄 Transitions & Momentum"):
            gr.Markdown("### Possession Rhythm, Turnover Fallout & Counter-Attacks")
            with gr.Row():
                momentum_timeline_plot = gr.Plot(
                    value=lambda: possession_timeline_chart(current_match_events),
                    label="Match Momentum & Possession Rhythm",
                )
            with gr.Row():
                with gr.Column(scale=1):
                    home_transition_plot = gr.Plot(
                        value=lambda: transition_rhythm_chart(
                            {
                                "possessions": calculate_transitions(current_match_events, "home").get("total_possessions", 0),
                                "avg_duration": calculate_transitions(current_match_events, "home").get("avg_possession_duration_s", 0),
                                "counter_presses": calculate_transitions(current_match_events, "home").get("counter_presses", 0),
                                "dangerous_turnovers": calculate_transitions(current_match_events, "home").get("dangerous_turnovers", 0),
                                "avg_time_between": calculate_transitions(current_match_events, "home").get("avg_time_between_events_s", 0),
                            },
                            team_name=current_scenario_data.get("home_team", "Home"),
                        ),
                        label="Home Transition Profile",
                    )
                with gr.Column(scale=1):
                    away_transition_plot = gr.Plot(
                        value=lambda: transition_rhythm_chart(
                            {
                                "possessions": calculate_transitions(current_match_events, "away").get("total_possessions", 0),
                                "avg_duration": calculate_transitions(current_match_events, "away").get("avg_possession_duration_s", 0),
                                "counter_presses": calculate_transitions(current_match_events, "away").get("counter_presses", 0),
                                "dangerous_turnovers": calculate_transitions(current_match_events, "away").get("dangerous_turnovers", 0),
                                "avg_time_between": calculate_transitions(current_match_events, "away").get("avg_time_between_events_s", 0),
                            },
                            team_name=current_scenario_data.get("away_team", "Away"),
                        ),
                        label="Away Transition Profile",
                    )

        # --------------------------------------------------------------------------------
        # TAB 7: Player Focus
        # --------------------------------------------------------------------------------
        with gr.Tab("👤 Player Focus"):
            gr.Markdown("### Individual Player Involvements & Tactical Contributions")
            player_names = sorted(list({e.get("player_name") for e in current_match_events if e.get("player_name")}))

            with gr.Row():
                with gr.Column(scale=1):
                    player_selector = gr.Dropdown(
                        choices=player_names,
                        value=player_names[0] if player_names else None,
                        label="Select Squad Player",
                    )
                    player_radar_plot = gr.Plot(label="Player Tactical Attributes Radar")
                with gr.Column(scale=1):
                    player_comparison_plot = gr.Plot(label="Squad Passing Comparison")

            with gr.Accordion("Player Tactical Contribution Breakdown", open=True):
                player_summary_text = gr.Markdown(label="Player Performance Card")

        # --------------------------------------------------------------------------------
        # TAB 8: Match Timeline & Insights
        # --------------------------------------------------------------------------------
        with gr.Tab("📈 Match Timeline"):
            gr.Markdown("### Full Match Timeline & Tactical Turning Points")
            with gr.Row():
                full_timeline_plot = gr.Plot(
                    value=lambda: create_match_timeline(
                        current_match_events,
                        home_team=current_scenario_data.get("home_team", "Home"),
                        away_team=current_scenario_data.get("away_team", "Away"),
                    ),
                    label="Complete Match Event Chronology",
                )
            with gr.Row():
                insights_timeline_plot = gr.Plot(
                    value=lambda: create_insight_timeline([
                        {
                            "timestamp_seconds": _get_insight_field(i, "timestamp_seconds", 0),
                            "category": _get_insight_field(i, "category", "moment"),
                            "headline": _get_insight_field(i, "headline", ""),
                            "supporting_event_ids": _get_insight_field(i, "supporting_event_ids", []),
                        }
                        for i in current_insights
                    ]),
                    label="Chronological Key Insights & Tactical Turning Points",
                )

        # --------------------------------------------------------------------------------
        # TAB 9: AI Tactical Report & Evidence Validation
        # --------------------------------------------------------------------------------
        with gr.Tab("🧠 AI Tactical Report & Evidence"):
            gr.Markdown("### Keras Deep Learning Inference + Deterministic Evidence Validation")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 🤖 Model Inference (1D-CNN+LSTM)")
                    inference_team_dd = gr.Dropdown(choices=["home", "away"], value="home", label="Classify Team Window")
                    classify_btn = gr.Button("Run Keras Classifier", variant="primary", size="sm")

                    model_pred_class = gr.Textbox(label="Predicted Tactical Archetype", interactive=False)
                    model_confidence_box = gr.Number(label="Softmax Confidence", interactive=False)
                    model_flags_box = gr.Textbox(label="Model Uncertainty Flags", interactive=False)

                with gr.Column(scale=1):
                    model_prob_plot = gr.Plot(label="Softmax Probability Distribution")

            # Evidence Linkage Section (Step 4 Requirement)
            with gr.Group():
                gr.Markdown("#### 🔎 Evidence-Backed Tactical Findings (Click to inspect supporting events)")
                insight_dropdown = gr.Dropdown(
                    choices=[(f"[{_get_insight_field(i, 'category', '').upper()}] {_get_insight_field(i, 'headline', '')}", _get_insight_field(i, 'insight_id', '')) for i in current_insights],
                    value=_get_insight_field(current_insights[0], "insight_id") if current_insights else None,
                    label="Select Tactical Finding",
                )
                with gr.Row():
                    with gr.Column(scale=2):
                        evidence_card = gr.Markdown(label="Evidence Details")
                    with gr.Column(scale=1):
                        evidence_inspect_pitch = gr.Plot(label="Evidence Events on Pitch")

            with gr.Accordion("Full Match Tactical Narrative Report", open=True):
                language_radio = gr.Radio(choices=["en", "es", "fr"], value="en", label="Report Language")
                generate_report_btn = gr.Button("Generate AI Narrative Report", size="sm")
                full_report_textbox = gr.Textbox(lines=10, label="Generated Match Tactical Narrative")

        # --------------------------------------------------------------------------------
        # TAB 10: Preferences & Export
        # --------------------------------------------------------------------------------
        with gr.Tab("⚙️ Preferences & Export"):
            gr.Markdown("### Personalization Controls & Match Data Export")
            with gr.Row():
                with gr.Column(scale=1):
                    mode_radio = gr.Radio(
                        choices=[("Casual Fan (Plain English & High-Level)", "casual_fan"), ("Tactical Analyst (Full Metrics, Probabilities & Zone Stats)", "tactical_analyst")],
                        value="casual_fan",
                        label="Audience Persona Mode",
                    )
                    lang_pref_dd = gr.Dropdown(choices=["en", "es", "fr"], value="en", label="Interface Language")
                    save_prefs_btn = gr.Button("Apply Preferences", size="sm")
                    prefs_status_box = gr.Textbox(label="Status", value="Viewer mode active: Casual Fan", interactive=False)

                with gr.Column(scale=1):
                    gr.Markdown("#### 📦 Export Tactical Overlays")
                    export_overlays_btn = gr.Button("Generate & Export Overlays JSON", variant="secondary", size="sm")
                    export_file = gr.File(label="Download Overlays JSON")

        # --------------------------------------------------------------------------------
        # Event Callbacks and Reactive Wiring
        # --------------------------------------------------------------------------------

        def update_replay_state(step: int, team_f: str, etype_f: str, period_f: str, arrows: bool, carries: bool, defensive: bool, flip_away: bool):
            visible = get_visible_events(step)
            clock, h_score, a_score, period = get_match_clock_and_score(visible)
            pitch_fig = render_tactical_pitch(
                visible,
                team_filter=team_f,
                event_type_filter=etype_f,
                period_filter=period_f,
                show_arrows=arrows,
                show_carries=carries,
                show_defensive=defensive,
                flip_away_coordinates=flip_away,
                home_team_name=current_scenario_data.get("home_team", "Manchester City"),
                away_team_name=current_scenario_data.get("away_team", "Real Madrid"),
            )
            table_data = build_event_table_data(visible)
            period_str = f"{period}st Half" if period == 1 else "2nd Half"
            return step, pitch_fig, clock, period_str, h_score, a_score, table_data

        # Slider Scrubbing
        replay_slider.change(
            fn=update_replay_state,
            inputs=[replay_slider, team_filter_dd, event_type_filter_dd, period_filter_dd, arrows_checkbox, carries_checkbox, defensive_checkbox, flip_away_checkbox],
            outputs=[active_step_state, live_pitch_plot, clock_display, period_display, home_score_disp, away_score_disp, event_feed_table],
            show_progress=False,
        )

        # Step +5
        def on_step_plus_5(curr: int, t_f: str, et_f: str, p_f: str, arr: bool, car: bool, defn: bool, flip: bool):
            nxt = min(len(current_match_events), curr + 5)
            _, p_fig, clk, per, hs, ascore, tbl = update_replay_state(nxt, t_f, et_f, p_f, arr, car, defn, flip)
            return nxt, nxt, p_fig, clk, per, hs, ascore, tbl

        step_btn.click(
            fn=on_step_plus_5,
            inputs=[active_step_state, team_filter_dd, event_type_filter_dd, period_filter_dd, arrows_checkbox, carries_checkbox, defensive_checkbox, flip_away_checkbox],
            outputs=[active_step_state, replay_slider, live_pitch_plot, clock_display, period_display, home_score_disp, away_score_disp, event_feed_table],
            show_progress=False,
        )

        # Step +1
        def on_step_plus_1(curr: int, t_f: str, et_f: str, p_f: str, arr: bool, car: bool, defn: bool, flip: bool):
            nxt = min(len(current_match_events), curr + 1)
            _, p_fig, clk, per, hs, ascore, tbl = update_replay_state(nxt, t_f, et_f, p_f, arr, car, defn, flip)
            return nxt, nxt, p_fig, clk, per, hs, ascore, tbl

        step_one_btn.click(
            fn=on_step_plus_1,
            inputs=[active_step_state, team_filter_dd, event_type_filter_dd, period_filter_dd, arrows_checkbox, carries_checkbox, defensive_checkbox, flip_away_checkbox],
            outputs=[active_step_state, replay_slider, live_pitch_plot, clock_display, period_display, home_score_disp, away_score_disp, event_feed_table],
            show_progress=False,
        )

        # Full Match
        def on_play_all(t_f: str, et_f: str, p_f: str, arr: bool, car: bool, defn: bool, flip: bool):
            nxt = len(current_match_events)
            _, p_fig, clk, per, hs, ascore, tbl = update_replay_state(nxt, t_f, et_f, p_f, arr, car, defn, flip)
            return nxt, nxt, p_fig, clk, per, hs, ascore, tbl

        play_all_btn.click(
            fn=on_play_all,
            inputs=[team_filter_dd, event_type_filter_dd, period_filter_dd, arrows_checkbox, carries_checkbox, defensive_checkbox, flip_away_checkbox],
            outputs=[active_step_state, replay_slider, live_pitch_plot, clock_display, period_display, home_score_disp, away_score_disp, event_feed_table],
            show_progress=False,
        )

        # Reset
        def on_reset(t_f: str, et_f: str, p_f: str, arr: bool, car: bool, defn: bool, flip: bool):
            _, p_fig, clk, per, hs, ascore, tbl = update_replay_state(5, t_f, et_f, p_f, arr, car, defn, flip)
            return 5, 5, p_fig, clk, per, hs, ascore, tbl

        reset_replay_btn.click(
            fn=on_reset,
            inputs=[team_filter_dd, event_type_filter_dd, period_filter_dd, arrows_checkbox, carries_checkbox, defensive_checkbox, flip_away_checkbox],
            outputs=[active_step_state, replay_slider, live_pitch_plot, clock_display, period_display, home_score_disp, away_score_disp, event_feed_table],
            show_progress=False,
        )

        # Load New Scenario
        def on_load_scenario(scen_key: str):
            init_scenario(scen_key, seed=np.random.randint(1, 9999))
            ev_len = len(current_match_events)
            h_name = current_scenario_data.get("home_team", "Home")
            a_name = current_scenario_data.get("away_team", "Away")
            visible = get_visible_events(50)
            p_fig = render_tactical_pitch(visible, home_team_name=h_name, away_team_name=a_name)
            clk, hs, ascore, per = get_match_clock_and_score(visible)
            per_str = f"{per}st Half" if per == 1 else "2nd Half"
            tbl = build_event_table_data(visible)
            dd_choices = [(f"[{_get_insight_field(i, 'category', '').upper()}] {_get_insight_field(i, 'headline', '')}", _get_insight_field(i, 'insight_id', '')) for i in current_insights]
            p_names = sorted(list({e.get("player_name") for e in current_match_events if e.get("player_name")}))
            return (
                50, gr.update(maximum=ev_len, value=50), p_fig, clk, per_str, hs, ascore, tbl,
                gr.update(choices=dd_choices, value=dd_choices[0][1] if dd_choices else None),
                gr.update(choices=p_names, value=p_names[0] if p_names else None),
                gr.update(choices=[e.get("event_id") for e in current_match_events[:40]]),
            )

        load_match_btn.click(
            fn=on_load_scenario,
            inputs=[scenario_dropdown],
            outputs=[
                active_step_state, replay_slider, live_pitch_plot, clock_display, period_display,
                home_score_disp, away_score_disp, event_feed_table, insight_dropdown, player_selector,
                event_select_dd
            ],
        )

        # Event Selector Inspection
        def on_inspect_event(ev_id: str):
            if not ev_id:
                return create_event_inspection_panel(current_match_events[:1])
            return create_event_inspection_panel(current_match_events, event_ids=[ev_id])

        event_select_dd.change(
            fn=on_inspect_event,
            inputs=[event_select_dd],
            outputs=[event_inspection_plot],
        )

        # Pitch Filters change
        for control in [team_filter_dd, event_type_filter_dd, period_filter_dd, arrows_checkbox, carries_checkbox, defensive_checkbox, flip_away_checkbox]:
            control.change(
                fn=update_replay_state,
                inputs=[active_step_state, team_filter_dd, event_type_filter_dd, period_filter_dd, arrows_checkbox, carries_checkbox, defensive_checkbox, flip_away_checkbox],
                outputs=[active_step_state, live_pitch_plot, clock_display, period_display, home_score_disp, away_score_disp, event_feed_table],
                show_progress=False,
            )

        # Player Focus Callback
        def on_select_player(player_name: str):
            if not player_name or not current_match_events:
                return go.Figure(), go.Figure(), "No player selected."

            pl_evs = [e for e in current_match_events if e.get("player_name") == player_name]
            p_id = pl_evs[0].get("player_id", "") if pl_evs else ""
            t_id = pl_evs[0].get("team_id", "home") if pl_evs else "home"
            pos = pl_evs[0].get("position", "MID") if pl_evs else "MID"

            metrics = calculate_player_metrics_from_events(current_match_events, p_id, player_name, t_id, pos)
            m_dict = {
                "total_passes": metrics.total_passes,
                "pass_accuracy": metrics.pass_completion_rate,
                "total_carries": metrics.total_carries,
                "carry_success": metrics.carry_success_rate,
                "pressing": metrics.pressures_applied,
                "tackles": metrics.tackles,
                "recoveries": metrics.recoveries,
                "total_shots": metrics.total_shots,
            }
            radar_fig = player_radar_chart(m_dict, player_name=player_name)

            # Compare with squad
            squad_names = list({e.get("player_name") for e in current_match_events if e.get("team_id") == t_id and e.get("player_name")})[:8]
            squad_data = []
            for sn in squad_names:
                cnt = sum(1 for e in current_match_events if e.get("player_name") == sn and e.get("event_type") == "pass")
                squad_data.append({"player_name": sn, "total_passes": cnt})
            squad_fig = player_metrics_chart(squad_data, metric="total_passes", title=f"{t_id.title()} Squad — Pass Comparison")

            summary_md = f"""
            ### {player_name} ({pos} • {t_id.title()})
            - **Total Match Involvements**: {len(pl_evs)} events
            - **Passing**: {metrics.completed_passes}/{metrics.total_passes} completed ({metrics.pass_completion_rate:.0%})
            - **Carries**: {metrics.successful_carries}/{metrics.total_carries} ({metrics.carry_success_rate:.0%})
            - **Defending**: {metrics.pressures_applied} pressures, {metrics.tackles_won}/{metrics.tackles} tackles won, {metrics.recoveries} recoveries
            - **Attacking**: {metrics.total_shots} shots ({metrics.shots_on_target} on target, {metrics.goals} goals)
            """
            return radar_fig, squad_fig, summary_md

        player_selector.change(
            fn=on_select_player,
            inputs=[player_selector],
            outputs=[player_radar_plot, player_comparison_plot, player_summary_text],
        )

        # Model Inference Classifier Callback
        def on_run_classifier(team: str):
            agent = inference_agent or load_model_agent()
            if not agent:
                return "Model Unavailable", 0.0, "Model file not found", go.Figure()

            m_events = events_to_model_dict(current_match_events, team)
            window = extract_model_window(m_events)
            if not window:
                return "uncertain", 0.0, "Insufficient events (<14)", go.Figure()

            res = agent.classify_sequence(window)
            p_class = res.get("predicted_class", "unknown")
            conf = res.get("model_confidence", 0.0)
            flags = ", ".join(res.get("flags", [])) or "None (Clean Confidence)"
            prob_dict = res.get("probabilities", {p_class: conf})
            chart_fig = model_prediction_chart(prob_dict, p_class)
            return p_class.replace("_", " ").title(), conf, flags, chart_fig

        classify_btn.click(
            fn=on_run_classifier,
            inputs=[inference_team_dd],
            outputs=[model_pred_class, model_confidence_box, model_flags_box, model_prob_plot],
        )

        # Evidence-Backed Finding Selection (Step 4 Connection)
        def on_select_insight(ins_id: str):
            if not ins_id or not current_insights:
                return "Select an insight to inspect evidence.", go.Figure()

            selected_ins = next((i for i in current_insights if _get_insight_field(i, "insight_id") == ins_id), current_insights[0])
            ev_ids = _get_insight_field(selected_ins, "supporting_event_ids") or []
            source_type = _get_insight_field(selected_ins, "source_type", "synthetic")
            source_badge = "🧪 CALCULATED FACT (Deterministic Data)" if source_type == "deterministic" else ("🤖 MODEL PREDICTION (Softmax Inference)" if source_type == "model" else "📋 TACTICAL INTERPRETATION")
            headline = _get_insight_field(selected_ins, "headline", "Tactical Finding")
            obs = _get_insight_field(selected_ins, "observation", "")
            interp = _get_insight_field(selected_ins, "interpretation", "") or _get_insight_field(selected_ins, "significance", "")
            sample_size = _get_insight_field(selected_ins, "sample_size") or len(ev_ids)
            conf = _get_insight_field(selected_ins, "confidence")
            conf_str = f"{conf:.0%}" if conf is not None else "High (Deterministic Stat)"

            card_md = f"""
            ### 📌 {headline}
            **Provenance Classification**: `{source_badge}`
            
            - **Observation (What was observed)**: {obs}
            - **Tactical Rationale (Why it matters)**: {interp}
            - **Supporting Events**: `{', '.join(ev_ids[:8]) if ev_ids else 'None'}` (Sample size: {sample_size})
            - **Confidence / Certainty**: {conf_str}
            - **Limitations & Uncertainty**: Synthetic data simulation generated by TACTIQ Match Generator v2.0. Pattern validity is subject to simulated game distribution.
            """

            # Render pitch with exact supporting events highlighted
            pitch_highlight_fig = render_tactical_pitch(
                current_match_events,
                highlight_event_ids=ev_ids,
                show_arrows=True,
                show_carries=True,
                show_defensive=True,
                home_team_name=current_scenario_data.get("home_team", "Manchester City"),
                away_team_name=current_scenario_data.get("away_team", "Real Madrid"),
            )
            return card_md, pitch_highlight_fig

        insight_dropdown.change(
            fn=on_select_insight,
            inputs=[insight_dropdown],
            outputs=[evidence_card, evidence_inspect_pitch],
        )

        # Full Match AI Tactical Report Generation
        def on_generate_report(lang: str, mode: str):
            generator = NarrativeGeneratorAgent()
            h_name = current_scenario_data.get("home_team", "Home Team")
            a_name = current_scenario_data.get("away_team", "Away Team")
            h_buildup = calculate_build_up_segments(current_match_events, "home")
            h_attack = calculate_attack_patterns(current_match_events, "home")
            h_defence = calculate_defensive_behaviour(current_match_events, "home")

            finding = {
                "category": "build_up",
                "team": h_name,
                "data": h_buildup,
            }
            narrative = generator.generate(finding, viewer_type=mode, language=lang)

            full_text = f"=== TACTIQ AI MATCH INTELLIGENCE REPORT ===\n"
            full_text += f"Match: {h_name} vs {a_name} | Language: {lang.upper()} | Audience: {mode.replace('_', ' ').title()}\n\n"
            full_text += f"1. TACTICAL OVERVIEW\n{narrative}\n\n"
            full_text += f"2. ATTACKING CONVERSION & THREAT\n"
            full_text += f"- Final Third Entries: {h_attack.get('final_third_entries', 0)} | Box Entries: {h_attack.get('box_entries', 0)}\n"
            full_text += f"- Total Shots: {h_attack.get('total_shots', 0)} | Goals: {h_attack.get('goals', 0)} (Conversion: {h_attack.get('shot_conversion_rate', 0.0):.0%})\n\n"
            full_text += f"3. DEFENSIVE RECOVERIES & PRESSING\n"
            full_text += f"- Total Pressures: {h_defence.get('pressures', 0)} (Success: {h_defence.get('pressure_success_rate', 0.0):.0%})\n"
            full_text += f"- Tackles Won: {h_defence.get('tackles_won', 0)}/{h_defence.get('tackles', 0)} ({h_defence.get('tackle_success_rate', 0.0):.0%})\n\n"
            full_text += f"4. DATA INTEGRITY & AUDIT PROVENANCE\n"
            full_text += f"[DATASET: SYNTHETIC] All tactical signals are extracted from verified event trajectories.\n"
            full_text += f"Keras Inference: Classifications denote sequential archetypes, not match winner prophecies."
            return full_text

        generate_report_btn.click(
            fn=on_generate_report,
            inputs=[language_radio, viewer_mode_state],
            outputs=[full_report_textbox],
        )

        # Apply Preferences Callback
        def on_save_preferences(mode: str, lang: str):
            status = f"Viewer mode set to: {mode.replace('_', ' ').title()} | Language: {lang.upper()}"
            return mode, status

        save_prefs_btn.click(
            fn=on_save_preferences,
            inputs=[mode_radio, lang_pref_dd],
            outputs=[viewer_mode_state, prefs_status_box],
        )

        # Export Overlays JSON
        def on_export_overlays():
            if not orchestrator:
                return None
            orch_res = orchestrator.run(current_match_events)
            out_path = PROJECT_ROOT / "tactiq_match_overlays.json"
            with open(out_path, "w") as f:
                json.dump(orch_res.overlays, f, indent=2)
            return str(out_path)

        export_overlays_btn.click(
            fn=on_export_overlays,
            inputs=[],
            outputs=[export_file],
        )

        # Initial trigger for player tab and evidence tab on load
        app.load(
            fn=lambda: on_select_player(player_names[0] if player_names else ""),
            inputs=[],
            outputs=[player_radar_plot, player_comparison_plot, player_summary_text],
        )
        app.load(
            fn=lambda: on_select_insight(_get_insight_field(current_insights[0], "insight_id") if current_insights else ""),
            inputs=[],
            outputs=[evidence_card, evidence_inspect_pitch],
        )

    return app


def launch_app():
    """Launch the TACTIQ AI platform."""
    app = build_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )


if __name__ == "__main__":
    launch_app()
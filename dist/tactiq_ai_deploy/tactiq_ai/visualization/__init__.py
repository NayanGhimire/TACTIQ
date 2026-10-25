"""TACTIQ AI - Visualization Subpackage."""

from . import pitch, charts, timeline

from .pitch import (
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
)

from .charts import (
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

from .timeline import (
    create_match_timeline,
    create_event_inspection_panel,
    create_insight_timeline,
)

__all__ = [
    "pitch",
    "charts",
    "timeline",
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
    "create_match_timeline",
    "create_event_inspection_panel",
    "create_insight_timeline",
]
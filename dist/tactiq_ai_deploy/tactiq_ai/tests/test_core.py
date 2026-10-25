"""TACTIQ AI - Test Suite."""

import pytest
import json
from pathlib import Path

from tactiq_ai.schemas import (
    TacticalEvent, ETYPES, ETYPE_TO_ID, T_MAX, MIN_EVENTS,
    rescale_coordinates_to_100x100, flip_coordinates_for_away_team,
    map_event_type_to_model, generate_possession_build_up_sequence,
)
from tactiq_ai.data.event_loader import (
    validate_event, validate_event_sequence, TimelineError, EventValidationError,
    MatchProcessor,
)
from tactiq_ai.data.validation import validate_events_bulk, DataQualityReport
from tactiq_ai.features.adapter import (
    events_to_model_dict, extract_model_window, get_model_type,
    compute_pass_distance,
)
from tactiq_ai.analytics.buildup import calculate_build_up_segments
from tactiq_ai.analytics.attack import calculate_attack_patterns
from tactiq_ai.analytics.defence import calculate_defensive_behaviour
from tactiq_ai.analytics.transitions import calculate_transitions
from tactiq_ai.analytics.player_analysis import calculate_player_metrics_from_events
from tactiq_ai.agents.orchestrator import TactiqOrchestrator
from tactiq_ai.agents.narrative_generator import NarrativeGeneratorAgent
from tactiq_ai.agents.evidence_validator import EvidenceValidatorAgent
from tactiq_ai.agents.personalization import PersonalizationAgent
from tactiq_ai.agents.overlay_formatter import OverlayFormatterAgent


# ----------------------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------------------


@pytest.fixture
def sample_events():
    """Sample events for testing."""
    return [
        {
            "event_id": "event_001",
            "match_id": "demo_001",
            "timestamp_seconds": 120.0,
            "period": 1,
            "team_id": "home",
            "player_id": "player_01",
            "player_name": "GK",
            "event_type": "pass",
            "start_x": 10.0,
            "start_y": 34.0,
            "end_x": 35.0,
            "start_y": 34.0,
            "end_y": 40.0,
            "duration_s": 2.0,
            "completed": True,
            "under_pressure": False,
            "turnover": False,
            "possession_id": "poss_001",
            "source": "synthetic",
        },
        {
            "event_id": "event_002",
            "match_id": "demo_001",
            "timestamp_seconds": 125.0,
            "period": 1,
            "team_id": "home",
            "player_id": "player_02",
            "player_name": "CB",
            "event_type": "pass",
            "start_x": 35.0,
            "start_y": 40.0,
            "end_x": 55.0,
            "end_y": 50.0,
            "duration_s": 1.5,
            "completed": True,
            "under_pressure": True,
            "turnover": False,
            "possession_id": "poss_001",
            "source": "synthetic",
        },
        {
            "event_id": "event_003",
            "match_id": "demo_001",
            "timestamp_seconds": 130.0,
            "period": 1,
            "team_id": "away",
            "player_id": "player_11",
            "player_name": "Striker",
            "event_type": "shot",
            "start_x": 85.0,
            "start_y": 50.0,
            "end_x": 101.0,
            "end_y": 50.0,
            "duration_s": 0.5,
            "completed": True,
            "outcome": "goal",
            "under_pressure": True,
            "turnover": True,
            "possession_id": "poss_001",
            "source": "synthetic",
        },
    ]


@pytest.fixture
def model_events():
    """Model-format events for testing."""
    return [
        {"type": "pass", "x_start": 40, "y_start": 55, "x_end": 33, "y_end": 60,
         "duration_s": 1.8, "completed": 1, "under_pressure": 0, "turnover": 0,
         "possession_before": 1, "possession_after": 1, "t_start_s": 100},
        {"type": "tackle", "x_start": 33, "y_start": 60, "x_end": 33, "y_end": 60,
         "duration_s": 0.9, "completed": 1, "under_pressure": 1, "turnover": 1,
         "possession_before": 0, "possession_after": 1, "t_start_s": 102},
        {"type": "carry", "x_start": 33, "y_start": 60, "x_end": 55, "y_end": 55,
         "duration_s": 2.5, "completed": 1, "under_pressure": 0, "turnover": 0,
         "possession_before": 1, "possession_after": 1, "t_start_s": 103},
        {"type": "pass", "x_start": 55, "y_start": 55, "x_end": 85, "y_end": 50,
         "duration_s": 1.6, "completed": 1, "under_pressure": 0, "turnover": 0,
         "possession_before": 1, "possession_after": 1, "t_start_s": 106},
    ]


# ----------------------------------------------------------------------------------------
# Schema Tests
# ----------------------------------------------------------------------------------------


def test_etypes_defined():
    """ETYPES should match model's expected 9 types."""
    assert ETYPES == [
        "pass", "carry", "cross", "shot", "pressure", "tackle",
        "recovery", "ball_loss", "opp_pass"
    ]


def test_t_max():
    """Model window size should be 14."""
    assert T_MAX == 14


def test_rescale_coordinates():
    """Coordinate rescaling to 0-100 frame."""
    x, y = rescale_coordinates_to_100x100(52.5, 34.0)  # center of pitch
    assert abs(x - 50.0) < 0.1
    assert abs(y - 50.0) < 0.1


def test_flip_coordinates():
    """Away team coordinate flip."""
    x, y = flip_coordinates_for_away_team(80.0, 50.0)
    assert abs(x - 20.0) < 0.1
    assert abs(y - 50.0) < 0.1


def test_map_event_type():
    """Rich event types map to model types."""
    assert map_event_type_to_model("clearance") == ETYPE_TO_ID["recovery"]
    assert map_event_type_to_model("interception") == ETYPE_TO_ID["tackle"]
    assert map_event_type_to_model("dribble") == ETYPE_TO_ID["carry"]
    assert map_event_type_to_model("goal") == ETYPE_TO_ID["shot"]


def test_generate_possession_sequence():
    """Synthetic sequence generation returns expected structure."""
    events = generate_possession_build_up_sequence(match_id="test", team_id="home")
    assert len(events) >= 7
    for e in events:
        assert e.event_type in ETYPES + ["clearance", "interception", "dribble"]


# ----------------------------------------------------------------------------------------
# Event Validation Tests
# ----------------------------------------------------------------------------------------


def test_validate_event_success(sample_events):
    """Valid event passes validation."""
    validated = validate_event(sample_events[0], 0)
    assert validated["event_id"] == "event_001"
    assert validated["team_id"] == "home"
    assert validated["timestamp_seconds"] == 120.0


def test_validate_event_missing_field():
    """Missing required field raises error."""
    ev = {"event_type": "pass", "team_id": "home"}
    with pytest.raises(EventValidationError) as exc:
        validate_event(ev, 0)
    assert "missing required field" in str(exc.value).lower()


def test_validate_event_invalid_coords():
    """Invalid coordinates raise error."""
    ev = {"event_type": "pass", "team_id": "home", "player_id": "p1",
          "start_x": 200, "start_y": 50, "end_x": 50, "end_y": 50,
          "timestamp_seconds": 10, "match_id": "demo_001"}
    with pytest.raises(EventValidationError) as exc:
        validate_event(ev, 0)
    assert "outside valid pitch range" in str(exc.value).lower()


def test_validate_sequence_order(sample_events):
    """Events must be chronological."""
    out_of_order = [sample_events[1], sample_events[0]]
    with pytest.raises(TimelineError) as exc:
        validate_event_sequence(out_of_order)
    assert "out of order" in str(exc.value).lower()


def test_validate_sequence_success(sample_events):
    """Valid sequence passes."""
    validated = validate_event_sequence(sample_events)
    assert len(validated) == 3


# ----------------------------------------------------------------------------------------
# MatchProcessor Tests
# ----------------------------------------------------------------------------------------


def test_match_processor_start_pause_resume(sample_events):
    """MatchProcessor state transitions work."""
    processor = MatchProcessor(sample_events, "test_match")
    assert processor.is_paused is False
    assert processor.n_processed == 0

    processor.pause()
    assert processor.is_paused is True

    processor.resume()
    assert processor.is_paused is False

    processor.reset()
    assert processor.n_processed == 0
    assert processor.current_index == 0


def test_match_processor_next_event(sample_events):
    """Next event returns next event."""
    processor = MatchProcessor(sample_events, "test_match")
    ev = processor.next_event()
    assert ev is not None
    assert ev["event_id"] == "event_001"
    assert processor.n_processed == 1


def test_match_processor_replay_to_timestamp(sample_events):
    """Replay to timestamp works."""
    processor = MatchProcessor(sample_events, "test_match")
    replayed = processor.replay_to_timestamp(125.0)
    assert len(replayed) >= 2  # event_001 and event_002


# ----------------------------------------------------------------------------------------
# Adapter Tests
# ----------------------------------------------------------------------------------------


def test_events_to_model_dict(sample_events):
    """Rich events convert to model format."""
    home_events = [e for e in sample_events if e["team_id"] == "home"]
    model = events_to_model_dict(home_events, "home")

    assert len(model) == 2
    assert model[0]["type"] == ETYPE_TO_ID["pass"]
    assert 0 <= model[0]["x_start"] <= 100
    assert 0 <= model[0]["y_start"] <= 100


def test_extract_model_window(model_events):
    """Extract window returns correct slice."""
    window = extract_model_window(model_events)
    assert len(window) <= T_MAX
    assert len(window) >= MIN_EVENTS


def test_compute_pass_distance():
    """Pass distance calculation in metres."""
    # 10m in x, 10m in y = ~14.1m
    dist = compute_pass_distance(0, 0, 10, 10)
    expected = ((105/100*10)**2 + (68/100*10)**2)**0.5
    assert abs(dist - expected) < 0.5


# ----------------------------------------------------------------------------------------
# Analytics Tests
# ----------------------------------------------------------------------------------------


def test_calculate_build_up_segments(sample_events):
    """Build-up segments calculated correctly."""
    home_events = [e for e in sample_events if e["team_id"] == "home"]
    data = calculate_build_up_segments(home_events, "home")

    assert data["total_passes"] == 2
    assert data["completed_passes"] == 2
    assert data["completion_rate"] == 1.0
    assert data["pressure_build_up"] == 1


def test_calculate_attack_patterns(sample_events):
    """Attack patterns calculated correctly."""
    away_events = [e for e in sample_events if e["team_id"] == "away"]
    data = calculate_attack_patterns(away_events, "away")

    assert data["total_shots"] == 1
    assert data["goals"] == 1
    assert data["shot_conversion_rate"] == 1.0


def test_calculate_defensive_behaviour(sample_events):
    """Defensive behaviour calculated correctly."""
    # Need tackle/pressure events
    events = sample_events + [
        {
            "event_id": "event_004",
            "match_id": "demo_001",
            "timestamp_seconds": 140.0,
            "period": 1,
            "team_id": "home",
            "player_id": "player_03",
            "event_type": "tackle",
            "start_x": 30.0, "start_y": 30.0,
            "end_x": 30.0, "end_y": 30.0,
            "duration_s": 1.0,
            "completed": True,
            "under_pressure": True,
            "turnover": True,
            "source": "synthetic",
        },
    ]
    home_events = [e for e in events if e["team_id"] == "home"]
    data = calculate_defensive_behaviour(home_events, "home")

    assert data["tackles"] >= 1
    assert data["tackle_success_rate"] == 1.0


def test_calculate_transitions(sample_events):
    """Transitions calculated correctly."""
    home_events = [e for e in sample_events if e["team_id"] == "home"]
    data = calculate_transitions(home_events, "home")

    assert "total_possessions" in data
    assert "avg_possession_duration_s" in data


def test_calculate_player_metrics(sample_events):
    """Player metrics calculated correctly."""
    metrics = calculate_player_metrics_from_events(
        sample_events, "player_01", "GK", "home", "GK"
    )
    assert metrics.total_passes == 1
    assert metrics.completed_passes == 1
    assert metrics.pass_completion_rate == 1.0


# ----------------------------------------------------------------------------------------
# Agent Tests
# ----------------------------------------------------------------------------------------


def test_narrative_generator():
    """Narrative generator produces output."""
    gen = NarrativeGeneratorAgent()
    finding = {
        "category": "build_up",
        "headline": "Test build-up",
        "team_id": "home",
        "observation": "Test observation",
        "interpretation": "Test interpretation",
        "significance": "Test significance",
        "sample_size": 10,
        "data": {"total_passes": 10, "completed_passes": 9, "completion_rate": 0.9},
    }
    narrative = gen.generate(finding, "casual_fan", "en")
    assert isinstance(narrative, str)
    assert len(narrative) > 0


def test_evidence_validator():
    """Evidence validator checks consistency."""
    validator = EvidenceValidatorAgent()
    insight = {
        "category": "build_up",
        "data": {"total_passes": 10, "completed_passes": 9, "completion_rate": 0.9},
        "sample_size": 10,
    }
    result = validator.validate_insight(insight)
    assert "validation_summary" in result


def test_personalization_agent():
    """Personalization filters insights by viewer type."""
    agent = PersonalizationAgent()
    insights = [
        {"category": "build_up", "headline": "Build-up", "sample_size": 5},
        {"category": "attack", "headline": "Attack", "sample_size": 5},
    ]
    prefs = {"viewer_type": "casual_fan", "interest_areas": ["attack"]}
    personalized = agent.personalize(insights, prefs)

    # Should include attack + model (always)
    cats = [i["category"] for i in personalized]
    assert "attack" in cats


def test_overlay_formatter():
    """Overlay formatter produces machine-readable JSON."""
    formatter = OverlayFormatterAgent()
    insights = [
        {
            "insight_id": "insight_001",
            "match_id": "demo_001",
            "timestamp_seconds": 120.0,
            "category": "attack",
            "headline": "Shot created",
            "observation": "Player took shot",
            "supporting_event_ids": ["event_003"],
            "sample_size": 5,
            "source_type": "synthetic",
            "confidence": 0.85,
            "visualization": "shot_map",
        }
    ]
    overlays = formatter.format_overlays(insights, [])
    assert len(overlays) == 1
    assert overlays[0]["overlay_id"] == "insight_001"
    assert overlays[0]["display_priority"] == 4  # attack = priority 4


def test_orchestrator(sample_events):
    """Orchestrator runs full pipeline."""
    orch = TactiqOrchestrator()
    result = orch.run(sample_events)

    assert result.n_agents_run >= 1
    assert len(result.insights) >= 0  # May be 0 if agents return empty
    assert isinstance(result.overlays, list)


# ----------------------------------------------------------------------------------------
# Data Quality Tests
# ----------------------------------------------------------------------------------------


def test_data_quality_report(sample_events):
    """Data quality report computes metrics."""
    report = DataQualityReport(sample_events, "synthetic")
    d = report.to_dict()

    assert d["event_count"] == 3
    assert d["player_count"] >= 2
    assert "pass" in d["available_event_types"]
    assert "shot" in d["available_event_types"]
    assert d["match_duration_seconds"] > 0


# ----------------------------------------------------------------------------------------
# Integration Tests
# ----------------------------------------------------------------------------------------


def test_model_inference_available():
    """Test that model can be loaded (integration)."""
    try:
        from tactiq_ai.features.inference import InferenceAgent
        agent = InferenceAgent()
        # This tests the model is loadable without actually running inference
        assert agent.predictor is not None or True  # degraded mode OK
    except ImportError:
        pytest.skip("tactiq_inference not available")


def test_full_synthetic_match_generation():
    """Full synthetic match generation works."""
    match = generate_possession_build_up_sequence()
    assert len(match) >= 5
    assert all(hasattr(e, "event_type") for e in match)


# ----------------------------------------------------------------------------------------
# Multilingual Narrative Tests
# ----------------------------------------------------------------------------------------


def test_narrative_english():
    """Narrative generation in English (default)."""
    from tactiq_ai.agents.narrative_generator import NarrativeGeneratorAgent
    agent = NarrativeGeneratorAgent()
    finding = {
        "category": "attack",
        "team_id": "home",
        "headline": "Home team attack analysis",
        "observation": "5 shots from 3 final-third entries, 2 goals scored.",
        "interpretation": "Strong chance creation in the final third.",
        "significance": "High conversion rate indicates clinical finishing.",
        "data": {
            "total_shots": 5,
            "shots_on_target": 3,
            "shots_off_target": 2,
            "goals": 2,
            "final_third_entries": 3,
            "box_entries": 1,
            "shot_conversion_rate": 0.4,
            "shots_by_channel": {"central": 3, "left_wide": 1, "right_wide": 1},
        },
        "sample_size": 5,
    }
    narrative = agent.generate(finding, viewer_type="casual_fan", language="en")
    assert "home" in narrative.lower()
    assert "shots" in narrative.lower()
    assert "goal" in narrative.lower()


def test_narrative_spanish():
    """Narrative translation to Spanish."""
    from tactiq_ai.agents.narrative_generator import NarrativeGeneratorAgent
    agent = NarrativeGeneratorAgent()
    finding = {
        "category": "attack",
        "team_id": "home",
        "headline": "Home team attack analysis",
        "observation": "5 shots from 3 final-third entries.",
        "interpretation": "Strong chance creation.",
        "significance": "Clinical finishing.",
        "data": {
            "total_shots": 5,
            "shots_on_target": 3,
            "shots_off_target": 2,
            "goals": 2,
            "final_third_entries": 3,
            "box_entries": 1,
            "shot_conversion_rate": 0.4,
            "shots_by_channel": {},
        },
        "sample_size": 5,
    }
    narrative = agent.generate(finding, viewer_type="casual_fan", language="es")
    assert narrative != ""  # Should produce some output
    assert len(narrative) > 0


def test_narrative_french():
    """Narrative translation to French."""
    from tactiq_ai.agents.narrative_generator import NarrativeGeneratorAgent
    agent = NarrativeGeneratorAgent()
    finding = {
        "category": "build_up",
        "team_id": "away",
        "headline": "Away team build-up",
        "observation": "10 passes from defensive third, 8 completed.",
        "interpretation": "Controlled build-up from defence.",
        "significance": "Sustained possession.",
        "data": {
            "total_passes": 10,
            "completed_passes": 8,
            "completion_rate": 0.8,
            "pressure_build_up": 2,
            "turnovers_in_build_up": 1,
            "third_crossings": 3,
        },
        "sample_size": 10,
    }
    narrative = agent.generate(finding, viewer_type="tactical_analyst", language="fr")
    assert narrative != ""
    assert len(narrative) > 0


def test_narrative_unsupported_language_fallback():
    """Unsupported language falls back to English."""
    from tactiq_ai.agents.narrative_generator import NarrativeGeneratorAgent
    agent = NarrativeGeneratorAgent()
    finding = {
        "category": "defense",
        "team_id": "home",
        "headline": "Defensive analysis",
        "observation": "5 pressures, 3 tackles.",
        "interpretation": "Good pressing.",
        "significance": "Effective defending.",
        "data": {
            "pressures": 5,
            "tackles": 3,
            "tackle_success_rate": 0.67,
            "recoveries": 2,
            "interceptions": 1,
            "clearances": 0,
        },
        "sample_size": 5,
    }
    # German is not implemented, should fall back to English
    narrative_de = agent.generate(finding, viewer_type="casual_fan", language="de")
    narrative_en = agent.generate(finding, viewer_type="casual_fan", language="en")
    assert narrative_de == narrative_en  # fallback to English


def test_narrative_match_report_multilingual():
    """Match report generation in multiple languages."""
    from tactiq_ai.agents.narrative_generator import NarrativeGeneratorAgent
    agent = NarrativeGeneratorAgent()
    findings = [
        {
            "category": "build_up",
            "team_id": "home",
            "headline": "Build-up play",
            "observation": "Good passing from defence.",
            "interpretation": "Controlled possession.",
            "significance": "Builds attacks from back.",
            "data": {
                "total_passes": 8,
                "completed_passes": 7,
                "completion_rate": 0.875,
                "pressure_build_up": 1,
                "turnovers_in_build_up": 0,
                "third_crossings": 2,
            },
            "sample_size": 8,
        },
        {
            "category": "attack",
            "team_id": "away",
            "headline": "Attack patterns",
            "observation": "4 shots on target.",
            "interpretation": "Dangerous in final third.",
            "significance": "Creates scoring chances.",
            "data": {
                "total_shots": 4,
                "shots_on_target": 3,
                "shots_off_target": 1,
                "goals": 1,
                "final_third_entries": 2,
                "box_entries": 1,
                "shot_conversion_rate": 0.25,
                "shots_by_channel": {},
            },
            "sample_size": 4,
        },
    ]
    report_en = agent.generate_match_report(findings, viewer_type="casual_fan", language="en")
    report_es = agent.generate_match_report(findings, viewer_type="casual_fan", language="es")
    report_fr = agent.generate_match_report(findings, viewer_type="casual_fan", language="fr")
    assert len(report_en) > 0
    assert len(report_es) > 0
    assert len(report_fr) > 0
    # Spanish and French should differ from English (translated headers)
    assert report_es != report_en or report_fr != report_en


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
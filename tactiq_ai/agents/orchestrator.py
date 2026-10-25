"""TACTIQ AI - Agent Orchestrator.

Coordinates the specialized agent modules to produce tactical insights
from match events, including model inference, evidence validation,
personalization, and overlay formatting.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import SOURCE_SYNTHETIC
from .event_interpreter import EventInterpreterAgent
from .tactical_analyst import TacticalAnalystAgent
from .keras_inference_agent import KerasInferenceAgent
from .player_analyst import PlayerAnalystAgent
from .narrative_generator import NarrativeGeneratorAgent
from .evidence_validator import EvidenceValidatorAgent
from .personalization import PersonalizationAgent
from .overlay_formatter import OverlayFormatterAgent

logger = logging.getLogger(__name__)


@dataclass
class Insight:
    """A structured insight object for tactical analysis."""

    insight_id: str
    match_id: str
    timestamp_seconds: float
    category: str
    headline: str
    observation: str
    interpretation: str
    significance: str
    supporting_event_ids: List[str]
    sample_size: Optional[int]
    source_type: str
    confidence: Optional[float]
    visualization: str
    player_ids: List[str] = None
    display_priority: int = 0


@dataclass
class AgentResult:
    """Result from running a single agent."""
    agent_name: str
    category: str
    findings: List[Dict[str, Any]]
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrchestratorResult:
    """Result from running the full orchestrator."""
    insights: List[Insight]
    agents: Dict[str, AgentResult]
    overlays: List[Dict[str, Any]]
    timeline_insights: List[Dict[str, Any]]
    generated_at: float
    n_agents_run: int
    errors: List[str] = field(default_factory=list)


class TactiqOrchestrator:
    """Orchestrates the specialized TACTIQ AI agents.

    Coordinates:
    - EventInterpreterAgent: validates events, tracks match state, sequences
    - TacticalAnalystAgent: computes build-up, attack, defence, transitions
    - KerasInferenceAgent: runs the existing .keras model
    - PlayerAnalystAgent: player-focused insights
    - NarrativeGeneratorAgent: converts findings into explanations
    - EvidenceValidatorAgent: validates statements against stats
    - PersonalizationAgent: adapts insights to viewer preferences
    - OverlayFormatterAgent: creates timestamped overlays
    """

    def __init__(self):
        self.event_interpreter = EventInterpreterAgent()
        self.tactical_analyst = TacticalAnalystAgent()
        self.keras_inference = KerasInferenceAgent()
        self.player_analyst = PlayerAnalystAgent()
        self.narrative_generator = NarrativeGeneratorAgent()
        self.evidence_validator = EvidenceValidatorAgent()
        self.personalization = PersonalizationAgent()
        self.overlay_formatter = OverlayFormatterAgent()

        self.insights: List[Insight] = []
        self.overlays: List[Dict[str, Any]] = []
        self.errors: List[str] = []

    def run(self, events: List[Dict[str, Any]], context: Dict[str, Any] = None) -> OrchestratorResult:
        """Run the full orchestrator pipeline.

        Args:
            events: List of validated match events
            context: Optional context (viewer preferences, team focus, etc.)

        Returns:
            OrchestratorResult with insights, overlays, and per-agent results.
        """
        started = time.time()
        context = context or {}
        viewer_prefs = context.get("viewer_preferences") or context.get("preferences") or {}

        self.insights = []
        self.overlays = []
        self.errors = []
        agent_results: Dict[str, AgentResult] = {}

        try:
            # 1. Event Interpreter
            interpreter_result = self.event_interpreter.interpret(events)
            agent_results["event_interpreter"] = AgentResult(
                agent_name="event_interpreter",
                category="interpretation",
                findings=interpreter_result,
            )

            # 2. Tactical Analyst
            tactical_result = self.tactical_analyst.analyze(events)
            agent_results["tactical_analyst"] = AgentResult(
                agent_name="tactical_analyst",
                category="tactics",
                findings=tactical_result,
            )

            # 3. Keras Inference
            inference_result = self.keras_inference.run(events)
            agent_results["keras_inference"] = AgentResult(
                agent_name="keras_inference",
                category="model",
                findings=inference_result,
                confidence=(inference_result[0].get("confidence") if inference_result and isinstance(inference_result[0], dict) else None),
            )

            # 4. Player Analyst
            player_result = self.player_analyst.analyze(events)
            agent_results["player_analyst"] = AgentResult(
                agent_name="player_analyst",
                category="players",
                findings=player_result,
            )

        except Exception as e:
            logger.error(f"Agent pipeline error: {e}")
            self.errors.append(str(e))

        # Generate insights from findings (deterministic, evidence-based)
        for result in agent_results.values():
            for finding in result.findings:
                if not finding or not isinstance(finding, dict):
                    continue

                insight = self._create_insight_from_finding(finding, events, viewer_prefs)
                if insight is not None:
                    self.insights.append(insight)

        # Personalize insights
        personalized_insights = self.personalization.personalize(self.insights, viewer_prefs)

        # Format overlays
        overlays = self.overlay_formatter.format_overlays(
            personalized_insights, events
        )

        elapsed = time.time() - started
        logger.info(f"Orchestrator ran in {elapsed:.2f}s with {len(self.insights)} insights")

        return OrchestratorResult(
            insights=personalized_insights,
            agents=agent_results,
            overlays=overlays,
            timeline_insights=[],
            generated_at=started,
            n_agents_run=len(agent_results),
            errors=self.errors,
        )

    def _create_insight_from_finding(
        self,
        finding: Dict[str, Any],
        events: List[Dict[str, Any]],
        viewer_prefs: Dict[str, Any],
    ) -> Optional[Insight]:
        """Convert an agent finding into a structured insight."""
        category = finding.get("category", "transition")
        if category not in (
            "build_up", "attack", "defense", "transition",
            "player", "model", "moment"
        ):
            category = "transition"

        timestamp = finding.get("timestamp_seconds") or 0.0
        headline = finding.get("headline") or finding.get("title") or "Match moment"

        return Insight(
            insight_id=f"insight_{abs(hash((headline, timestamp)) % 100000):03d}",
            match_id=finding.get("match_id", "match_001"),
            timestamp_seconds=timestamp,
            category=category,
            headline=headline,
            observation=finding.get("observation", ""),
            interpretation=finding.get("interpretation", ""),
            significance=finding.get("significance", ""),
            supporting_event_ids=finding.get("supporting_event_ids", []),
            sample_size=finding.get("sample_size"),
            source_type=finding.get("source_type", SOURCE_SYNTHETIC),
            confidence=finding.get("confidence"),
            visualization=finding.get("visualization", "pitch_highlight"),
            player_ids=finding.get("player_ids") or [],
            display_priority=finding.get("display_priority", 0),
        )

    def generate_overlays(
        self,
        insights: List[Insight],
        events: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate machine-readable overlay JSON objects."""
        return self.overlay_formatter.format_overlays(insights, events)

    def get_tactical_report(
        self,
        insights: List[Insight],
    ) -> Dict[str, Any]:
        """Generate a final tactical match report from insights."""
        return {
            "generated_at": time.time(),
            "n_insights": len(insights),
            "insights": [
                {
                    "insight_id": i.insight_id,
                    "category": i.category,
                    "headline": i.headline,
                    "timestamp_seconds": i.timestamp_seconds,
                    "source_type": i.source_type,
                }
                for i in insights
            ],
        }


_default_orchestrator = None


def get_orchestrator() -> TactiqOrchestrator:
    global _default_orchestrator
    if _default_orchestrator is None:
        _default_orchestrator = TactiqOrchestrator()
    return _default_orchestrator

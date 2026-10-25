"""TACTIQ AI - Personalization Agent.

Agent 7: Adapts insights, language, technical depth, and visual presentation
to the viewer's preferences.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PersonalizationAgent:
    """Agent 7: personalizes insights based on viewer preferences."""

    VIEWER_PROFILES = {
        "casual_fan": {
            "explanation_depth": "short",
            "jargon": "minimal",
            "key_metrics_only": True,
            "visual_density": "high",
            "use_model_predictions": False,
            "multilingual": False,
        },
        "tactical_analyst": {
            "explanation_depth": "detailed",
            "jargon": "full",
            "key_metrics_only": False,
            "visual_density": "medium",
            "use_model_predictions": True,
            "multilingual": False,
        },
        "player_focus": {
            "explanation_depth": "centered",
            "jargon": "moderate",
            "key_metrics_only": True,
            "visual_density": "high",
            "use_model_predictions": False,
            "multilingual": True,
        },
        "custom": {
            "explanation_depth": "variable",
            "jargon": "variable",
            "key_metrics_only": True,
            "visual_density": "medium",
            "use_model_predictions": True,
            "multilingual": True,
        },
    }

    INTEREST_AREAS = [
        "pressing",
        "progressive_passing",
        "chance_creation",
        "defensive_recoveries",
        "build_up",
        "transitions",
    ]

    def personalize(
        self,
        insights: List[Dict[str, Any]],
        viewer_prefs: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Adapt insights to viewer preferences.

        Args:
            insights: Raw insights from orchestrator
            viewer_prefs: Dict with keys: viewer_type, language, interest_areas, depth

        Returns:
            Personalized insights list, filtered and adapted.
        """
        viewer_type = viewer_prefs.get("viewer_type", "casual_fan")
        language = viewer_prefs.get("language", "en")
        depth = viewer_prefs.get("depth", "medium")
        interest_areas = viewer_prefs.get("interest_areas", [])

        # Use default profile
        if viewer_type not in self.VIEWER_PROFILES:
            viewer_type = "casual_fan"
        profile = self.VIEWER_PROFILES[viewer_type]

        # Adjust depth
        if depth == "shallow":
            profile["explanation_depth"] = "short"
        elif depth == "deep":
            profile["explanation_depth"] = "detailed"
        else:
            profile["explanation_depth"] = depth

        # Apply interest area filtering
        filtered = []
        for insight in insights:
            # Keep all insights if no filtering specified
            if not interest_areas:
                adapted = self._adapt_depth(insight, profile, language, viewer_type)
                adapted["language"] = language
                adapted["viewer_type"] = viewer_type
                filtered.append(adapted)
            elif insight.get("category") in interest_areas:
                adapted = self._adapt_depth(insight, profile, language, viewer_type)
                adapted["language"] = language
                adapted["viewer_type"] = viewer_type
                filtered.append(adapted)
            elif insight.get("category") == "model":
                # Always include model predictions even in filtered mode
                adapted = self._adapt_depth(insight, profile, language, viewer_type)
                adapted["language"] = language
                adapted["viewer_type"] = viewer_type
                filtered.append(adapted)

        return filtered

    def _adapt_depth(
        self,
        insight: Any,
        profile: Dict[str, Any],
        language: str = "en",
        viewer_type: str = "casual_fan",
    ) -> Dict[str, Any]:
        """Adapt a single insight's depth and style per profile."""
        # Convert Insight dataclass to dict if needed
        if dataclasses.is_dataclass(insight):
            adapted = dataclasses.asdict(insight)
        elif isinstance(insight, dict):
            adapted = dict(insight)
        else:
            adapted = {}

        depth = profile["explanation_depth"]
        jargon = profile["jargon"]
        use_model = profile["use_model_predictions"]
        multilingual = profile["multilingual"]

        # Truncate explanations for shallow depth
        if depth == "short":
            obs = adapted.get("observation", "")
            if len(obs) > 200:
                adapted["observation"] = obs[:200] + "..."
        else:
            pass

        if not use_model and adapted.get("category") == "model":
            adapted = {k: v for k, v in adapted.items() if k != "model_result"}
            adapted["category"] = "moment"

        adapted["visual_density"] = profile["visual_density"]
        adapted["language"] = language
        adapted["jargon_level"] = jargon
        adapted["viewer_profile"] = viewer_type

        return adapted

    def apply_interest_filters(
        self,
        insights: List[Dict[str, Any]],
        interest_areas: List[str],
    ) -> List[Dict[str, Any]]:
        """Filter insights by interest areas (used separately from personalization).

        Args:
            insights: List of insights
            interest_areas: List of category names to keep

        Returns:
            Filtered insights list
        """
        return [i for i in insights if i.get("category") in interest_areas]

    def get_viewer_profile_summary(self, viewer_prefs: Dict[str, Any]) -> str:
        """Return a human-readable summary of the viewer profile."""
        viewer_type = viewer_prefs.get("viewer_type", "casual_fan")
        profile = self.VIEWER_PROFILES.get(viewer_type, self.VIEWER_PROFILES["casual_fan"])

        summary_parts = []
        summary_parts.append(f"Viewer Type: {viewer_type}")
        summary_parts.append(f"Explanation Depth: {profile['explanation_depth']}")
        summary_parts.append(f"Jargon Level: {profile['jargon']}")
        summary_parts.append(f"Model Predictions: {'Enabled' if profile['use_model_predictions'] else 'Disabled'}")
        summary_parts.append(f"Multilingual: {'Enabled' if profile['multilingual'] else 'Disabled'}")
        summary_parts.append(f"Key Metrics Only: {'Yes' if profile['key_metrics_only'] else 'No'}")

        return " | ".join(summary_parts)
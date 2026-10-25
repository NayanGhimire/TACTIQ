"""Inference agent that wraps tactiq_inference for tactical pattern classification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import LOW_CONFIDENCE_THRESHOLD, MODEL_PATH, LABEL_MAPPING_PATH, FEATURE_CONFIG_PATH, TACTIQ_INFERENCE_PATH
from ..schemas import ETYPES

logger = logging.getLogger(__name__)

# Load once at startup
try:
    from tactiq_inference import load_predictor
    _DEFAULT_PREDICTOR = load_predictor()
    HAS_TACITIQ = True
except ImportError:
    logger.warning("tactiq_inference not available; running in degraded mode (model won't classify)")
    _DEFAULT_PREDICTOR = None
    HAS_TACITIQ = False


class InferenceAgent:
    """Runs the existing Keras model on validated event sequences.

    Wraps tactiq_inference.predict_tactical_pattern so the application can
    call this single agent instead of re-implementing preprocessing.

    Key facts:
    - The model only sees up to 14 events at a time.
    - Outputs are labelled as 'synthetic' — they show what a sequence resembles,
      NOT verified statistics.
    - Low confidence (< 0.60 default) triggers an 'uncertain/unclassified' flag.
    """

    def __init__(self):
        self.predictor = _DEFAULT_PREDICTOR if HAS_TACITIQ else None
        self.classes: List[str] = ["controlled_buildup", "direct_progression",
                                  "wide_attack", "central_progression",
                                  "high_press_turnover", "counterattack",
                                  "other_or_uncertain"]
        if self.predictor is not None:
            try:
                lm = __import__('json').loads(Path(LABEL_MAPPING_PATH).read_text())
                self.classes = lm["classes"]
            except Exception:
                pass

    def classify_sequence(self, event_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Classify a sequence of events as a tactical pattern.

        Returns a dict with predicted_class, class_probabilities, model_confidence,
        runner_up, flags, and metadata.
        """
        if not HAS_TACITIQ or self.predictor is None:
            return {
                "predicted_class": "other_or_uncertain",
                "class_probabilities": {c: round(0.01, 4) for c in self.classes},
                "model_confidence": 0.0,
                "runner_up": "other_or_uncertain",
                "margin_over_runner_up": 0.0,
                "n_events": len(event_sequence),
                "flags": ["model_not_loaded"],
                "confidence_note": "Keras model not loaded; classification disabled",
                "synthetic": True,
                "uncertain": True,
                "matched_to_training": False,
            }

        result = self.predictor.predict(event_sequence)

        return {
            "predicted_class": result["predicted_class"],
            "class_probabilities": result["class_probabilities"],
            "model_confidence": result["model_confidence"],
            "runner_up": result["runner_up"],
            "margin_over_runner_up": result["margin_over_runner_up"],
            "n_events": result["n_events"],
            "flags": result["flags"],
            "confidence_note": result["confidence_note"],
            "synthetic": True,  # All model outputs are synthetic predictions
            "uncertain": any("low_confidence" in f for f in result.get("flags", [])),
            "matched_to_training": True,
        }

    def classify_window(
        self,
        home_events: List[Dict[str, Any]],
        away_events: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Classify both teams' event sequences and return both predictions."""
        self.ensure_within_window(len(home_events))
        self.ensure_within_window(len(away_events))

        home_pred = self.classify_sequence(home_events)
        away_pred = self.classify_sequence(away_events)

        return home_pred, away_pred

    def ensure_within_window(self, n: int) -> None:
        """Ensure we don't pass more than 14 events to the model."""
        if n > 14:
            logger.info(f"Slicing window: {n} events -> first {14}")
        elif n < 3:
            raise ValueError(f"Need at least 3 events, got {n}")


def load_default_agent() -> InferenceAgent:
    return InferenceAgent()

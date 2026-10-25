"""Keras Inference Agent.

Agent 3: Runs the existing .keras model on validated event sequences and
returns tactical pattern classifications.

Wraps tactiq_inference.TactiqPredictor, applying the rich->model schema adapter.
All outputs are labelled as synthetic predictions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..features.inference import InferenceAgent

logger = logging.getLogger(__name__)


class KerasInferenceAgent:
    """Agent that runs the existing Keras model classifier."""

    def __init__(self):
        self.inference_agent = InferenceAgent()

    def run(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run model inference on the event stream.

        Extracts sliding windows of up to 14 events per team, classifies each,
        and returns findings with timestamps and model output.

        Returns list of findings with:
        - predicted_class, model_confidence, flags, synthetic=True
        - supporting_event_ids, timestamp_seconds
        """
        from ..features.adapter import events_to_model_dict, extract_model_window

        teams = sorted(set(e.get("team_id", "home") for e in events if e.get("team_id")))

        findings = []

        for team_id in teams:
            team_events = [e for e in events if e.get("team_id") == team_id]
            # Convert rich events to model format (rescale coords, map types)
            model_events = events_to_model_dict(team_events, team_id)

            # Extract windows
            from ..schemas import T_MAX
            windows = []
            n = len(model_events)
            if n >= 3:
                # Use sliding windows with stride 3
                stride = max(1, n // 6)
                for start in range(0, max(0, n - 3), stride):
                    window = model_events[start:start + T_MAX]
                    windows.append(window)

                # Also process the full sequence if < T_MAX
                if n <= T_MAX:
                    windows.append(model_events)

                # Deduplicate
                seen = set()
                unique_windows = []
                for w in windows:
                    key = tuple((e["type"], round(e["x_start"], 1), round(e["y_start"], 1)) for e in w)
                    if key not in seen:
                        seen.add(key)
                        unique_windows.append(w)
                windows = unique_windows[:3]  # limit to 3 windows per team

                for window in windows:
                    try:
                        result = self.inference_agent.classify_sequence(window)
                        ts = float(window[0].get("t_start_s", window[0].get("timestamp_seconds", 0)))

                        findings.append({
                            "category": "model",
                            "headline": f"{team_id}: model classifies as {result['predicted_class']}",
                            "observation": (
                                f"Model prediction: {result['predicted_class']} "
                                f"(confidence: {result['model_confidence']:.2f}, "
                                f"runner-up: {result['runner_up']})"
                            ),
                            "interpretation": (
                                "This is a synthetic prediction based on training with "
                                "synthetically-rule-labelled sequences. "
                                "It shows what the sequence resembles under those rules, not a verified tactical finding."
                            ),
                            "significance": (
                                f"Model flagged this as {result['predicted_class']}. "
                                "Use as one signal among several deterministic statistics."
                            ),
                            "supporting_event_ids": [
                                w.get("event_id", "") or f"w_{i}"
                                for i, w in enumerate(window)
                            ],
                            "sample_size": result["n_events"],
                            "source_type": "model_prediction",
                            "confidence": result["model_confidence"],
                            "visualization": "pitch_highlight",
                            "team_id": team_id,
                            "model_result": result,
                            "synthetic": True,
                            "timestamp_seconds": ts,
                            "uncertain": result.get("uncertain", False),
                        })

                        if result.get("flags"):
                            findings[-1]["model_flags"] = result["flags"]

                    except Exception as e:
                        logger.warning(f"Inference window error: {e}")
                        findings.append({
                            "category": "model",
                            "headline": f"{team_id}: classification failed",
                            "observation": f"Model inference error: {e}",
                            "interpretation": "Insufficient valid events for model input.",
                            "significance": "",
                            "supporting_event_ids": [],
                            "sample_size": 0,
                            "source_type": "model_prediction",
                            "confidence": None,
                            "visualization": None,
                            "team_id": team_id,
                            "synthetic": True,
                            "model_error": str(e),
                        })

        return findings

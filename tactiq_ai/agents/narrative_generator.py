"""TACTIQ AI - Narrative Generator Agent.

Agent 5: Converts verified findings into clear football explanations.
Uses deterministic templates with multilingual fallback.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NarrativeGeneratorAgent:
    """Agent 5: converts findings into explanations using deterministic templates."""

    TEMPLATES = {
        "build_up_casual": (
            "{team_cap} is building attacks from the back with {rate:.0%} pass accuracy. "
            "They've made {passes} passes out of defence, "
            "{pressure} of which came under pressure."
        ),
        "build_up_analyst": (
            "BUILD-UP ANALYSIS — {team_cap}\n"
            "Defensive-third pass sequence: {passes} passes attempted, "
            "{completed} completed ({rate:.0%} completion rate).\n"
            "Pressure events during build-up: {pressure} ({pressure_pct:.0%} of total).\n"
            "Turnovers during build-up: {turnovers}.\n"
            "Third crossings: {crossings}.\n"
            "Sample size: {n} events."
        ),
        "attack_casual": (
            "{team_cap} has had {shots} shots ({ontarget} on target) "
            "from {entries} final-third entries. "
            "They've scored {goals} goal(s) so far."
        ),
        "attack_analyst": (
            "ATTACKING ANALYSIS — {team_cap}\n"
            "Final-third entries: {entries}\n"
            "Box entries: {box}\n"
            "Total shots: {shots} (on target: {ontarget}, off target: {offtarget})\n"
            "Goals: {goals} | Shot conversion: {conv:.0%}\n"
            "Shots by channel: {channels}\n"
            "Sample size: {n} shots."
        ),
        "defense_casual": (
            "{team_cap} has applied {pressures} pressing actions and made "
            "{tackles} tackles ({winrate:.0%} success rate)."
        ),
        "defense_analyst": (
            "DEFENSIVE ANALYSIS — {team_cap}\n"
            "Pressure events: {pressures} ({success:.0%} success rate)\n"
            "Tackles: {tackles} ({won} won, {winrate:.0%} success)\n"
            "Recoveries: {recoveries} | Interceptions: {interceptions}\n"
            "Clearances: {clearances}\n"
            "Defensive-third turnovers: {turnovers}\n"
            "Opponent progression allowed: {progression}\n"
            "Zones with most defensive actions: {zones}"
        ),
        "transition_casual": (
            "{team_cap} has won possession {possessions} times, "
            "averaging {duration}s per possession. "
            "They've counter-pressed {counter} times."
        ),
        "transition_analyst": (
            "TRANSITION ANALYSIS — {team_cap}\n"
            "Possession changes: {possessions}\n"
            "Avg possession duration: {duration}s\n"
            "Counter-presses: {counter}\n"
            "Dangerous turnovers: {dangerous}\n"
            "Average time between events: {tempo}s\n"
            "Match rhythm: {rhythm}"
        ),
        "player_casual": (
            "{name} has been involved in {n} events for {team_cap}, "
            "completing {passes} of {attempted} passes "
            "({rate:.0%} accuracy)."
        ),
        "player_analyst": (
            "PLAYER FOCUS — {name} ({team_cap}, {position})\n"
            "Involvement: {n} events\n"
            "Passing: {passes}/{attempted} ({rate:.0%})\n"
            "Carries: {carries} ({carries_done} successful)\n"
            "Defensive actions: {pressures} pressures, {tackles} tackles\n"
            "Shot: {shots} ({on_target} on target, {goals} goals)\n"
            "Zone activity: {zones}\n"
            "Channel activity: {channels}"
        ),
        "model_casual": (
            "TACTIQ AI model sees this sequence as resembling: {prediction} "
            "(confidence: {confidence:.0%}). "
            "Remember: this is a prediction based on synthetic training data, "
            "not a real-match finding."
        ),
        "model_analyst": (
            "MODEL CLASSIFICATION (synthetic training)\n"
            "Predicted class: {prediction} (softmax: {confidence:.4f})\n"
            "Runner-up: {runner_up} (margin: {margin:.4f})\n"
            "Sequence length: {n} events\n"
            "Note: {note}\n"
            "Flags: {flags}"
        ),
        "uncertain": (
            "TACTIQ AI could not confidently classify this sequence. "
            "The model returned '{prediction}' with low confidence ({confidence:.0%}). "
            "This may indicate insufficient evidence or an unusual sequence."
        ),
        "moment_casual": (
            "At {time}m: {headline}. {observation}"
        ),
        "moment_analyst": (
            "KEY MOMENT — {timestamp}s\n"
            "Headline: {headline}\n"
            "What: {observation}\n"
            "Why it matters: {interpretation}\n"
            "Tactical significance: {significance}\n"
            "Evidence: {n} supporting events from sample size {sample_size}\n"
            "Players: {players}"
        ),
    }

    def generate(
        self,
        finding: Dict[str, Any],
        viewer_type: str = "casual_fan",
        language: str = "en",
    ) -> str:
        """Generate a narrative explanation for a single finding.

        Args:
            finding: A finding dict from the tactical analyst or classifier
            viewer_type: One of casual_fan, tactical_analyst, player_focus, custom
            language: ISO language code (en, es, fr, de, it)

        Returns:
            Narrative string in the requested language.
        """
        category = finding.get("category", "")
        view_suffix = "casual" if viewer_type == "casual_fan" else (
            "analyst" if viewer_type == "tactical_analyst" else
            category if category in ("player",) else "analyst"
        )

        # Select template
        if category == "model":
            if finding.get("uncertain", False):
                template_key = f"uncertain_{view_suffix}" if f"uncertain_{view_suffix}" in self.TEMPLATES else "uncertain"
            else:
                template_key = f"model_{view_suffix}"
        elif category == "build_up":
            template_key = f"build_up_{view_suffix}"
        elif category == "attack":
            template_key = f"attack_{view_suffix}"
        elif category == "defense":
            template_key = f"defense_{view_suffix}"
        elif category == "transition":
            template_key = f"transition_{view_suffix}"
        elif category == "player":
            template_key = f"player_{view_suffix}"
        elif category == "moment":
            template_key = f"moment_{view_suffix}"
        else:
            template_key = f"moment_{view_suffix}"

        template = self.TEMPLATES.get(template_key, self.TEMPLATES.get("moment_casual"))

        if template is None:
            return finding.get("headline", "No narrative available.")

        # Build format context
        data = finding.get("data", finding.get("model_result", {}))
        team = finding.get("team_id", "")
        team_cap = team.upper() if team else ""
        result = finding.get("model_result", {})

        ctx = {
            "team_cap": team_cap,
            "team": team,
            "headline": finding.get("headline", ""),
            "observation": finding.get("observation", ""),
            "interpretation": finding.get("interpretation", ""),
            "significance": finding.get("significance", ""),
            "sample_size": finding.get("sample_size", 0),
            "players": ", ".join(finding.get("player_ids", []) or []),
            "timestamp": round(finding.get("timestamp_seconds", 0), 1),
            "time": round(finding.get("timestamp_seconds", 0) / 60, 1),
            "prediction": result.get("predicted_class", data.get("predicted_class", "unknown")),
            "confidence": result.get("model_confidence", 0.0),
            "runner_up": result.get("runner_up", ""),
            "margin": result.get("margin_over_runner_up", 0.0),
            "note": result.get("confidence_note", ""),
            "flags": ", ".join(result.get("flags", [])) if result else "",
            "n": finding.get("sample_size", 0),
        }

        # Fill in category-specific data
        if "data" in finding:
            d = finding["data"]
            if isinstance(d, dict):
                ctx.update({
                    "passes": d.get("total_passes", d.get("total_passes", 0)),
                    "completed": d.get("completed_passes", 0),
                    "rate": d.get("completion_rate", 0.0),
                    "pressure": d.get("pressure_build_up", d.get("pressures", 0)),
                    "turnovers": d.get("turnovers_in_build_up", 0),
                    "crossings": d.get("third_crossings", 0),
                    "shots": d.get("total_shots", 0),
                    "ontarget": d.get("shots_on_target", 0),
                    "offtarget": d.get("shots_off_target", 0),
                    "goals": d.get("goals", 0),
                    "entries": d.get("final_third_entries", 0),
                    "box": d.get("box_entries", 0),
                    "conv": d.get("shot_conversion_rate", 0.0),
                    "channels": d.get("shots_by_channel", {}),
                    "pressures": d.get("pressures", 0),
                    "tackles": d.get("tackles", 0),
                    "winrate": d.get("tackle_success_rate", 0.0),
                    "won": sum(1 for t in d.get("defensive_actions", {}).get("tackles", []) if t.get("won")),
                    "interceptions": d.get("interceptions", 0),
                    "clearances": d.get("clearances", 0),
                })
            else:
                m = data  # PlayerMetrics dataclass
                ctx.update({
                    "name": finding.get("player_id", ""),
                    "passes": m.total_passes if hasattr(m, 'total_passes') else data.get("total_passes", 0),
                    "attempted": m.total_passes if hasattr(m, 'total_passes') else 0,
                    "rate": m.pass_completion_rate if hasattr(m, 'pass_completion_rate') else 0,
                    "carries": m.total_carries if hasattr(m, 'total_carries') else data.get("total_carries", 0),
                    "carries_done": m.successful_carries if hasattr(m, 'successful_carries') else 0,
                    "pressures": m.pressures_applied if hasattr(m, 'pressures_applied') else 0,
                    "tackles": m.tackles if hasattr(m, 'tackles') else 0,
                    "shots": m.total_shots if hasattr(m, 'total_shots') else 0,
                    "on_target": m.shots_on_target if hasattr(m, 'shots_on_target') else 0,
                    "goals": m.goals if hasattr(m, 'goals') else 0,
                })

        # Also handle model_result
        if "model_result" in finding:
            mr = finding["model_result"]
            ctx.update({
                "prediction": mr.get("predicted_class", "unknown"),
                "confidence": mr.get("model_confidence", 0.0),
                "runner_up": mr.get("runner_up", ""),
                "margin": mr.get("margin_over_runner_up", 0.0),
                "note": mr.get("confidence_note", ""),
                "flags": ", ".join(mr.get("flags", [])),
            })

        # Handle transitions
        if category == "transition" and "data" in finding:
            d = finding["data"]
            rhythm = "end-to-end" if d.get("avg_possession_duration_s", 0) < 30 else "sustained control"
            ctx.update({
                "possessions": d.get("total_possessions", 0),
                "duration": d.get("avg_possession_duration_s", 0),
                "counter": d.get("counter_presses", 0),
                "dangerous": d.get("dangerous_turnovers", 0),
                "tempo": d.get("avg_time_between_events_s", 0),
                "rhythm": rhythm,
            })

        try:
            narrative = template.format(**ctx)
        except KeyError as e:
            logger.warning(f"Missing template key: {e}")
            narrative = finding.get("headline", "No narrative available.")

        return self._translate(narrative, language, viewer_type)

    def generate_match_report(
        self,
        findings: List[Dict[str, Any]],
        viewer_type: str = "casual_fan",
        language: str = "en",
    ) -> str:
        """Generate a complete match report from findings."""
        sections = []

        if viewer_type == "casual_fan":
            sections.append("=== TACTIQ AI Match Report ===\n")
        else:
            sections.append("=== TACTIQ AI - Tactical Match Report ===\n")

        for finding in findings:
            narrative = self.generate(finding, viewer_type, language)
            categories = {
                "build_up": "[Build-up]",
                "attack": "[Attack]",
                "defense": "[Defense]",
                "transition": "[Transitions]",
                "player": "[Player]",
                "model": "[Model]",
                "moment": "[Moment]",
            }
            cat_label = categories.get(finding.get("category", ""), "")
            sections.append(f"\n{cat_label} {narrative}\n")

        report = "\n".join(sections)
        if viewer_type == "casual_fan":
            report += "\n\n(Tip: Switch to 'Analyst Mode' for detailed statistics)"

        return report

    def _translate(self, text: str, language: str, viewer_type: str) -> str:
        """Translate narrative to the requested language.

        Uses simple deterministic templates for French and Spanish.
        Falls back to English if translation is unavailable.
        """
        if language == "en":
            return text

        # For non-English, use pre-translated key phrases
        if language == "es":
            return self._translate_es(text, viewer_type)
        elif language == "fr":
            return self._translate_fr(text, viewer_type)

        return text

    def _translate_es(self, text: str, viewer_type: str) -> str:
        """Simple Spanish translation for common phrases."""
        replacements = {
            "TACTIQ AI": "TACTIQ AI",
            "Match Report": "Informe del Partido",
            "Tactical Match Report": "Informe Táctico del Partido",
            "BUILD-UP ANALYSIS": "ANÁLISIS DE CONSTRUCCIÓN",
            "ATTACKING ANALYSIS": "ANÁLISIS DE ATAQUE",
            "DEFENSIVE ANALYSIS": "ANÁLISIS DEFENSIVO",
            "TRANSITION ANALYSIS": "ANÁLISIS DE TRANSICIÓN",
            "PLAYER FOCUS": "ENFOQUE EN JUGADOR",
            "MODEL CLASSIFICATION": "CLASIFICACIÓN DEL MODELO",
            "is building attacks": "está construyendo ataques",
            "pass accuracy": "precisión de pase",
            "shots": "tiros",
            "on target": "a puerta",
            "goals": "goles",
            "pressing actions": "acciones de presión",
            "tackles": "entradas",
            "recoveries": "recuperaciones",
            "interceptions": "intercepciones",
            "possession changes": "cambios de posesión",
            "counter-presses": "contragolpes",
        }
        for eng, esp in replacements.items():
            text = text.replace(eng, esp)
        return text

    def _translate_fr(self, text: str, viewer_type: str) -> str:
        """Simple French translation for common phrases."""
        replacements = {
            "TACTIQ AI": "TACTIQ AI",
            "Match Report": "Rapport de Match",
            "Tactical Match Report": "Rapport Tactique de Match",
            "BUILD-UP ANALYSIS": "ANALYSE DE CONSTRUCTION",
            "ATTACKING ANALYSIS": "ANALYSE D'ATTAQUE",
            "DEFENSIVE ANALYSIS": "ANALYSE DEFENSIVE",
            "TRANSITION ANALYSIS": "ANALYSE DE TRANSITION",
            "PLAYER FOCUS": "FOCUS JOUEUR",
            "MODEL CLASSIFICATION": "CLASSIFICATION PAR LE MODÈLE",
            "is building attacks": "construit des attaques",
            "pass accuracy": "précision des passes",
            "shots": "tirs",
            "on target": "cadrés",
            "goals": "buts",
            "pressing actions": "actions de pression",
            "tackles": "tacles",
            "recoveries": "récupérations",
            "interceptions": "interceptions",
            "possession changes": "changements de possession",
        }
        for eng, fr in replacements.items():
            text = text.replace(eng, fr)
        return text

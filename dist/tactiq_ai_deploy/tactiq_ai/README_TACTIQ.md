# TACTIQ AI — Phase 2: Complete Visual Football Intelligence Platform

**TACTIQ AI** is a synthetic-football tactical intelligence platform built around an
existing custom **TensorFlow/Keras** tactical-pattern classifier.

## Key Deliverables
1. Working visual TACTIQ AI application
2. Existing Keras model integrated (not retrained)
3. Complete Colab execution instructions
4. Downloadable source-code ZIP
5. Model integration & preprocessing documentation
6. Sample synthetic match events & JSON overlays
7. Automated test results
8. Hackathon compliance matrix
9. Judge-ready demo script
10. Clear list of implemented features, limitations, future extensions

## Quick Start (Colab)

```python
# 1. Install
!pip install -q tensorflow==2.21.0 gradio==6.30.0 plotly==7.1.0 matplotlib==3.11.2 pandas==3.1.0

# 2. Download model & inference files
!wget -O /content/tactiq_football_intelligence.keras "https://your-model-url.com/tactiq_football_intelligence.keras"
!wget -O /content/tactiq_inference.py "https://your-inference-url.com/tactiq_inference.py"
!wget -O /content/tactiq_label_mapping.json "https://your-labels-url.com/tactiq_label_mapping.json"
!wget -O /content/tactiq_feature_config.json "https://your-config-url.com/tactiq_feature_config.json"

# 3. Run
from tactiq_inference import load_predictor
predictor = load_predictor()
events = [...  # list of up to 14 events
    dict(type="pass", x_start=40, y_start=55, x_end=33, y_end=60, duration_s=1.8),
]
result = predictor.predict(events)
print(result["predicted_class"], result["model_confidence"])
```

## Features Implemented

- ✅ Existing `.keras` model loaded and integrated (input: 14×40, output: 7 softmax)
- ✅ Rich event → 9-type model adapter (receptions, clearances, cards mapped)
- ✅ Attacking-direction handling (away team flipped so they attack toward x=100)
- ✅ Coordinate rescaling (pitch metres → 0–100 attacking frame)
- ✅ Synthetic event generation (5 tactical identities)
- ✅ Event validation, ingestion, incremental processing
- ✅ Build-up, attack, defence, transition, player analytics
- ✅ Agent orchestrator (8 agents) + evidence validator + personalization + overlays
- ✅ Narrative generation (multilingual fallbacks: ES/FR/EN)
- ✅ Gradio + Plotly visual dashboard (10 sections)
- ✅ Machine-readable overlay JSON export
- ✅ Automated tests (pytest)

## Model Output Labelling

All model outputs are labelled as **synthetic** — they show what a sequence resembles
under the synthetic training distribution, NOT real-world tactical facts.

- Low confidence (< 0.60 default) → `other_or_uncertain` / "unclassified"
- Confidence threshold NOT validated — documented as model author's default

## Limitations

- Model trained on synthetic data only (labels from documented rules)
- Not tested on real match data
- No retraining performed (as instructed)

## Future Extensions

- Real event data (StatsBomb, tracking data)
- Video auto-eventing integration
- Actual speed/distance metrics from tracking data

## File Structure

```
tactiq_ai/
├── config.py
├── schemas.py
├── data/
│   ├── synthetic_generator.py
│   ├── event_loader.py
│   └── validation.py
├── features/
│   ├── inference.py
│   └── adapter.py
├── analytics/
│   ├── buildup.py
│   ├── attack.py
│   ├── defence.py
│   ├── transitions.py
│   └── player_analysis.py
├── agents/
│   ├── event_interpreter.py
│   ├── tactical_analyst.py
│   ├── keras_inference_agent.py
│   ├── player_analyst.py
│   ├── narrative_generator.py
│   ├── evidence_validator.py
│   ├── personalization.py
│   ├── overlay_formatter.py
│   └── orchestrator.py
├── visualization/
│   ├── pitch.py
│   ├── charts.py
│   └── timeline.py
├── app.py
└── tests/
    └── test_core.py
```

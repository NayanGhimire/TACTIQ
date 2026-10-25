# TACTIQ AI — Phase 3: Tactical Visualisation & Product Integration

**TACTIQ AI** is a visual-first football tactical intelligence platform that analyzes how teams **build up, attack, press, defend, and transition** during a match.

It integrates deterministic tactical analytics, deep learning inference with an existing **TensorFlow/Keras 1D-CNN+LSTM** model, an 8-agent evidence validation pipeline, and a visual Gradio dashboard with interactive Plotly tactical pitch visualizations.

---

## 🚀 Key Phase 3 Features

### 1. Interactive Tactical Pitch (`pitch.py`)
- **Directional Pass Arrows**: Real vector arrows showing trajectory direction, pass distance, and success status.
- **Progressive Passes Highlighted**: Dynamic progression detection (defensive half forward >= 15m, attacking half forward >= 10m, or box entries) rendered in glowing tactical gold.
- **Dedicated Glyph Layers**:
  - **Carries**: Emerald dashed lines with square markers.
  - **Crosses**: Distinct trajectories into the penalty area.
  - **Shots**: Color-coded by outcome (Gold star for Goals, Orange triangle for On Target, Muted X for Off Target).
  - **Defensive Actions**: Pressures (open purple circles), Tackles (red X), Recoveries (teal star-triangles), and Turnovers (rust bowtie markers).
- **Pitch Markings**: Authentic grass cut stripes, penalty boxes, 6-yard boxes, penalty spots, center circle, corner arcs, and goal outlines.
- **Tactical Overlays**: Attacking/Middle/Defensive Third boundaries and 5-channel lanes (Left Wing, Left Halfspace, Central, Right Halfspace, Right Wing).
- **Interactive Pitch Filtering**: Filter instantly by Team (Home/Away), Event Category, Player, Match Period, and Toggle Direction Normalization.
- **Clickable Events**: Custom data attached to all points with rich hover cards (Event ID, Player, Team, Action, Outcome, Zone, Coordinates).
- **No Hallucinated Tracking Data**: Renders observed event trajectories only; never invents tracking data.

### 2. Tactical Pattern Intelligence (`charts.py`)
Dedicated visual modules deriving from match data:
1. **Build-Up**: Passing networks (player node centroids and edge pass volumes) + defensive-third to attacking-third progression stages.
2. **Attack**: Shot map with conversion metrics + 5-channel progression distribution (Wing vs Central).
3. **Defence**: Pitch zone distribution of pressures, tackles, recoveries, and tackle win rates.
4. **Transitions**: Rolling match possession momentum timeline + counter-press counts and turnover danger.
5. **Player Focus**: Individual player tactical radar attributes (Passing, Accuracy, Carries, Pressing, Tackles, Recoveries, Shots) and squad comparison.

### 3. Evidence-Backed Insights (`orchestrator.py` & `app.py`)
Every tactical insight connects to concrete event evidence:
- **Observation**: What occurred factually in the match.
- **Tactical Rationale**: Why the pattern matters strategically.
- **Evidence Provenance**: Explicitly tagged as `[Calculated Fact]`, `[Model Prediction]`, or `[Domain Interpretation]`.
- **Supporting Events**: Links to exact event IDs (e.g. `syn_012, syn_014`). Selecting an insight highlights those exact events on the pitch and timeline.
- **Uncertainty & Limitations**: Sample sizes, confidence scores, and synthetic distribution notes.

### 4. Polished Replay Experience & Modes
- **Match Replay Controls**: Step Forward (+5 / +1), Full Match playback, Reset, and scrubbable Event Slider.
- **Live Match Clock & Scoreboard**: Updates synchronously with replay state.
- **Live Event Stream Feed**: Scrollable tabular feed with pressure and turnover flags.
- **Audience Persona Switcher**:
  - **Casual Fan Mode**: High-level, plain-English summaries and clean visual takeaways.
  - **Tactical Analyst Mode**: In-depth metrics, zone frequencies, model softmax distribution, and uncertainty flags.
- **Clear Synthetic vs Real Provenance**: Continuous badges stating dataset origin and model scope.

---

## 🧪 Automated Test Suite (100% Passing)

Run all 53 automated unit and integration tests:

```bash
python -m pytest tactiq_ai/tests/ -v
```

Test coverage includes:
- Coordinate transformations, rescaling, and attacking direction flips.
- Event schemas, validation bounds, chronological ordering, and timeline integrity.
- Incremental `MatchProcessor` replay engine (start, pause, resume, step, replay to timestamp).
- Tactical analytics (build-up segments, attack channels, defensive actions, transitions, player radar).
- 8-agent orchestrator pipeline, evidence validator, personalization, and multilingual generator (EN, ES, FR).
- Plotly pitch rendering, directional arrows, progressive pass metrics, shot maps, and timeline inspection.
- Complete Gradio dashboard assembly and callback registration.

---

## 💻 Quick Start & Google Colab Instructions

### Running Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch interactive tactical dashboard
python -m tactiq_ai.app
```
Access the dashboard at `http://localhost:7860`.

### Running in Google Colab

Run this snippet in a Colab notebook cell:

```python
# 1. Install dependencies
!pip install -q tensorflow gradio plotly matplotlib pandas numpy

# 2. Launch TACTIQ AI with public Gradio share link
import os
os.chdir("/content/TACTIQ")  # Adjust path to your uploaded workspace

from tactiq_ai.app import build_gradio_app
app = build_gradio_app()
app.launch(share=True)
```

---

## 📦 Deployment & Downloadable Archive

Create a deployable standalone ZIP archive of the complete platform:

```bash
python deploy.py --zip
```
The output is created in `dist/tactiq_ai_<timestamp>.zip`.

---

## ⚖️ Data Provenance & Ethical Disclosure

- **Simulation Engine**: Event sequences in default scenarios are generated using TACTIQ Match Generator v2.0 with contrasting tactical identities (Possession Build-Up, Counter-Attack, High Press, Compact Defence, Direct Play).
- **Keras Tactical Classifier**: The included 1D-CNN+LSTM (`tactiq_football_intelligence.keras`) classifies sequential patterns under the synthetic training distribution. Softmax confidence reflects likeness to training archetypes, not real-match outcome predictions.
- **No Score Prediction Hallucination**: TACTIQ AI prioritizes tactical mechanics (how teams build up, press, and transition) rather than predicting final scorelines.

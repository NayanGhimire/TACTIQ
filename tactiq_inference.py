"""tactiq_inference.py  -  TACTIQ AI football sequence classifier: features + inference.

Self-contained: needs only numpy (+ tensorflow/keras to load the model). It is the
single source of truth for feature engineering, used by BOTH training and inference.

Coordinate convention (attacking-team frame): x runs 0 (own goal line) -> 100 (opponent
goal line), y runs 0 -> 100 across the width. Pitch assumed 105 m x 68 m.
"""
import json, math
from pathlib import Path
import numpy as np

MODULE_VERSION = "1.0"
PITCH_L, PITCH_W = 105.0, 68.0
T_MAX = 14            # model window: max events per sequence (post-padded with zeros)
MIN_EVENTS = 3

ETYPES = ["pass", "carry", "cross", "shot", "pressure", "tackle", "recovery", "ball_loss", "opp_pass"]
ETYPE_TO_ID = {n: i for i, n in enumerate(ETYPES)}
RAW_COLS = ["etype", "x0", "y0", "x1", "y1", "t_start", "dur", "completed",
            "under_pressure", "turnover", "poss_after", "prev_poss"]
RC = {n: i for i, n in enumerate(RAW_COLS)}
N_RAW = len(RAW_COLS)

CHANNELS = ["left_wide", "left_halfspace", "central", "right_halfspace", "right_wide"]
CHANNEL_EDGES = [20.0, 38.0, 62.0, 80.0]          # on the 0-100 y axis
ZONES = ["defensive_third", "middle_third", "attacking_third"]
ZONE_EDGES = [33.3333, 66.6667]                   # on the 0-100 x axis

FEATURE_NAMES = (
    [f"type_{n}" for n in ETYPES]
    + ["x_start", "y_start", "x_end", "y_end", "forward_m", "lateral_m", "distance_m"]
    + ["completed", "under_pressure", "turnover", "possession_after", "possession_before"]
    + ["is_cross", "is_shot", "is_tackle", "is_recovery"]
    + [f"channel_{c}" for c in CHANNELS]
    + [f"zone_{z}" for z in ZONES]
    + ["event_duration_s", "elapsed_s", "own_possession_time_s", "cum_passes",
       "cum_net_forward_m", "progression_speed_mps"]
    + ["valid_step"]
)
IX = {n: i for i, n in enumerate(FEATURE_NAMES)}
N_FEATURES = len(FEATURE_NAMES)
CONTINUOUS_FEATURES = ["x_start", "y_start", "x_end", "y_end", "forward_m", "lateral_m", "distance_m",
                       "event_duration_s", "elapsed_s", "own_possession_time_s", "cum_passes",
                       "cum_net_forward_m", "progression_speed_mps"]
CONTINUOUS_IDX = [IX[n] for n in CONTINUOUS_FEATURES]


# ----------------------------------------------------------------------------------------
# Feature engineering (CAUSAL: row t only uses events 0..t, never future events)
# ----------------------------------------------------------------------------------------
def raw_to_features(raw, n):
    """raw: (T_MAX, N_RAW) event table, n: number of real events. Returns UNSCALED (T_MAX, N_FEATURES)."""
    raw = np.asarray(raw, dtype=np.float64)
    n = int(n)
    out = np.zeros((T_MAX, N_FEATURES), dtype=np.float32)
    t_first = raw[0, RC["t_start"]]
    x_first = raw[0, RC["x0"]]
    own_time, cum_pass = 0.0, 0
    for t in range(n):
        r = raw[t]
        et = int(r[RC["etype"]])
        x0, y0, x1, y1 = r[RC["x0"]], r[RC["y0"]], r[RC["x1"]], r[RC["y1"]]
        dur = r[RC["dur"]]
        fwd = (x1 - x0) * PITCH_L / 100.0
        lat = (y1 - y0) * PITCH_W / 100.0
        if r[RC["prev_poss"]] == 1:
            own_time += dur
        if ETYPES[et] in ("pass", "cross"):
            cum_pass += 1
        elapsed = r[RC["t_start"]] + dur - t_first
        net_fwd = (x1 - x_first) * PITCH_L / 100.0
        row = out[t]
        row[IX["type_" + ETYPES[et]]] = 1.0
        row[IX["x_start"]], row[IX["y_start"]] = x0 / 100.0, y0 / 100.0
        row[IX["x_end"]], row[IX["y_end"]] = x1 / 100.0, y1 / 100.0
        row[IX["forward_m"]], row[IX["lateral_m"]] = fwd, lat
        row[IX["distance_m"]] = math.hypot(fwd, lat)
        row[IX["completed"]] = r[RC["completed"]]
        row[IX["under_pressure"]] = r[RC["under_pressure"]]
        row[IX["turnover"]] = r[RC["turnover"]]
        row[IX["possession_after"]] = r[RC["poss_after"]]
        row[IX["possession_before"]] = r[RC["prev_poss"]]
        row[IX["is_cross"]] = 1.0 if ETYPES[et] == "cross" else 0.0
        row[IX["is_shot"]] = 1.0 if ETYPES[et] == "shot" else 0.0
        row[IX["is_tackle"]] = 1.0 if ETYPES[et] == "tackle" else 0.0
        row[IX["is_recovery"]] = 1.0 if ETYPES[et] == "recovery" else 0.0
        row[IX["channel_" + CHANNELS[int(np.searchsorted(CHANNEL_EDGES, y1, side="right"))]]] = 1.0
        row[IX["zone_" + ZONES[int(np.searchsorted(ZONE_EDGES, x1, side="right"))]]] = 1.0
        row[IX["event_duration_s"]] = dur
        row[IX["elapsed_s"]] = elapsed
        row[IX["own_possession_time_s"]] = own_time
        row[IX["cum_passes"]] = cum_pass
        row[IX["cum_net_forward_m"]] = net_fwd
        row[IX["progression_speed_mps"]] = net_fwd / max(elapsed, 1.0)
        row[IX["valid_step"]] = 1.0
    return out


def apply_scaler(feats, n, mean, std, clip=8.0):
    """Standardise continuous columns of the first n (real) steps. Padded steps stay all-zero
    (so a Masking(0.0) layer ignores them). Binary / one-hot columns are left unchanged."""
    out = np.array(feats, dtype=np.float32, copy=True)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    block = out[:n, CONTINUOUS_IDX]
    out[:n, CONTINUOUS_IDX] = np.clip((block - mean) / std, -clip, clip)
    return out


# ----------------------------------------------------------------------------------------
# Input validation: list-of-dict events  ->  raw event table
# ----------------------------------------------------------------------------------------
REQUIRED_KEYS = ["type", "x_start", "y_start", "x_end", "y_end"]
# Optional keys: duration_s, completed, under_pressure, turnover, possession_before,
#                possession_after, t_start_s  (sensible per-type defaults are filled in)


def _num(ev, key, i, lo=None, hi=None):
    try:
        v = float(ev[key])
    except (TypeError, ValueError):
        raise ValueError(f"event {i}: '{key}' must be a number, got {ev[key]!r}")
    if not math.isfinite(v):
        raise ValueError(f"event {i}: '{key}' must be finite")
    if lo is not None and not (lo <= v <= hi):
        raise ValueError(f"event {i}: '{key}'={v} outside allowed range [{lo}, {hi}]")
    return v


def _flag(ev, key, i, default):
    if key not in ev or ev[key] is None:
        return int(default)
    v = ev[key]
    if isinstance(v, (bool, np.bool_)) or v in (0, 1):
        return int(bool(v))
    raise ValueError(f"event {i}: '{key}' must be a bool/0/1, got {v!r}")


def validate_sequence(event_sequence):
    """Validate a list of event dicts and convert to (raw (T_MAX, N_RAW), n)."""
    if not isinstance(event_sequence, (list, tuple)):
        raise ValueError("event_sequence must be a list of event dicts")
    n = len(event_sequence)
    if n < MIN_EVENTS:
        raise ValueError(f"need at least {MIN_EVENTS} events, got {n}")
    if n > T_MAX:
        raise ValueError(f"at most {T_MAX} events per sequence (model window); got {n}. "
                         "Slice a window of the most relevant events first.")
    raw = np.zeros((T_MAX, N_RAW), dtype=np.float32)
    raw[:, RC["etype"]] = -1
    prev_after, t_cursor, prev_t = None, 0.0, -1e9
    for i, ev in enumerate(event_sequence):
        if not isinstance(ev, dict):
            raise ValueError(f"event {i}: must be a dict")
        for k in REQUIRED_KEYS:
            if k not in ev:
                raise ValueError(f"event {i}: missing required key '{k}'")
        if ev["type"] not in ETYPE_TO_ID:
            raise ValueError(f"event {i}: unknown type {ev['type']!r}; allowed: {ETYPES}")
        et = ev["type"]
        x0, y0 = _num(ev, "x_start", i, 0, 100), _num(ev, "y_start", i, 0, 100)
        x1, y1 = _num(ev, "x_end", i, 0, 100), _num(ev, "y_end", i, 0, 100)
        dur = _num(ev, "duration_s", i, 0, 60) if "duration_s" in ev else 1.5

        if et in ("pass", "cross"):
            completed = _flag(ev, "completed", i, 1)
            prev_d, after_d, turn_d, up_d = 1, completed, 1 - completed, 0
        elif et == "carry":
            completed = _flag(ev, "completed", i, 1)
            prev_d, after_d, turn_d, up_d = 1, 1, 0, 0
        elif et == "shot":
            completed = _flag(ev, "completed", i, 1)          # 1 = on target
            prev_d, after_d, turn_d, up_d = 1, 0, 0, 0
        elif et == "pressure":
            completed, prev_d, after_d, turn_d, up_d = 1, 0, 0, 0, 1
        elif et == "tackle":
            completed = _flag(ev, "completed", i, 1)          # 1 = won the ball
            prev_d, after_d, turn_d, up_d = 0, completed, completed, 0
        elif et == "recovery":
            completed, prev_d, after_d, turn_d, up_d = 1, 0, 1, 1, 0
        elif et == "ball_loss":
            completed, prev_d, after_d, turn_d, up_d = 0, 1, 0, 1, 0
        else:  # opp_pass
            completed = _flag(ev, "completed", i, 1)
            prev_d, after_d, turn_d, up_d = 0, 0, 0, 0

        prev = _flag(ev, "possession_before", i, prev_d if prev_after is None else prev_after)
        after = _flag(ev, "possession_after", i, after_d)
        turn = _flag(ev, "turnover", i, turn_d)
        up = _flag(ev, "under_pressure", i, up_d)
        if prev_after is not None and prev != prev_after:
            raise ValueError(f"event {i}: possession_before={prev} contradicts previous event's "
                             f"possession_after={prev_after}")
        t_start = _num(ev, "t_start_s", i, 0, 1e5) if "t_start_s" in ev else t_cursor
        if t_start < prev_t:
            raise ValueError(f"event {i}: events must be in chronological order")
        raw[i] = [ETYPE_TO_ID[et], x0, y0, x1, y1, t_start, dur, completed, up, turn, after, prev]
        prev_after, prev_t, t_cursor = after, t_start, t_start + dur + 0.7
    return raw, n


def raw_to_events(raw, n):
    """Inverse of validate_sequence: raw event table -> list of fully-specified event dicts."""
    events = []
    for r in np.asarray(raw)[:n]:
        events.append(dict(
            type=ETYPES[int(r[RC["etype"]])],
            x_start=float(r[RC["x0"]]), y_start=float(r[RC["y0"]]),
            x_end=float(r[RC["x1"]]), y_end=float(r[RC["y1"]]),
            t_start_s=float(r[RC["t_start"]]), duration_s=float(r[RC["dur"]]),
            completed=bool(r[RC["completed"]]), under_pressure=bool(r[RC["under_pressure"]]),
            turnover=bool(r[RC["turnover"]]), possession_before=int(r[RC["prev_poss"]]),
            possession_after=int(r[RC["poss_after"]])))
    return events


# ----------------------------------------------------------------------------------------
# Predictor
# ----------------------------------------------------------------------------------------
CONFIDENCE_NOTE = ("model_confidence is the softmax probability of a network trained on SYNTHETIC, rule-labelled "
                   "sequences. It measures resemblance to those synthetic patterns - NOT the real-world "
                   "probability that a team actually used this tactic.")


class TactiqPredictor:
    def __init__(self, model_path, label_path, feature_config_path, low_confidence_threshold=0.60):
        import tensorflow as tf  # imported lazily so feature code works without TF
        self.model = tf.keras.models.load_model(model_path)
        lm = json.loads(Path(label_path).read_text())
        fc = json.loads(Path(feature_config_path).read_text())
        self.classes = lm["classes"]
        self.mean = np.array(fc["scaler_mean"], dtype=np.float32)
        self.std = np.array(fc["scaler_std"], dtype=np.float32)
        self.clip = float(fc["scaler_clip"])
        self.tmin = np.array(fc["train_min"], dtype=np.float32)
        self.tmax = np.array(fc["train_max"], dtype=np.float32)
        assert fc["feature_names"] == FEATURE_NAMES and fc["t_max"] == T_MAX, \
            "feature config does not match tactiq_inference.py version"
        self.low_conf = low_confidence_threshold

    def prepare(self, event_sequence):
        raw, n = validate_sequence(event_sequence)
        feats = raw_to_features(raw, n)
        return feats, apply_scaler(feats, n, self.mean, self.std, self.clip), n

    def predict(self, event_sequence):
        feats, x, n = self.prepare(event_sequence)
        probs = np.asarray(self.model(x[None], training=False))[0]
        order = np.argsort(-probs)
        flags = []
        if probs[order[0]] < self.low_conf:
            flags.append(f"low_confidence (<{self.low_conf:.2f}): treat as uncertain")
        if n < 5:
            flags.append("very_short_sequence: fewer events than most training sequences")
        cont = feats[:n, CONTINUOUS_IDX]
        span = np.maximum(self.tmax - self.tmin, 1e-6)
        bad = (cont < self.tmin - 0.1 * span) | (cont > self.tmax + 0.1 * span)
        if bad.any():
            names = sorted({CONTINUOUS_FEATURES[j] for j in np.where(bad.any(axis=0))[0]})
            flags.append("outside_training_range: " + ", ".join(names))
        return {
            "predicted_class": self.classes[int(order[0])],
            "class_probabilities": {self.classes[int(i)]: round(float(probs[i]), 4) for i in order},
            "model_confidence": round(float(probs[order[0]]), 4),
            "runner_up": self.classes[int(order[1])],
            "margin_over_runner_up": round(float(probs[order[0]] - probs[order[1]]), 4),
            "n_events": n,
            "flags": flags,
            "confidence_note": CONFIDENCE_NOTE,
        }


_DEFAULT = None


def load_predictor(model_path=None, label_path=None, feature_config_path=None):
    """Load once; paths default to the files sitting next to this module."""
    global _DEFAULT
    here = Path(__file__).resolve().parent
    _DEFAULT = TactiqPredictor(model_path or here / "tactiq_football_intelligence.keras",
                               label_path or here / "tactiq_label_mapping.json",
                               feature_config_path or here / "tactiq_feature_config.json")
    return _DEFAULT


def predict_tactical_pattern(event_sequence, predictor=None):
    """Classify ONE sequence of events (list of dicts, see validate_sequence).

    Returns predicted_class, class_probabilities, model_confidence (+ notes/flags).
    This is a *classifier only*: statistics and narrative belong to separate layers.
    """
    predictor = predictor or _DEFAULT or load_predictor()
    return predictor.predict(event_sequence)

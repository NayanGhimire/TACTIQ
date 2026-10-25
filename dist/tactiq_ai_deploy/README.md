# TACTIQ AI - football sequence classifier (v1.0)

Small GRU network that classifies a window of up to 14 football events into one of
7 tactical-pattern classes: controlled_buildup, direct_progression, wide_attack, central_progression, high_press_turnover, counterattack, other_or_uncertain.

## Files
- tactiq_football_intelligence.keras  trained Keras model (input: 14 x 40, output: 7 softmax probabilities)
- tactiq_label_mapping.json           class names, ids and the rule definition of every class
- tactiq_feature_config.json          feature names, scaler (fit on TRAIN only), event schema, training config
- tactiq_test_metrics.json            held-out test metrics, baselines, confusion matrix, caveats
- tactiq_inference.py                 standalone feature engineering + validation + predictor (numpy + tensorflow only)
- tactiq_training_history.json, tactiq_plots/   training curves, confusion matrix, dataset report

## Use
```python
import tactiq_inference as tq          # keep the 4 files above in the same folder
result = tq.predict_tactical_pattern([
    dict(type="opp_pass", x_start=40, y_start=55, x_end=33, y_end=60, duration_s=1.8),
    dict(type="tackle",   x_start=33, y_start=60, x_end=33, y_end=60, duration_s=0.9, completed=True),
    dict(type="carry",    x_start=33, y_start=60, x_end=55, y_end=55, duration_s=2.5),
    dict(type="pass",     x_start=55, y_start=55, x_end=85, y_end=50, duration_s=1.6),
])
print(result["predicted_class"], result["model_confidence"])
```
Coordinates are in the ATTACKING team's frame (x: 0 own goal line -> 100 opponent goal line, y: 0..100).

## Read this before relying on it
- Trained ONLY on synthetic sequences whose labels are outputs of explicit numeric rules (see label mapping).
  It reproduces those rules; it has not learned real-world tactics and has not been tested on real match data.
- `model_confidence` is a softmax probability under that synthetic distribution - NOT the real-world probability
  that a team used a tactic. Low-confidence and out-of-range inputs are flagged in `flags`.
- Held-out test accuracy 0.964, macro-F1 0.966 (synthetic data, unseen matches, same
  generator). A simple order-blind logistic-regression baseline reached 0.956.

## Integration plan
This model only classifies event sequences. Measurable statistics belong in a separate analytical layer and
explanations in a separate narrative layer that cites those statistics; the classifier output should be one input
among several, never the report itself.

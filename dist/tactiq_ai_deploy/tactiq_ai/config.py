"""TACTIQ AI - Configuration module.

Contains project configuration constants, paths, and settings.
"""

from pathlib import Path

# Project root - directory containing tactiq_ai package
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Model artifact paths (KEEP originals unchanged)
MODEL_PATH = PROJECT_ROOT / "tactiq_football_intelligence.keras"
LABEL_MAPPING_PATH = PROJECT_ROOT / "tactiq_label_mapping.json"
FEATURE_CONFIG_PATH = PROJECT_ROOT / "tactiq_feature_config.json"
TEST_METRICS_PATH = PROJECT_ROOT / "tactiq_test_metrics.json"

# Tactiq inference module path
TACTIQ_INFERENCE_PATH = PROJECT_ROOT / "tactiq_inference.py"

# Data paths
DATA_DIR = PROJECT_ROOT / "tactiq_ai" / "data"
SYNTHETIC_MATCHES_DIR = DATA_DIR / "synthetic_matches"
SAMPLE_EVENTS_PATH = DATA_DIR / "sample_events.json"
SAMPLE_OVERLAYS_PATH = DATA_DIR / "sample_overlays.json"

# Application settings
APP_NAME = "TACTIQ AI"
APP_VERSION = "2.0"

# Confidence settings
# NOTE: 0.60 is a default picked by the model author, NOT a validated value.
# The predictor flags low-confidence results below this threshold.
LOW_CONFIDENCE_THRESHOLD = 0.60

# Pitch settings
PITCH_LENGTH_M = 105.0
PITCH_WIDTH_M = 68.0

# Data source types
SOURCE_SYNTHETIC = "synthetic"
SOURCE_REAL = "real"
SOURCE_UPLOADED = "uploaded"

# Supported languages for multilingual output
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "it": "Italiano",
}

# Viewer types
VIEWER_TYPES = [
    "casual_fan",
    "tactical_analyst",
    "player_focus",
    "custom",
]

# Interest areas for custom mode
INTEREST_AREAS = [
    "pressing",
    "progressive_passing",
    "chance_creation",
    "defensive_recoveries",
    "build_up",
    "transitions",
]

# Export settings
OVERLAYS_DIR = DATA_DIR / "overlays"
REPORTS_DIR = DATA_DIR / "reports"

# Ensure directories exist
for _d in [DATA_DIR, SYNTHETIC_MATCHES_DIR, OVERLAYS_DIR, REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

from pathlib import Path

root = Path(r"C:\Users\Micro\OneDrive\Desktop\TACTIQ")

pats = ["run_model_inference", "analyze_player", "create_pitch_figure", "add_pass_trajectories", "add_event_markers", "create_match_timeline", "generate_match_flow", "add_pass_arrows"]
for pat in pats:
    hits = [str(p) for p in root.rglob("*.py") if pat in p.read_text(errors="ignore")]
    print(pat, "->", hits)

for name in ["tactiq_ai/features/inference.py", "tactiq_ai/features/adapter.py"]:
    p = root / name
    if p.exists():
        txt = p.read_text(errors="ignore")
        for i, line in enumerate(txt.splitlines(), 1):
            if "run_model_inference" in line or "def run_model" in line or "class InferenceAgent" in line or "def classify_sequence" in line:
                print(name, i, line)

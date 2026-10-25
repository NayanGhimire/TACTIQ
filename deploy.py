"""TACTIQ AI - Deployment packaging script.

Creates a deployable ZIP containing:
- tactiq_ai/ Python package
- Keras model (.keras)
- Config files (feature config, label mapping)
- Requirements
- Documentation (README, Compliance Matrix)
"""

from __future__ import annotations

import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
PACKAGE_NAME = "tactiq_ai"
MODEL_FILE = "tactiq_football_intelligence.keras"
REQUIREMENTS = "requirements.txt"
BUILD_DIR = PROJECT_ROOT / "build"
DIST_DIR = PROJECT_ROOT / "dist"


def _collect_files() -> list[tuple[Path, str]]:
    """Collect files to include in the deployment package.

    Returns list of (source_path, archive_path) tuples.
    """
    files = []

    # tactiq_ai package (all .py files)
    pkg_dir = PROJECT_ROOT / PACKAGE_NAME
    for root, dirs, fnames in os.walk(pkg_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fname in fnames:
            if fname.endswith(".py"):
                src = Path(root) / fname
                arc = str(src.relative_to(PROJECT_ROOT))
                files.append((src, arc))

    # Model file
    model_path = PROJECT_ROOT / MODEL_FILE
    if model_path.exists():
        files.append((model_path, MODEL_FILE))

    # Config files
    for cfg in ["tactiq_feature_config.json", "tactiq_label_mapping.json",
                "tactiq_test_metrics.json", "tactiq_training_history.json"]:
        cp = PROJECT_ROOT / cfg
        if cp.exists():
            files.append((cp, cfg))

    # Requirements
    req_path = PROJECT_ROOT / REQUIREMENTS
    if req_path.exists():
        files.append((req_path, REQUIREMENTS))

    # Documentation
    for doc in ["README.md", "tactiq_ai/README_TACTIQ.md",
                "tactiq_ai/Compliance_Matrix.md"]:
        dp = PROJECT_ROOT / doc
        if dp.exists():
            files.append((dp, doc))

    # Plot assets
    plots_dir = PROJECT_ROOT / "tactiq_plots"
    if plots_dir.exists():
        for root, dirs, fnames in os.walk(plots_dir):
            for fname in fnames:
                src = Path(root) / fname
                arc = str(src.relative_to(PROJECT_ROOT))
                files.append((src, arc))

    return files


def build_zip(output_path: Path | None = None) -> Path:
    """Build the deployment ZIP archive.

    Args:
        output_path: Where to write the ZIP. Defaults to dist/tactiq_ai_<date>.zip

    Returns:
        Path to the created ZIP file.
    """
    DIST_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DIST_DIR / f"tactiq_ai_{timestamp}.zip"

    files = _collect_files()

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for src, arc in files:
            zf.write(src, arc)

    print(f"Created deployment package: {output_path}")
    print(f"  Files included: {len(files)}")
    print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return output_path


def build_directory(output_dir: Path | None = None) -> Path:
    """Build an uncompressed deployment directory.

    Args:
        output_dir: Target directory. Defaults to dist/tactiq_ai_deploy/

    Returns:
        Path to the created directory.
    """
    if output_dir is None:
        output_dir = DIST_DIR / "tactiq_ai_deploy"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _collect_files()

    for src, arc in files:
        dest = output_dir / arc
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)

    print(f"Created deployment directory: {output_dir}")
    print(f"  Files included: {len(files)}")

    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Package TACTIQ AI for deployment")
    parser.add_argument("--zip", action="store_true",
                        help="Create a ZIP archive (default: directory)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for the package")
    args = parser.parse_args()

    if args.zip:
        build_zip(Path(args.output) if args.output else None)
    else:
        build_directory(Path(args.output) if args.output else None)
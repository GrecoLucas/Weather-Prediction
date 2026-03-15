"""
run_pipeline.py
---------------
Convenience script: runs the full forecasting pipeline end-to-end.

    python run_pipeline.py

Steps:
  1. 01_preprocess.py          → data/processed.parquet
  2. 02_feature_engineering.py → data/features.parquet
  3. 03_walk_forward.py        → results/walk_forward_results.csv
  4. 05_evaluate.py            → results/plots/ + score printed
"""

import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    "01_preprocess.py",
    "02_feature_engineering.py",
    "03_walk_forward.py",
    "05_evaluate.py",
]


def run(script: str):
    path = os.path.join(BASE_DIR, script)
    print(f"\n{'='*60}")
    print(f"  Running: {script}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, path], check=True)
    return result.returncode


if __name__ == "__main__":
    for script in SCRIPTS:
        run(script)
    print("\n Pipeline complete.")
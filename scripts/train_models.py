"""
Run this once (or whenever the synthetic dataset / models need
regenerating): builds the synthetic dataset via the physics simulator and
trains + saves the three ML models used by the dashboard.

Usage:
    python scripts/train_models.py
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from digital_twin.data_gen import generate_dataset
from digital_twin.ml_models import train_all, save_models


def main():
    print("Generating synthetic dataset from physics simulator...")
    df = generate_dataset(n_cycles=300)
    print(f"  {len(df)} rows across {df['cycle_id'].nunique()} cycles")

    print("Training models...")
    models = train_all(df)
    for k, v in models.metrics.items():
        print(f"  {k}: {v:.4f}")

    print("Saving models to saved_models/ ...")
    save_models(models)
    print("Done.")


if __name__ == "__main__":
    main()

"""
data_gen.py — Generates a synthetic multi-cycle training dataset by running
the physics simulator across a range of CSS/SRP settings. This stands in
for real Oil India historical data during the prototype stage (see README).
"""

import numpy as np
import pandas as pd

from .simulator import CycleConfig, simulate_cycle

RNG = np.random.default_rng(42)


def generate_dataset(n_cycles: int = 300) -> pd.DataFrame:
    """Run n_cycles random CSS/SRP configurations through the simulator and
    return one combined daily-record dataframe, with light sensor noise
    added so ML models don't just learn the exact deterministic physics."""
    frames = []
    for cycle_id in range(n_cycles):
        cfg = CycleConfig(
            steam_volume_bbl=float(RNG.uniform(1500, 6000)),
            soak_days=int(RNG.integers(2, 10)),
            produce_days=int(RNG.integers(20, 60)),
            spm=float(RNG.uniform(2.0, 7.0)),
            stroke_length_in=float(RNG.uniform(64, 100)),
        )
        df = simulate_cycle(cfg)
        df["cycle_id"] = cycle_id
        df["days_since_steam"] = df["day"]
        df["steam_volume_bbl"] = cfg.steam_volume_bbl

        # sensor noise, representative of real telemetry
        df["temp_c_noisy"] = df["temp_c"] + RNG.normal(0, 0.5, len(df))
        df["oil_rate_bopd_noisy"] = np.clip(
            df["oil_rate_bopd"] + RNG.normal(0, 1.5, len(df)), 0, None
        )

        # synthetic failure label: elevated risk_ratio + noise -> failure event
        failure_prob = np.clip((df["rod_floating_risk_ratio"] - 0.6) * 0.8, 0, 0.9)
        df["failure_event"] = RNG.random(len(df)) < failure_prob

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    data = generate_dataset()
    data.to_csv("synthetic_baghewala_dataset.csv", index=False)
    print(f"Generated {len(data)} rows across {data['cycle_id'].nunique()} cycles.")

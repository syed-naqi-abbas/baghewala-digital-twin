"""
simulator.py — Coupled CSS + SRP daily simulation for one lumped
near-wellbore volume (single well). This is the "physics twin" the ML
models are trained on and validated against.

Scope (see README "What remains" for what is deliberately out of scope):
  - One well, one lumped reservoir volume. No spatial steam-front model.
  - Steam raises near-wellbore temperature; heat decays exponentially
    between cycles (see physics.temperature_decay).
  - Oil mobility ~ 1 / viscosity(T). Production rate is the lesser of a
    temperature-adjusted inflow rate and SRP lift capacity.
  - Rod-floating risk is checked every simulated day using the physics
    criterion in physics.py.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from . import physics


@dataclass
class CycleConfig:
    steam_volume_bbl: float = 4000.0
    soak_days: int = 5
    produce_days: int = 40
    spm: float = 4.0
    stroke_length_in: float = 86.0    # ~ 2.18 m, common API stroke length
    q_max_bopd: float = 60.0          # max reservoir inflow at zero viscosity penalty
    pump_capacity_bopd_per_spm: float = 12.0


def _stroke_length_m(stroke_in: float) -> float:
    return stroke_in * 0.0254


def simulate_cycle(cfg: CycleConfig) -> pd.DataFrame:
    """Simulate one full CSS cycle (soak + produce) day by day."""
    rows = []
    total_days = cfg.soak_days + cfg.produce_days
    stroke_m = _stroke_length_m(cfg.stroke_length_in)

    for day in range(total_days):
        phase = "soak" if day < cfg.soak_days else "produce"
        temp_c = physics.temperature_decay(day, cfg.steam_volume_bbl)
        mu_cp = physics.dynamic_viscosity_cp(temp_c)
        mu_ref = physics.dynamic_viscosity_cp(physics.RESERVOIR_TEMP_BASE_C)
        mobility_ratio = mu_ref / mu_cp  # >1 means oil is more mobile than baseline

        # Inflow: temperature-adjusted productivity index x mobility ratio
        inflow_bopd = cfg.q_max_bopd * min(mobility_ratio, 6.0) ** 0.65 if phase == "produce" else 0.0

        # SRP lift capacity (very simplified: proportional to SPM, degraded
        # by rod-floating risk if present)
        rod = physics.rod_floating_check(temp_c, cfg.spm, stroke_m)
        volumetric_efficiency = 1.0 if rod.risk_ratio < 0.7 else max(0.15, 1.0 - (rod.risk_ratio - 0.7))
        pump_capacity_bopd = cfg.spm * cfg.pump_capacity_bopd_per_spm * volumetric_efficiency

        oil_rate_bopd = min(inflow_bopd, pump_capacity_bopd) if phase == "produce" else 0.0

        loads = physics.beam_pump_loads(cfg.spm, cfg.stroke_length_in, temp_c)

        rows.append(dict(
            day=day,
            phase=phase,
            temp_c=temp_c,
            viscosity_cp=mu_cp,
            mobility_ratio=mobility_ratio,
            inflow_bopd=inflow_bopd,
            oil_rate_bopd=oil_rate_bopd,
            spm=cfg.spm,
            stroke_in=cfg.stroke_length_in,
            rod_floating_risk_ratio=rod.risk_ratio,
            rod_floating=rod.floating,
            pprl_n=loads.pprl_n,
            mprl_n=loads.mprl_n,
        ))

    df = pd.DataFrame(rows)
    return df


def cycle_summary(df: pd.DataFrame, steam_volume_bbl: float) -> dict:
    """Roll a simulated cycle up into headline KPIs, incl. Steam-Oil Ratio."""
    cum_oil_bbl = df["oil_rate_bopd"].sum()
    sor = steam_volume_bbl / cum_oil_bbl if cum_oil_bbl > 0 else np.inf
    floating_days = int(df["rod_floating"].sum())
    max_risk = float(df["rod_floating_risk_ratio"].max())
    return dict(
        cumulative_oil_bbl=cum_oil_bbl,
        steam_oil_ratio=sor,
        floating_days=floating_days,
        max_floating_risk_ratio=max_risk,
        peak_temp_c=float(df["temp_c"].max()),
    )

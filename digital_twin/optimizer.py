"""
optimizer.py — Joint CSS + SRP operating-point optimizer.

This is a bounded grid-search optimizer (deliberately simple and fully
explainable for a hackathon demo) rather than the Bayesian-optimization /
reinforcement-learning approach described as a stretch goal in the
solution architecture (see README "What remains").

Objective:
    maximize   cumulative_oil - lambda_steam * steam_volume
    subject to max_floating_risk_ratio < risk_limit   (hard constraint)

This directly implements the "pump-survivable" framing: the optimizer
never recommends a setting the physics twin flags as exceeding the
rod-floating threshold, regardless of how much oil it would produce.
"""

from dataclasses import dataclass
import itertools
import numpy as np

from .simulator import CycleConfig, simulate_cycle, cycle_summary


@dataclass
class OptimizationResult:
    best_config: CycleConfig
    best_summary: dict
    all_results: list
    rejected_for_risk: int


def optimize_cycle(
    steam_options=(2500, 3500, 4500, 5500),
    soak_options=(2, 3, 5, 7),
    spm_options=(3.0, 4.5, 6.0, 8.0, 10.0),
    stroke_in: float = 86.0,
    produce_days: int = 40,
    risk_limit: float = 0.85,
    lambda_steam: float = 0.01,
) -> OptimizationResult:
    results = []
    rejected = 0

    for steam, soak, spm in itertools.product(steam_options, soak_options, spm_options):
        cfg = CycleConfig(
            steam_volume_bbl=steam,
            soak_days=soak,
            produce_days=produce_days,
            spm=spm,
            stroke_length_in=stroke_in,
        )
        df = simulate_cycle(cfg)
        summary = cycle_summary(df, steam)

        feasible = summary["max_floating_risk_ratio"] < risk_limit
        if not feasible:
            rejected += 1

        score = summary["cumulative_oil_bbl"] - lambda_steam * steam
        results.append(dict(cfg=cfg, summary=summary, score=score, feasible=feasible))

    feasible_results = [r for r in results if r["feasible"]]
    pool = feasible_results if feasible_results else results  # fall back if nothing feasible
    best = max(pool, key=lambda r: r["score"])

    return OptimizationResult(
        best_config=best["cfg"],
        best_summary=best["summary"],
        all_results=results,
        rejected_for_risk=rejected,
    )

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from digital_twin.simulator import CycleConfig, simulate_cycle, cycle_summary
from digital_twin.optimizer import optimize_cycle
from digital_twin.data_gen import generate_dataset


def test_simulate_cycle_shape():
    cfg = CycleConfig(steam_volume_bbl=4000, soak_days=5, produce_days=10)
    df = simulate_cycle(cfg)
    assert len(df) == 15
    expected_cols = {"day", "phase", "temp_c", "viscosity_cp", "oil_rate_bopd", "rod_floating_risk_ratio"}
    assert expected_cols.issubset(df.columns)


def test_no_production_during_soak():
    cfg = CycleConfig(steam_volume_bbl=4000, soak_days=5, produce_days=10)
    df = simulate_cycle(cfg)
    soak_rows = df[df["phase"] == "soak"]
    assert (soak_rows["oil_rate_bopd"] == 0).all()


def test_cycle_summary_keys():
    cfg = CycleConfig(steam_volume_bbl=4000, soak_days=5, produce_days=10)
    df = simulate_cycle(cfg)
    summary = cycle_summary(df, cfg.steam_volume_bbl)
    for key in ["cumulative_oil_bbl", "steam_oil_ratio", "floating_days", "max_floating_risk_ratio"]:
        assert key in summary


def test_optimizer_returns_feasible_when_possible():
    result = optimize_cycle(
        steam_options=(3000,), soak_options=(5,), spm_options=(2.0, 3.0, 6.0),
        risk_limit=0.85, produce_days=15,
    )
    assert result.best_config is not None
    assert "cumulative_oil_bbl" in result.best_summary


def test_generate_dataset_small():
    df = generate_dataset(n_cycles=5)
    assert df["cycle_id"].nunique() == 5
    assert "failure_event" in df.columns

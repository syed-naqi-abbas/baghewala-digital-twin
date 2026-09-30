import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from digital_twin import physics


def test_viscosity_decreases_with_temperature():
    v_cold = physics.dynamic_viscosity_cp(47.0)
    v_hot = physics.dynamic_viscosity_cp(140.0)
    assert v_hot < v_cold, "Viscosity should drop sharply as temperature rises"


def test_viscosity_roughly_matches_calibration_targets():
    v_cold = physics.dynamic_viscosity_cp(47.0)
    v_hot = physics.dynamic_viscosity_cp(140.0)
    assert 3000 < v_cold < 15000
    assert 30 < v_hot < 400


def test_rod_floating_risk_increases_with_spm():
    low = physics.rod_floating_check(temp_c=47.0, spm=2.0, stroke_length_m=2.18)
    high = physics.rod_floating_check(temp_c=47.0, spm=7.0, stroke_length_m=2.18)
    assert high.risk_ratio > low.risk_ratio


def test_rod_floating_risk_drops_at_higher_temperature():
    cold = physics.rod_floating_check(temp_c=47.0, spm=4.0, stroke_length_m=2.18)
    hot = physics.rod_floating_check(temp_c=120.0, spm=4.0, stroke_length_m=2.18)
    assert hot.risk_ratio < cold.risk_ratio, "Lower viscosity at high temp should reduce floating risk"


def test_beam_loads_positive():
    loads = physics.beam_pump_loads(spm=4.0, stroke_length_in=86.0, temp_c=60.0)
    assert loads.pprl_n > loads.mprl_n > 0


def test_temperature_decay_returns_to_baseline():
    t_immediate = physics.temperature_decay(day=0, steam_volume_bbl=4000)
    t_far = physics.temperature_decay(day=200, steam_volume_bbl=4000)
    assert t_immediate > t_far
    assert abs(t_far - physics.RESERVOIR_TEMP_BASE_C) < 1.0

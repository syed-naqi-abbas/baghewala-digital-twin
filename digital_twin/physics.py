"""
physics.py — First-principles engineering relationships for the Baghewala
well-to-surface digital twin (SIH26120).

IMPORTANT / HONESTY NOTE
-------------------------
These equations are the *real, named* engineering relationships used in
heavy-oil / artificial-lift engineering (Walther viscosity equation, API
RP 11L beam-pump loads, a hydrodynamic rod-floating criterion). However,
the *constants* below (A, B, rod/tubing dimensions, decay time constants,
etc.) are illustrative placeholders chosen to produce physically
plausible numbers for a 17-19 API / 46-48 C reservoir. They are NOT
calibrated against real Baghewala core/PVT/completion data.

Before any field use, these constants must be recalibrated against real
fluid-assay (PVT) reports and well-completion data supplied by Oil India
Limited. That recalibration step is explicitly listed as "remaining
work" in the README.

In the full solution architecture, this physics layer is designed to run
in MATLAB (pdepe for heat conduction, a 1-D damped wave equation for the
rod string). It is reimplemented here in pure Python/NumPy so the
prototype has zero licensing dependencies and is trivially runnable by
hackathon judges. Swapping this module for a MATLAB Engine API call is a
drop-in replacement — see README "What remains".
"""

from dataclasses import dataclass
import numpy as np

# ---------------------------------------------------------------------------
# Reservoir / fluid constants (Baghewala-representative, from the SIH26120
# problem statement + public field disclosures; see README references)
# ---------------------------------------------------------------------------
API_GRAVITY = 18.0                      # PS states 17-19 API
SG_OIL = 141.5 / (131.5 + API_GRAVITY)  # specific gravity from API gravity
RHO_OIL = SG_OIL * 1000.0               # kg/m3
RHO_STEEL = 7850.0                      # kg/m3
G = 9.81                                # m/s2

RESERVOIR_TEMP_BASE_C = 47.0            # PS states 46-48 C
RESERVOIR_PRESSURE_BASE_BAR = 45.0      # representative low reservoir pressure

# Sucker rod / tubing geometry (representative API sizes for this depth class)
ROD_DIAMETER_M = 0.022      # 7/8" sucker rod
TUBING_ID_M = 0.062         # ~2.441" tubing ID
ROD_LENGTH_M = 1000.0       # representative pump depth (~1000 m, per field data)

# Walther (ASTM D341) constants, back-solved so that viscosity ~8000 cP at
# 47 C and ~100 cP at 140 C -- illustrative, see module docstring.
WALTHER_A = 7.150
WALTHER_B = 2.618

# Downhole friction/drag calibration factor. Real dynamometer models use
# an "iterated downhole friction factor" fitted to measured cards (this is
# standard practice, not a shortcut specific to this prototype -- see the
# Sandia Downhole Dynamometer Database reference in the README). Here it is
# a placeholder tuned so that realistic SPM (2-8) at cold, unstimulated
# reservoir temperature (~47 C) spans low-to-critical floating risk, and
# drops once CSS raises near-wellbore temperature. MUST be recalibrated
# against real Baghewala dynamometer cards before field use.
CALIBRATION_FRICTION_FACTOR = 2.6


def kinematic_viscosity_cst(temp_c: float) -> float:
    """Walther / ASTM D341 equation: log10log10(v + 0.7) = A - B*log10(T[K])"""
    t_k = temp_c + 273.15
    y = WALTHER_A - WALTHER_B * np.log10(t_k)
    y = np.clip(y, -1.5, 1.5)  # numerical guard against runaway exponents
    return float(10 ** (10 ** y) - 0.7)


def dynamic_viscosity_cp(temp_c: float) -> float:
    """Dynamic viscosity (cP) ~= kinematic viscosity (cSt) * specific gravity."""
    return kinematic_viscosity_cst(temp_c) * SG_OIL


@dataclass
class RodFloatResult:
    f_viscous_n: float
    w_buoyant_n: float
    risk_ratio: float       # >= 1.0 means floating criterion met
    floating: bool


def rod_floating_check(temp_c: float, spm: float, stroke_length_m: float) -> RodFloatResult:
    """
    Rod-floating criterion: the rod string floats when upward viscous drag
    on the downstroke (F_viscous) meets or exceeds the string's buoyant
    weight (W_buoyant).

        W_buoyant = rho_steel * g * A_rod * L * (1 - rho_fluid / rho_steel)
        F_viscous = pi * mu * v_down * L / ln(D_tubing / D_rod)

    A risk_ratio >= 1.0 signals imminent slack-string / rod-float risk.
    """
    a_rod = np.pi / 4 * ROD_DIAMETER_M ** 2
    w_buoyant = RHO_STEEL * G * a_rod * ROD_LENGTH_M * (1 - RHO_OIL / RHO_STEEL)

    mu_pa_s = dynamic_viscosity_cp(temp_c) * 1e-3
    v_down = (2 * stroke_length_m * spm) / 60.0  # avg downstroke velocity, m/s
    ln_ratio = np.log(TUBING_ID_M / ROD_DIAMETER_M)
    f_viscous = CALIBRATION_FRICTION_FACTOR * np.pi * mu_pa_s * v_down * ROD_LENGTH_M / ln_ratio

    ratio = f_viscous / w_buoyant if w_buoyant > 0 else np.inf
    return RodFloatResult(
        f_viscous_n=f_viscous,
        w_buoyant_n=w_buoyant,
        risk_ratio=ratio,
        floating=ratio >= 1.0,
    )


@dataclass
class BeamLoadResult:
    pprl_n: float
    mprl_n: float


def beam_pump_loads(spm: float, stroke_length_in: float, temp_c: float) -> BeamLoadResult:
    """
    Simplified API RP 11L peak/minimum polished-rod load.

        PPRL = Wf + Wr*(1 + S*SPM^2/1784)
        MPRL = Wr*(1 - S*SPM^2/1784 - rho_fluid/rho_steel)

    S is stroke length in inches (API RP 11L convention); SPM = strokes/min.
    Wf, Wr are fluid load and rod-string weight respectively.
    """
    a_rod = np.pi / 4 * ROD_DIAMETER_M ** 2
    a_plunger = np.pi / 4 * (ROD_DIAMETER_M * 1.4) ** 2  # representative plunger area

    wf = RHO_OIL * G * a_plunger * ROD_LENGTH_M
    wr = RHO_STEEL * G * a_rod * ROD_LENGTH_M

    accel_factor = stroke_length_in * spm ** 2 / 1784.0
    friction_bonus = 0.05 * wr  # small representative upstroke friction term

    pprl = wf + wr * (1 + accel_factor) + friction_bonus
    mprl = wr * (1 - accel_factor - RHO_OIL / RHO_STEEL)
    return BeamLoadResult(pprl_n=pprl, mprl_n=mprl)


def temperature_decay(day: int, steam_volume_bbl: float, base_temp_c: float = RESERVOIR_TEMP_BASE_C,
                       k_steam: float = 0.018, tau_days: float = 18.0) -> float:
    """
    Near-wellbore temperature at `day` days after a steam slug of
    `steam_volume_bbl` barrels was injected. Simplified radial heat-loss
    decay: T(t) = T_base + (T_peak - T_base) * exp(-t / tau).
    """
    t_peak = base_temp_c + k_steam * steam_volume_bbl
    return base_temp_c + (t_peak - base_temp_c) * np.exp(-day / tau_days)

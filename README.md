# Baghewala well-to-surface digital twin — SIH26120 prototype

**Smart India Hackathon — Problem Statement SIH26120**
**Organization:** Oil India Limited
**Target field:** Baghewala Field, Bikaner-Nagaur sub-basin, Rajasthan (Jodhpur Sandstone reservoir)

---

## What this prototype demonstrates

The problem statement asks for an AI-enabled digital twin that jointly
optimizes Cyclic Steam Stimulation (CSS) and Sucker Rod Pump (SRP)
operations instead of tuning them separately. This repo implements the
full pipeline end-to-end on synthetic data so the *approach* can be judged
and interacted with, ahead of any real field-data access:

```
MATLAB-equivalent physics layer  ->  data ingestion  ->  ML prediction
        ->  constrained optimizer  ->  engineer dashboard
        (closed loop: outcomes recalibrate the models)
```

Run it yourself with `streamlit run app.py` (see Quick start below) — it's
a live, interactive dashboard, not a static mockup.

---

## What is actually implemented (working code, tested)

- **Physics layer** (`digital_twin/physics.py`)
  - Walther / ASTM D341 viscosity-temperature equation, calibrated so
    viscosity runs from roughly 8,000 cP at 47°C down to roughly 100 cP
    at 140°C, consistent with the 17–19° API crude and 46–48°C reservoir
    temperature stated in the problem statement
  - A hydrodynamic rod-floating criterion: floating risk is computed as
    the ratio of downstroke viscous drag force to the rod string's
    buoyant weight; a ratio ≥ 1.0 signals floating risk
  - Simplified API RP 11L peak/minimum polished-rod load (PPRL/MPRL)
    equations
  - A thermal-decay function modelling near-wellbore cooling after a
    steam slug, back-calculated to an exponential decay curve

- **Coupled CSS + SRP simulator** (`digital_twin/simulator.py`)
  - Simulates a full soak-then-produce cycle day by day for one well
  - Oil rate is the lesser of temperature-adjusted reservoir inflow and
    SRP lift capacity, with lift capacity degraded when the floating-risk
    ratio rises
  - Produces Steam-Oil Ratio (SOR), cumulative oil, and floating-risk
    time series per cycle

- **Synthetic dataset generator** (`digital_twin/data_gen.py`)
  - Runs the simulator across 300 randomised CSS/SRP configurations
    (~13,000 daily records) with injected sensor noise, standing in for
    real Oil India historical data at this stage

- **Three trained ML surrogate models** (`digital_twin/ml_models.py`),
  with real (not invented) metrics from the last training run in this
  environment:

  | Model | Algorithm | Metric | Value |
  |---|---|---|---|
  | Production forecaster | Gradient boosted regressor | MAE / R² | 4.25 BOPD / 0.845 |
  | Thermal decay predictor | Polynomial ridge regression | MAE / R² | 0.67 °C / 0.998 |
  | Rod-floating / failure risk classifier | Random forest | Precision / Recall | 0.14 / 1.00 |

  Read these honestly: high R² reflects the models learning the
  *simulator's* internal logic well, which is expected on synthetic data
  — it is not a claim of real-world field accuracy. The risk classifier
  is deliberately biased toward recall (it catches 100% of synthetic
  failure events, at the cost of over-flagging) — for a safety-relevant
  alert, missing a real failure is worse than a false alarm, so this
  precision/recall trade-off is an intentional starting point, not a
  bug, though it should be retuned once real failure-rate base rates are
  known.

- **Constrained CSS + SRP optimizer** (`digital_twin/optimizer.py`)
  - Grid-searches steam volume, soak duration, and SPM to maximise
    cumulative oil net of steam cost
  - Treats the rod-floating threshold as a **hard constraint** —
    verified in testing to actually reject unsafe high-SPM/short-soak
    combinations (in a typical run, 23 of 80 candidate settings are
    rejected for exceeding the risk limit), not just penalise them

- **Interactive dashboard** (`app.py`, Streamlit)
  - What-if cycle simulator with live charts (temperature/viscosity,
    production/floating-risk) and threshold alerts
  - Optimizer tab showing the recommended pump-safe CSS/SRP plan and the
    full candidate table, including which settings were rejected and why
  - ML model performance tab
  - Explicit scope/safety banner

- **Test suite** (`tests/`) — 11 tests covering the physics relationships
  (viscosity trends correctly with temperature, floating risk trends
  correctly with SPM and temperature, beam loads are physically ordered)
  and the simulator/optimizer (correct shape, no production during soak,
  optimizer returns a feasible result). All 11 pass in this environment.

---

## What remains (explicitly out of scope for this prototype)


1. **Real field data.** Nothing here has been trained or validated on
   actual Oil India production, CSS, SRP/VFD, or failure-history data.
   The data schema in `data_gen.py` is designed to be swappable for a
   real historical dataset without changing the ML/optimizer code, but
   that swap and the resulting re-validation has not been done.
2. **Physics constant calibration.** The Walther equation constants, rod/
   tubing dimensions, the friction calibration factor, and the thermal
   decay time constant are illustrative placeholders chosen to be
   physically plausible for a 17–19° API, 46–48°C reservoir — not fitted
   to real Baghewala PVT/core/completion reports.
3. **MATLAB physics engine.** The solution architecture specifies MATLAB
   (`pdepe` for radial heat conduction, a 1-D damped wave equation for
   the rod string) as the production physics layer. This prototype
   reimplements the same named equations in Python/NumPy for zero-
   dependency portability. Swapping in an actual MATLAB Engine API call
   behind the same function signatures in `physics.py` is a scoped,
   well-defined follow-up, not a redesign.
4. **Reinforcement-learning SPM controller.** The optimizer here is a
   transparent, fully explainable grid search. An RL agent trained
   across many simulated cycles (the stretch goal in the solution
   architecture) has not been built.
5. **Multi-well / field-level scaling.** This prototype models one well.
   Extending to Baghewala's ~30+ active wells (individual calibration,
   a field-level dashboard) is architecturally straightforward given the
   current module boundaries but not implemented.
6. **SCADA / live telemetry integration.** No live data ingestion exists;
   everything currently runs on simulator output.
7. **Historical back-testing.** The most important validation question —
   would this system have flagged known past Baghewala rod/pump failures
   earlier than they were actually caught — cannot be answered without
   real historical failure data, and hasn't been attempted here.
8. **Security/auth/audit trail.** No role-based access control or
   decision-audit logging is implemented; the solution architecture
   specifies this as required before any pilot deployment.

---

## Quick start

```bash
git clone <this-repo-url>
cd baghewala-digital-twin
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# optional: generate data + train + save models explicitly
# (the dashboard also does this automatically on first run)
python scripts/train_models.py

# run tests
pytest tests/ -v

# launch the dashboard
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. First run trains and
caches the models (a few seconds); subsequent runs load the saved models
from `saved_models/`.

---

## Project layout

```
digital_twin/
  physics.py      # Walther viscosity, rod-floating criterion, API RP11L loads, thermal decay
  simulator.py     # coupled daily CSS + SRP simulation for one well
  data_gen.py      # synthetic multi-cycle dataset generator
  ml_models.py     # train/save/load the 3 ML surrogate models
  optimizer.py     # constrained CSS + SRP grid-search optimizer
app.py             # Streamlit dashboard (entry point)
scripts/
  train_models.py  # standalone training script
tests/
  test_physics.py
  test_simulator.py
```

---

## Technical references

- Walther / ASTM D341 viscosity-temperature equation
- API RP 11L — beam pumping unit design/load calculation standard
- Sandia National Laboratories — Downhole Dynamometer Database
  (precedent for friction-factor calibration against real dynamometer
  cards): sandia.gov
- SIH26120 problem statement — Oil India Limited, Baghewala Field
- Oil India Limited — Rajasthan Fields (CSS + SRP field configuration):
  oil-india.com/rajasthan-fields

See the accompanying solution PPT for the full reference list, including
published ML studies on real sucker-rod-pump failure datasets that
motivate the modelling approach used here.

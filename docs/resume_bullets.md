# Resume bullets

Pick one block. Numbers are from `reports/metrics.json`; keep them in sync if you re-run with other settings.

## Three-line version (core / PSU resume)

**Transmission-Line Fault Detection, Classification and Location** — Python, NumPy/SciPy, scikit-learn, Streamlit · [github.com/MayankGangwal-code/transmission-line-fault-classification](https://github.com/MayankGangwal-code/transmission-line-fault-classification)
* Built a numerical line-protection relay from first principles — delta disturbance detector, full-cycle DFT
  phasors, symmetrical components, sequence-angle phase selection and impedance-zone logic — that
  **classifies 11 fault classes with zero errors on 3,300 test events and on an unseen 150 km line, one
  cycle after the fault**, with settings derived from the CT error model and the line data, no training.
* Wrote and validated a physics simulator of a 220 kV two-source line (coupled-phase π-sections, arc and
  ground resistance, exact matrix-exponential integration; fault currents within 1 % of textbook formulas)
  and generated 12,100 randomised events (faults and load switchings); Random Forest / gradient boosting on the same relay
  quantities match the relay logic (100 %) and beat it only when the decision is forced to half a cycle.
* Implemented reactance and Takagi one-ended fault location (**0.68 km mean error on 100 km**, beating an
  ML regressor), studied decision time, measurement noise and high-resistance faults, and shipped a
  Streamlit app, CLI and 31 tests.

## One-line version

* **Transmission-line fault relay** — physics simulator + numerical-relay logic (DFT phasors, symmetrical
  components, Takagi location) vs. ML on 12,100 simulated 220 kV events: zero classification errors with
  derived relay settings, matched by a Random Forest, 0.68 km location error; Streamlit app, 31 tests.

## Skills line

Power-system protection (distance relaying, symmetrical components, fault analysis, fault location) ·
Python (NumPy, SciPy, pandas, scikit-learn, matplotlib, Streamlit, pytest) · time-domain simulation ·
signal processing (DFT phasor estimation) · machine learning evaluation (generalisation, robustness studies)

## How to talk about it in one breath

"I built the relay that protects a transmission line — the way real numerical relays do it — and a physics
simulator to test it on thousands of faults; textbook relay logic with properly derived settings gets every
test event right, machine learning only matches it, and I can explain exactly where each one fails and why."

"""Interactive demo: create a fault on the line and watch the relay decide.

Run with:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tlf import CLASSES, GROUPS  # noqa: E402
from tlf.classical import ABC_ZONE, BALANCED_I2, GROUND_RATIO, RULES  # noqa: E402
from tlf.features import analyse_record, feature_names  # noqa: E402
from tlf.locate import reactance_distance_km, takagi_distance_km  # noqa: E402
from tlf.models import decode  # noqa: E402
from tlf.plots import plot_phasors, plot_waveforms  # noqa: E402
from tlf.simulate import FaultSpec, SystemParams, add_noise, simulate  # noqa: E402

st.set_page_config(page_title="Transmission-line fault relay", page_icon="⚡", layout="wide")

CLASS_LABELS = {
    "none": "No fault (balanced load switching)", "AG": "A–ground", "BG": "B–ground", "CG": "C–ground", "AB": "A–B", "BC": "B–C", "CA": "C–A",
    "ABG": "A–B–ground", "BCG": "B–C–ground", "CAG": "C–A–ground", "ABC": "three-phase (A–B–C)",
}


@st.cache_resource
def get_models():
    import joblib

    clf = joblib.load(ROOT / "models" / "classifiers.joblib")
    loc = joblib.load(ROOT / "models" / "locator.joblib")
    return clf, loc


try:
    clf_bundle, loc_bundle = get_models()
except FileNotFoundError:
    st.error("models/ not found — run `python scripts/run_experiments.py` first")
    st.stop()

st.title("⚡ Transmission-Line Fault Detection, Classification and Location")
st.caption(
    "A 220 kV, 100 km line between two sources, simulated from first principles. Choose a fault, and the "
    "relay at the sending end detects it, identifies the type with textbook relay logic and with machine "
    "learning, and estimates where it is."
)

with st.sidebar:
    st.header("Event")
    kind = st.selectbox("Fault type", CLASSES, index=1, format_func=lambda c: f"{c} — {CLASS_LABELS[c]}")
    distance = st.slider("Distance from the relay (km)", 2.0, 98.0, 60.0, 1.0)
    r_arc = st.slider("Arc resistance per phase (Ω)", 0.01, 5.0, 0.5, 0.01)
    r_ground = st.slider("Ground resistance (Ω) — ground faults only", 0.1, 50.0, 5.0, 0.1)
    inception = st.slider("Inception angle of the phase-A voltage (°)", 0, 359, 90, 1, help="90° = voltage zero crossing → maximum DC offset in the fault current")
    load_step = st.slider("Load switched on at F for the no-fault case (MW)", 0, 150, 100, 10)
    st.header("System")
    delta = st.slider("Load angle δ (°) — remote source lags", 0, 30, 15, 1)
    snr = st.slider("Measurement noise SNR (dB)", 20, 60, 40, 1)
    decision = st.select_slider("Decision time after the trigger (cycles)", [0.25, 0.5, 0.75, 1.0, 1.5, 2.0], value=1.0)

system = SystemParams(delta_deg=float(delta))
spec = FaultSpec(kind=kind, distance_km=float(distance), r_arc=float(r_arc), r_ground=float(r_ground), inception_deg=float(inception),
                 load_step_mw=float(load_step) if kind == "none" else 0.0)
rec = add_noise(simulate(spec, system, post_cycles=5), float(snr), seed=42)
an = analyse_record(rec, float(decision))
an3 = analyse_record(rec, 3.0)
f = an.features
n = system.samples_per_cycle

# ----------------------------------------------------------------------------
# decisions
# ----------------------------------------------------------------------------
rows = []
for name, rule in RULES.items():
    rows.append({"method": name, "decision": rule(f)})
X = pd.DataFrame([f])[clf_bundle["feature_names"]]
for name, pipe in clf_bundle["models"].items():
    rows.append({"method": name, "decision": decode(pipe.predict(X))[0]})
dec = pd.DataFrame(rows)
dec["correct"] = np.where(dec["decision"] == kind, "✓", "✗")
dec["fault type"] = dec["decision"].map(GROUPS)

c1, c2, c3, c4 = st.columns(4)
c1.metric("True event", f"{kind} — {CLASS_LABELS[kind]}")
c2.metric("Disturbance detector", "tripped" if an.triggered else "quiet", delta=(f"{(an.trigger_index - rec.fault_index) / system.fs * 1e3:.1f} ms after the event" if an.triggered else None), delta_color="off")
best_pred = dec.loc[dec.method == clf_bundle["best"], "decision"].iloc[0]
c3.metric(f"{clf_bundle['best']} says", best_pred, delta="correct" if best_pred == kind else "wrong", delta_color="normal" if best_pred == kind else "inverse")
seq_pred = dec.loc[dec.method == "Sequence-angle relay logic", "decision"].iloc[0]
c4.metric("Relay logic says", seq_pred, delta="correct" if seq_pred == kind else "wrong", delta_color="normal" if seq_pred == kind else "inverse")

st.subheader("Fault classification")
st.dataframe(dec.set_index("method"))

if kind != "none":
    st.subheader("Fault location (window ending 3 cycles after the trigger)")
    loc_rows = {
        "Reactance method": reactance_distance_km(an3, kind, system),
        "Takagi method": takagi_distance_km(an3, kind, system),
    }
    X3 = pd.DataFrame([an3.features])
    onehot = pd.get_dummies(pd.Categorical([best_pred], categories=CLASSES)).astype(float)
    L = pd.concat([X3, onehot], axis=1)[loc_bundle["columns"]]
    loc_rows["ML regressor (gradient boosting)"] = float(loc_bundle["locator"].predict(L)[0])
    loc_df = pd.DataFrame({"estimate (km)": loc_rows})
    loc_df["error (km)"] = loc_df["estimate (km)"] - distance
    st.dataframe(loc_df.round(2))

# ----------------------------------------------------------------------------
# signals
# ----------------------------------------------------------------------------
st.subheader("What the relay measures")
col_w, col_p = st.columns([2, 1])
with col_w:
    fig = plot_waveforms(rec, f"{CLASS_LABELS[kind]} at {distance:.0f} km", trigger=an.trigger_index,
                         decision_end=int(an.trigger_index + decision * n) if an.triggered else None)
    st.pyplot(fig)
    plt.close(fig)
with col_p:
    fig_p = plot_phasors(an.V, an.I, title="Phasors in the decision window")
    st.pyplot(fig_p)
    plt.close(fig_p)

st.subheader("Sequence components in the decision window")
seq = pd.DataFrame(
    {
        "zero (0)": [abs(an.seq_V[0]) / 1e3, abs(an.seq_I[0])],
        "positive (1)": [abs(an.seq_V[1]) / 1e3, abs(an.seq_I[1])],
        "negative (2)": [abs(an.seq_V[2]) / 1e3, abs(an.seq_I[2])],
    },
    index=["voltage (kV peak)", "current (A peak)"],
).round(1)
st.dataframe(seq)
a20 = np.degrees(np.arctan2(f["sin_a20"], f["cos_a20"]))
a21 = np.degrees(np.arctan2(f["sin_a21"], f["cos_a21"]))
st.markdown(
    f"**Relay selector angles:** ∠(I₂ / ΔI₁) = `{a21:+.0f}°` (AG 0°, BG +120°, CG −120°, BC/BCG 180°, AB/ABG +60°, CA/CAG −60°) · "
    f"∠(I₂ / I₀) = `{a20:+.0f}°` · residual current / largest phase current = `{f['ires']:.2f}` (ground if > {GROUND_RATIO:.2f}) · "
    f"|I₂| / |ΔI₁| = `{f['r_i2_di1']:.2f}` (balanced if < {BALANCED_I2:.2f}) · "
    f"largest phase-loop |Z| = `{max(f['zmag_ab'], f['zmag_bc'], f['zmag_ca']):.2f}` × line impedance (three-phase fault if < {ABC_ZONE:.1f} when balanced)"
)

with st.expander("How does this work?"):
    st.markdown(
        """
**The line is simulated from physics.** Two sources, a 100 km line with coupled phases (positive- and
zero-sequence constants), the fault as a resistance network at the chosen point — integrated exactly with
a matrix exponential, so every waveform (including the DC offset that depends on the inception angle)
comes from the circuit equations.

**The relay** samples voltage and current at 4 kHz, detects the disturbance from the change in current
against one cycle earlier, computes full-cycle DFT phasors, and from them the symmetrical components:

| quantity | meaning |
|---|---|
| I₀ (zero sequence) | flows only when ground is involved (residual current 3 I₀) |
| I₂ (negative sequence) | appears for every unbalanced fault; absent for a three-phase fault or a load change |
| ΔI₁ (superimposed positive sequence) | the "pure fault" current — what the fault added to the load current |

The angle between I₂ and ΔI₁ identifies the faulted phase(s) — that is how numerical distance relays select
the phase. Residual current then separates line-to-line from double-line-to-ground faults, and for a
balanced event the loop impedance tells a three-phase fault (inside the zone) from a load change (far
outside it).

**Fault location** uses the apparent impedance of the faulted loop: a bolted fault at d km shows exactly
d · Z₁. Fault resistance plus current from the far end distorts the reactance method; Takagi's method
projects on the superimposed current to cancel most of that error.
"""
    )

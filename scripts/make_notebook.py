"""Generate notebooks/fault_relay_walkthrough.ipynb (plain JSON, no jupyter needed to build it).

Usage:  python scripts/make_notebook.py             # write the notebook without outputs
        python scripts/make_notebook.py --execute   # also run every cell and store its outputs (tables, figures)

``--execute`` uses a small built-in runner (stdout, the last expression of each cell, matplotlib figures)
so that the committed notebook renders with its results on GitHub without needing jupyter here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "fault_relay_walkthrough.ipynb"

cells: list[dict] = []


def md(text: str) -> None:
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text: str) -> None:
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n")})


md("""
# Transmission-line fault relay — walkthrough

This notebook rebuilds the project step by step on single events: simulate a fault, look at what the relay
measures, derive the sequence-angle phase selection, run the rule and the trained models, locate the fault,
and finish with the saved results of the full experiment. Run it from the repository root after
`python scripts/run_experiments.py` (the saved features and models in `data/` and `models/` are used in the
later sections).
""")

code("""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

ROOT = Path.cwd() if (Path.cwd() / "src" / "tlf").exists() else Path.cwd().parent
sys.path.insert(0, str(ROOT / "src"))

from tlf import CLASSES, GROUPS
from tlf.simulate import FaultSpec, SystemParams, simulate, add_noise, sequence_impedances
from tlf.dsp import dft_phasor, sequence_components, detect_disturbance
from tlf.features import analyse_record, feature_names, k0_factor
from tlf.classical import sequence_rule, magnitude_rule, apply_rule
from tlf.locate import reactance_distance_km, takagi_distance_km
from tlf.plots import plot_waveforms, plot_phasors, plot_sequence_angle_map, plot_confusion
from tlf.evaluate import summarise, confusion

system = SystemParams()
N = system.samples_per_cycle
print(f"{system.v_ll_kv:.0f} kV, {system.f0:.0f} Hz, {system.line.length_km:.0f} km line, {system.fs:.0f} Hz sampling = {N} samples per cycle")
""")

md("""
## 1. Simulate a fault and look at what the relay measures

A phase-A-to-ground fault at 60 km through 0.5 Ω of arc and 5 Ω of ground resistance, starting at the
voltage zero crossing (inception 90°) — the worst case for DC offset. The record is passed through the
measurement model (±2 % ratio errors, 40 dB noise) exactly as in the experiments.
""")

code("""
spec = FaultSpec(kind="AG", distance_km=60, r_arc=0.5, r_ground=5.0, inception_deg=90.0)
rec = add_noise(simulate(spec, system, post_cycles=5), 40.0, seed=1)
an = analyse_record(rec, decision_cycles=1.0)
print("disturbance detector tripped at sample", an.trigger_index, f"= {(an.trigger_index - rec.fault_index) / system.fs * 1e3:.2f} ms after the fault")
plot_waveforms(rec, "A-G fault at 60 km, Rf = 5.5 Ω, inception at the voltage zero crossing", trigger=an.trigger_index, decision_end=an.trigger_index + N);
""")

md("""
The dashed line is the fault instant, the dotted line the trigger of the delta-current detector (every
sample compared with the sample one cycle earlier); the shaded band is the one-cycle DFT window the relay
decides on. Note the DC offset of the phase-A current — the first peak is much larger than the steady
fault current.
""")

md("""
## 2. Does the simulator agree with the textbook?

With the remote source removed (radial feed) and a bolted fault, the fault currents must follow the
hand formulas: I₃ϕ = E / Z₁, I_AG = 3E / (Z₁ + Z₂ + Z₀) with Z₂ = Z₁, I_BC = √3 E / (Z₁ + Z₂).
""")

code("""
radial = SystemParams(r_sc_mva=1e-3, e_r_pu=0.0, delta_deg=0.0)
rows = []
for kind, d in [("ABC", 20), ("ABC", 60), ("AG", 20), ("AG", 60), ("BC", 40)]:
    r = simulate(FaultSpec(kind=kind, distance_km=d, r_arc=1e-4, r_ground=1e-4), radial, post_cycles=8)
    I = dft_phasor(r.i, r.fault_index + 6 * N, N)
    z = sequence_impedances(radial, d)
    E = radial.v_phase_peak
    expected = {"ABC": E / abs(z["Z1_total"]), "AG": 3 * E / abs(2 * z["Z1_total"] + z["Z0_total"]), "BC": np.sqrt(3) * E / abs(2 * z["Z1_total"])}[kind]
    measured = {"ABC": abs(I).mean(), "AG": abs(I[0]), "BC": abs(I[1])}[kind]
    rows.append({"fault": kind, "distance km": d, "formula (A peak)": expected, "simulated (A peak)": measured, "error %": 100 * (measured / expected - 1)})
pd.DataFrame(rows).round(2)
""")

md("""
## 3. Symmetrical components and the phase-selection angles

From the phasors of the decision window: the Fortescue components, the superimposed positive-sequence
current ΔI₁ = I₁ − I₁(pre-fault), and the two selector angles. For an A-G fault the boundary conditions give
I₁ = I₂ = I₀ at the fault, so ∠(I₂/ΔI₁) ≈ 0° and ∠(I₂/I₀) ≈ 0°.
""")

code("""
f = an.features
print("sequence currents (A peak): I0 = %.0f, I1 = %.0f, I2 = %.0f, dI1 = %.0f" % (abs(an.seq_I[0]), abs(an.seq_I[1]), abs(an.seq_I[2]), abs(an.dI1)))
print("angle(I2 / dI1) = %+.1f deg,  angle(I2 / I0) = %+.1f deg" % (np.degrees(np.arctan2(f["sin_a21"], f["cos_a21"])), np.degrees(np.arctan2(f["sin_a20"], f["cos_a20"]))))
print("residual current / largest phase current = %.2f" % f["ires"])
plot_phasors(an.V, an.I, title="Phasors in the decision window");
""")

md("""
Now the same angle for every class — one clean fault of each type at 45 km. The table is the whole
phase-selection logic: six sectors of ∠(I₂/ΔI₁), residual current for the ground flag, no I₂ at all for a
balanced event.
""")

code("""
rows = []
for kind in CLASSES:
    s = FaultSpec(kind=kind, distance_km=45.0, r_arc=0.3, r_ground=2.0, inception_deg=30.0, load_step_mw=100.0 if kind == "none" else 0.0)
    ff = analyse_record(add_noise(simulate(s, system), 40.0, seed=2), 1.0).features
    rows.append({"class": kind, "type": GROUPS[kind], "angle(I2/dI1) deg": np.degrees(np.arctan2(ff["sin_a21"], ff["cos_a21"])),
                 "angle(I2/I0) deg": np.degrees(np.arctan2(ff["sin_a20"], ff["cos_a20"])), "|I2|/|dI1|": ff["r_i2_di1"], "residual ratio": ff["ires"],
                 "I1 rise": ff["ri1"], "sequence rule": sequence_rule(ff), "overcurrent rule": magnitude_rule(ff)})
pd.DataFrame(rows).set_index("class").round(2)
""")

md("""
## 4. The decision space of the whole test set

Every one of the 3,300 test events (features saved by `run_experiments.py`), coloured by its true class.
Left: ground faults in the plane of the two angles — single-phase faults sit at 0°, ±120° on both axes,
double-line-to-ground faults keep their ∠(I₂/ΔI₁) sector but drift along ∠(I₂/I₀) as the ground
resistance grows. Right: phase-to-phase faults at 180°, ±60°, three-phase faults with no negative sequence.
""")

code("""
test = pd.read_csv(ROOT / "data" / "events_test.csv")
X_test, y_test = test[feature_names()], test["label"].to_numpy()
plot_sequence_angle_map(X_test, y_test);
""")

md("""
## 5. Rules versus trained models on the test set
""")

code("""
import joblib
from tlf.models import decode
bundle = joblib.load(ROOT / "models" / "classifiers.joblib")
preds = {"Overcurrent rule": apply_rule(magnitude_rule, X_test), "Sequence-angle relay logic": apply_rule(sequence_rule, X_test)}
for name, pipe in bundle["models"].items():
    preds[name] = decode(pipe.predict(X_test))
pd.DataFrame({n: summarise(y_test, p) for n, p in preds.items()}).T.round(4)
""")

code("""
plot_confusion(confusion(y_test, preds["Sequence-angle relay logic"]), title="Sequence-angle relay logic — test set");
""")

md("""
Where is the rule wrong? On the test set nowhere; on the 6,600 training events exactly once — a
double-line-to-ground fault at the far end through 32 Ω of ground resistance, whose residual current at
the relay (5.7 % of the phase current) is just under the 6 % ground pick-up because most of its
zero-sequence current flows in from the remote end. The cell lists every misclassified event of both sets.
""")

code("""
train = pd.read_csv(ROOT / "data" / "events_train.csv")
cols = ["label", "distance_km", "r_arc", "r_ground", "load_step_mw", "delta_deg", "ires", "r_i2_di1", "zmag_ab"]
rows = []
for name, df in [("train", train), ("test", test)]:
    p = apply_rule(sequence_rule, df[feature_names()])
    wrong = p != df["label"].to_numpy()
    e = df.loc[wrong, cols].copy()
    e.insert(0, "set", name)
    e["predicted"] = p[wrong]
    rows.append(e)
pd.concat(rows).round(3)
""")

md("""
## 6. Fault location

The reactance method reads the distance from the imaginary part of the loop impedance; Takagi projects on
the superimposed current to cancel the effect of fault resistance with remote infeed. Both use the window
ending 3 cycles after the trigger, when the DC offset has decayed. Try a bolted fault and a resistive one.
""")

code("""
rows = []
for d, r_arc, r_g in [(60, 1e-4, 1e-4), (60, 0.5, 5.0), (60, 1.0, 30.0), (90, 1.0, 30.0)]:
    r = add_noise(simulate(FaultSpec(kind="AG", distance_km=d, r_arc=r_arc, r_ground=r_g, inception_deg=20.0), system, post_cycles=5), 40.0, seed=3)
    a3 = analyse_record(r, 3.0)
    rows.append({"true km": d, "Rf (ohm)": r_arc + r_g, "reactance km": reactance_distance_km(a3, "AG", system), "Takagi km": takagi_distance_km(a3, "AG", system)})
pd.DataFrame(rows).round(2)
""")

md("""
## 7. The full experiment

All numbers of the README come from `reports/metrics.json`, written by `scripts/run_experiments.py`.
""")

code("""
import json
m = json.loads((ROOT / "reports" / "metrics.json").read_text())
print("events: train %d, test %d, unseen system %d; features %d; best ML model: %s" % (m["n_train"], m["n_test"], m["n_unseen"], m["n_features"], m["best_ml"]))
display(pd.DataFrame(m["test"]).T[["accuracy", "macro_f1", "detection_accuracy", "missed_faults", "false_trips"]].round(4))
print("unseen system accuracy:"); display(pd.DataFrame(m["unseen_system"]).T[["accuracy"]].round(4))
print("accuracy versus decision time (cycles after trigger):"); display(pd.DataFrame(m["decision_time"]).T.round(3))
print("accuracy versus SNR:"); display(pd.DataFrame(m["noise"]).T.round(3))
print("fault location (km):"); display(pd.DataFrame(m["location"]).T.round(2))
""")

md("""
## 8. Take-aways

* The angles between the sequence currents are the physics of the fault; with them, relay settings derived
  from the measurement model classify 11 event types with zero errors on the test set one cycle after the
  trigger — and with one error in 6,600 training events, which shows where the margin ends.
* Machine learning on the same per-unit relay quantities matches that (100 %) and transfers to a line it
  never saw; it earns its place only at very early decision times, not by replacing the relay logic.
* Takagi's method — a 1982 closed-form formula using the superimposed current — locates faults better than
  a gradient-boosting regressor trained on 6,000 faults.
* Raw samples (the Kaggle dataset) give 87 %; a one-cycle window of relay quantities gives 100 %. The
  feature is the engineering.
""")

nb = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}


def execute(nb: dict) -> None:
    """Run the code cells in one namespace and store notebook-style outputs in place."""
    import ast
    import base64
    import contextlib
    import io
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cwd = os.getcwd()
    os.chdir(OUT.parent)  # the notebook resolves ROOT relative to its own folder
    ns: dict = {}
    outputs_of_cell: list[dict] = []

    def rich(obj) -> dict:
        data = {"text/plain": repr(obj)}
        if hasattr(obj, "_repr_html_"):
            data["text/html"] = obj._repr_html_()
        return data

    def display(obj) -> None:  # notebook-style display() inside a cell
        outputs_of_cell.append({"output_type": "display_data", "data": rich(obj), "metadata": {}})

    ns["display"] = display
    count = 0
    try:
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            count += 1
            outputs_of_cell = []
            buf = io.StringIO()
            tree = ast.parse(cell["source"])
            last = None
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = ast.Expression(tree.body[-1].value)
                tree.body = tree.body[:-1]
            with contextlib.redirect_stdout(buf):
                exec(compile(tree, f"<cell {count}>", "exec"), ns)
                value = eval(compile(last, f"<cell {count}>", "eval"), ns) if last is not None else None
            outs = []
            if buf.getvalue():
                outs.append({"output_type": "stream", "name": "stdout", "text": buf.getvalue()})
            outs.extend(outputs_of_cell)
            ends_with_semicolon = cell["source"].rstrip().endswith(";")
            if value is not None and not ends_with_semicolon:
                outs.append({"output_type": "execute_result", "execution_count": count, "data": rich(value), "metadata": {}})
            for num in plt.get_fignums():  # figures made in this cell, as the inline backend would show them
                fig = plt.figure(num)
                png = io.BytesIO()
                fig.savefig(png, format="png", dpi=110, bbox_inches="tight")
                outs.append({"output_type": "display_data", "data": {"image/png": base64.b64encode(png.getvalue()).decode("ascii"), "text/plain": f"<Figure size {fig.get_size_inches()[0]:.0f}x{fig.get_size_inches()[1]:.0f} with {len(fig.axes)} Axes>"}, "metadata": {}})
                plt.close(fig)
            cell["outputs"] = outs
            cell["execution_count"] = count
    finally:
        os.chdir(cwd)


OUT.parent.mkdir(exist_ok=True)
if "--execute" in sys.argv[1:]:
    execute(nb)
OUT.write_text(json.dumps(nb, indent=1))
n_out = sum(len(c.get("outputs", [])) for c in cells if c["cell_type"] == "code")
print("wrote", OUT, f"({len(cells)} cells, {n_out} outputs)")

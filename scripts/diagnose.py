"""Simulate one event and print what the relay decides — or run the relay on your own record.

    python scripts/diagnose.py --kind BCG --distance 35 --r-arc 0.3 --r-ground 2 --inception 70
    python scripts/diagnose.py --csv my_record.csv --fs 4000     # columns: Va,Vb,Vc,Ia,Ib,Ic (V and A, instantaneous)

For a CSV the relay needs no fault time: the disturbance detector finds it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tlf import CLASSES, GROUPS  # noqa: E402
from tlf.classical import RULES  # noqa: E402
from tlf.features import analyse_record  # noqa: E402
from tlf.locate import reactance_distance_km, takagi_distance_km  # noqa: E402
from tlf.models import decode  # noqa: E402
from tlf.simulate import FaultSpec, Record, SystemParams, add_noise, simulate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Transmission-line fault relay: detect, classify, locate.")
    ap.add_argument("--kind", default="AG", choices=CLASSES)
    ap.add_argument("--distance", type=float, default=60.0, help="km from the relay")
    ap.add_argument("--r-arc", type=float, default=0.5, help="arc resistance per phase (ohm)")
    ap.add_argument("--r-ground", type=float, default=5.0, help="ground resistance (ohm)")
    ap.add_argument("--inception", type=float, default=90.0, help="phase-A voltage angle at the fault (deg)")
    ap.add_argument("--load-step", type=float, default=100.0, help="MW switched on for --kind none")
    ap.add_argument("--delta", type=float, default=15.0, help="load angle (deg)")
    ap.add_argument("--snr", type=float, default=40.0, help="measurement noise SNR (dB)")
    ap.add_argument("--csv", type=str, default=None, help="your own record with columns Va,Vb,Vc,Ia,Ib,Ic")
    ap.add_argument("--fs", type=float, default=4000.0, help="sampling rate of the CSV (Hz)")
    args = ap.parse_args()

    system = SystemParams(delta_deg=args.delta, fs=args.fs)
    if args.csv:
        df = pd.read_csv(args.csv)
        cols = ["Va", "Vb", "Vc", "Ia", "Ib", "Ic"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            ap.error(f"missing columns: {missing}")
        n = len(df)
        rec = Record(t=np.arange(n) / args.fs, v=df[cols[:3]].to_numpy(float), i=df[cols[3:]].to_numpy(float), fault_index=0, spec=FaultSpec(kind="none"), system=system)
        truth = None
    else:
        spec = FaultSpec(kind=args.kind, distance_km=args.distance, r_arc=args.r_arc, r_ground=args.r_ground, inception_deg=args.inception,
                         load_step_mw=args.load_step if args.kind == "none" else 0.0)
        rec = add_noise(simulate(spec, system, post_cycles=5), args.snr, seed=1)
        truth = args.kind

    an = analyse_record(rec, 1.0)
    f = an.features
    print(f"Disturbance detector : {'tripped at %.1f ms' % (an.trigger_index / system.fs * 1e3) if an.triggered else 'quiet'}")
    a21 = np.degrees(np.arctan2(f["sin_a21"], f["cos_a21"]))
    print(f"Sequence currents    : |I0| = {abs(an.seq_I[0]):.0f} A, |I1| = {abs(an.seq_I[1]):.0f} A, |I2| = {abs(an.seq_I[2]):.0f} A, angle(I2/dI1) = {a21:+.0f} deg, residual ratio = {f['ires']:.2f}")
    print()
    bundle = joblib.load(ROOT / "models" / "classifiers.joblib")
    X = pd.DataFrame([f])[bundle["feature_names"]]
    decisions = {name: rule(f) for name, rule in RULES.items()}
    decisions.update({name: decode(pipe.predict(X))[0] for name, pipe in bundle["models"].items()})
    for name, d in decisions.items():
        flag = "" if truth is None else ("  OK" if d == truth else f"  WRONG (truth {truth})")
        print(f"{name:28s}: {d:5s} ({GROUPS[d]}){flag}")
    best = decisions[bundle["best"]]
    if best != "none":
        an3 = analyse_record(rec, 3.0)
        loc = joblib.load(ROOT / "models" / "locator.joblib")
        onehot = pd.get_dummies(pd.Categorical([best], categories=CLASSES)).astype(float)
        L = pd.concat([pd.DataFrame([an3.features]), onehot], axis=1)[loc["columns"]]
        print()
        print(f"Fault location (3-cycle window), class {best}:")
        for name, d in [("Reactance method", reactance_distance_km(an3, best, system)), ("Takagi method", takagi_distance_km(an3, best, system)),
                        ("ML regressor", float(loc["locator"].predict(L)[0]))]:
            print(f"  {name:18s} {d:6.1f} km" + (f"   (true {args.distance:.1f} km)" if truth else ""))


if __name__ == "__main__":
    main()

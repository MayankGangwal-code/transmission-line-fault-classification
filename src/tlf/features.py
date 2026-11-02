"""The relay's decision quantities for one event.

``analyse_record`` mimics what a numerical relay does:

1. run the disturbance detector on the current samples,
2. take the pre-fault phasors from the full cycle that ends at the trigger,
3. take the post-fault phasors from the full cycle that ends ``decision_cycles``
   after the trigger (the "decision window"; a shorter decision time means a
   window that still contains pre-fault samples),
4. compute sequence components, superimposed ("delta") quantities and the
   six loop impedances a distance relay uses.

Everything is normalised to fixed bases (nominal phase voltage, a nominal
current) so that the feature vector is a per-unit description of the event.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dsp import angle_deg, detect_disturbance, dft_phasor, sequence_components, wrap_deg
from .simulate import Record, SystemParams

I_NOM = 1000.0  # A peak, current base (the line carries ~700 A at the default operating point)
PHASE_NAMES = ["a", "b", "c"]
LOOPS = ["ag", "bg", "cg", "ab", "bc", "ca"]


def k0_factor(system: SystemParams) -> complex:
    """Residual compensation factor k0 = (Z0 - Z1) / (3 Z1) of the line."""
    line, w = system.line, system.omega
    z1 = line.R1 + 1j * w * line.L1
    z0 = line.R0 + 1j * w * line.L0
    return (z0 - z1) / (3 * z1)


def loop_impedances(V: np.ndarray, I: np.ndarray, system: SystemParams) -> dict[str, complex]:
    """Apparent impedances of the three ground loops and three phase loops (ohm)."""
    k0 = k0_factor(system)
    i_res = I.sum()  # 3 I0
    Z = {}
    for k, name in enumerate(PHASE_NAMES):
        Z[f"{name}g"] = V[k] / (I[k] + k0 * i_res) if abs(I[k] + k0 * i_res) > 1e-9 else np.inf + 0j
    for (p, q), name in zip([(0, 1), (1, 2), (2, 0)], ["ab", "bc", "ca"]):
        dI = I[p] - I[q]
        Z[name] = (V[p] - V[q]) / dI if abs(dI) > 1e-9 else np.inf + 0j
    return Z


@dataclass
class Analysis:
    triggered: bool
    trigger_index: int | None
    V_pre: np.ndarray
    I_pre: np.ndarray
    V: np.ndarray
    I: np.ndarray
    seq_V: np.ndarray  # [V0, V1, V2]
    seq_I: np.ndarray  # [I0, I1, I2]
    dI: np.ndarray  # superimposed phase currents I - I_pre
    dI1: complex  # superimposed positive-sequence current
    Z: dict[str, complex]
    features: dict[str, float]


def analyse_record(rec: Record, decision_cycles: float = 1.0, det_threshold: float = 0.15, trigger_index: int | None = None) -> Analysis:
    """Detector + phasors + features for one record (see module docstring)."""
    sysp = rec.system
    n = sysp.samples_per_cycle
    v_nom = sysp.v_phase_peak
    trig = trigger_index if trigger_index is not None else detect_disturbance(rec.i, n, I_NOM, threshold=det_threshold)
    triggered = trig is not None
    if trig is None:
        # nothing detected: evaluate the last full cycle against the one before it
        trig = len(rec.t) - int(round(decision_cycles * n))
        trig = max(trig, n)
    end_post = int(round(trig + decision_cycles * n))
    end_post = min(max(end_post, n), len(rec.t))
    end_pre = max(min(trig, len(rec.t) - 1), n)

    V_pre, I_pre = dft_phasor(rec.v, end_pre, n), dft_phasor(rec.i, end_pre, n)
    V, I = dft_phasor(rec.v, end_post, n), dft_phasor(rec.i, end_post, n)
    # refer both windows to the same time origin (the window start differs by end_post - end_pre samples)
    shift = np.exp(1j * 2 * np.pi * (end_post - end_pre) / n)
    V_pre, I_pre = V_pre * shift, I_pre * shift

    sV, sI = sequence_components(V), sequence_components(I)
    sI_pre = sequence_components(I_pre)
    dI = I - I_pre
    dI1 = sI[1] - sI_pre[1]
    Z = loop_impedances(V, I, sysp)
    line = sysp.line
    z_line = abs((line.R1 + 1j * sysp.omega * line.L1) * line.length_km)
    x_line = sysp.omega * line.L1 * line.length_km

    f: dict[str, float] = {"triggered": float(triggered)}
    for k, p in enumerate(PHASE_NAMES):
        f[f"v_{p}"] = abs(V[k]) / v_nom
        f[f"i_{p}"] = abs(I[k]) / I_NOM
        f[f"di_{p}"] = abs(dI[k]) / I_NOM
        f[f"dv_{p}"] = abs(V[k] - V_pre[k]) / v_nom
        f[f"ri_{p}"] = abs(I[k]) / max(abs(I_pre[k]), 1e-6)
    dmax = max(abs(dI).max(), 1e-9)
    for k, p in enumerate(PHASE_NAMES):
        f[f"dir_{p}"] = abs(dI[k]) / dmax  # share of the largest superimposed current
    for k, name in enumerate(["0", "1", "2"]):
        f[f"v{name}"] = abs(sV[k]) / v_nom
        f[f"i{name}"] = abs(sI[k]) / I_NOM
    i1 = max(abs(sI[1]), 1e-9)
    f["r_i0"] = abs(sI[0]) / i1
    f["r_i2"] = abs(sI[2]) / i1
    f["r_v0"] = abs(sV[0]) / max(abs(sV[1]), 1e-9)
    f["r_v2"] = abs(sV[2]) / max(abs(sV[1]), 1e-9)
    f["di1"] = abs(dI1) / I_NOM
    f["ri1"] = abs(sI[1]) / max(abs(sI_pre[1]), 1e-6)  # positive-sequence current rise
    f["r_i0_di1"] = abs(sI[0]) / max(abs(dI1), 1e-9)
    f["r_i2_di1"] = abs(sI[2]) / max(abs(dI1), 1e-9)
    f["ires"] = abs(I.sum()) / max(abs(I).max(), 1e-9)  # residual current relative to the largest phase current
    for name, z in [("a20", sI[2] / sI[0] if abs(sI[0]) > 1e-9 else 0j), ("a21", sI[2] / dI1 if abs(dI1) > 1e-9 else 0j),
                    ("a01", sI[0] / dI1 if abs(dI1) > 1e-9 else 0j)]:
        ang = np.angle(z) if z != 0 else 0.0
        f[f"cos_{name}"], f[f"sin_{name}"] = float(np.cos(ang)), float(np.sin(ang))
    for name in LOOPS:
        z = Z[name]
        if np.isfinite(z):
            f[f"zmag_{name}"] = min(abs(z) / z_line, 10.0)
            f[f"x_{name}"] = float(np.clip(z.imag / x_line, -10.0, 10.0))
            f[f"r_{name}"] = float(np.clip(z.real / z_line, -10.0, 10.0))
        else:
            f[f"zmag_{name}"], f[f"x_{name}"], f[f"r_{name}"] = 10.0, 10.0, 10.0
    return Analysis(triggered, trig if triggered else None, V_pre, I_pre, V, I, sV, sI, dI, dI1, Z, f)


def feature_names() -> list[str]:
    """Feature names in the order ``analyse_record`` produces them."""
    from .simulate import FaultSpec, simulate

    rec = simulate(FaultSpec(kind="AG", distance_km=50), post_cycles=2)
    return list(analyse_record(rec).features)


def features_table(records: list[Record], decision_cycles: float = 1.0, det_threshold: float = 0.15) -> pd.DataFrame:
    rows = [analyse_record(r, decision_cycles, det_threshold).features for r in records]
    return pd.DataFrame(rows)

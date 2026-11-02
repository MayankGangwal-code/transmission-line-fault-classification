"""Impedance-based fault location (the "simple reactance" method).

A distance relay measures the apparent impedance of the faulted loop:

* ground fault on phase a:   Z = V_a / (I_a + k0 · 3 I0)     k0 = (Z0 − Z1) / (3 Z1)
* phase-to-phase fault a-b:  Z = (V_a − V_b) / (I_a − I_b)

For a bolted fault Z equals the line impedance up to the fault, so the
distance is d = Im(Z) / X1 (X1 = positive-sequence reactance per km).  Using
only the reactance makes the estimate insensitive to the (resistive) arc —
as long as the current fed from the remote end is in phase with the local
current.  With fault resistance *and* remote infeed the reactance method
over- or under-reaches: that error is what the report measures.

The **Takagi** method (Takagi et al., IEEE Trans. PAS, 1982) reduces that
error by projecting on the superimposed ("pure fault") current ΔI instead
of the total current:   d = Im(V · ΔI*) / Im(z1 · I_loop · ΔI*).
"""
from __future__ import annotations

import numpy as np

from . import CLASSES, PHASES
from .features import Analysis
from .simulate import SystemParams

LOOP_FOR_CLASS = {
    "AG": "ag", "BG": "bg", "CG": "cg", "AB": "ab", "BC": "bc", "CA": "ca",
    "ABG": "ab", "BCG": "bc", "CAG": "ca", "ABC": "ab",
}


def _loop(an: Analysis, fault_class: str, system: SystemParams):
    """(V_loop, I_loop, ΔI_loop) of the loop a relay would use for this fault class."""
    from .features import k0_factor

    loop = LOOP_FOR_CLASS[fault_class]
    idx = {"a": 0, "b": 1, "c": 2}
    if loop.endswith("g"):
        k = idx[loop[0]]
        i_res = an.I.sum()
        return an.V[k], an.I[k] + k0_factor(system) * i_res, an.dI[k]
    p, q = idx[loop[0]], idx[loop[1]]
    return an.V[p] - an.V[q], an.I[p] - an.I[q], an.dI[p] - an.dI[q]


def takagi_distance_km(an: Analysis, fault_class: str, system: SystemParams) -> float:
    """Takagi one-ended fault location; NaN for 'none'."""
    if fault_class not in LOOP_FOR_CLASS:
        return float("nan")
    v, i, di = _loop(an, fault_class, system)
    z1 = system.line.R1 + 1j * system.omega * system.line.L1  # ohm per km
    den = np.imag(z1 * i * np.conj(di))
    if abs(den) < 1e-12:
        return float("nan")
    d = np.imag(v * np.conj(di)) / den
    return float(np.clip(d, -0.5 * system.line.length_km, 1.5 * system.line.length_km))


def reactance_distance_km(an: Analysis, fault_class: str, system: SystemParams) -> float:
    """Distance to the fault from the loop reactance; NaN for 'none'."""
    if fault_class not in LOOP_FOR_CLASS:
        return float("nan")
    z = an.Z[LOOP_FOR_CLASS[fault_class]]
    if not np.isfinite(z):
        return float("nan")
    x1 = system.omega * system.line.L1  # ohm per km
    return float(np.clip(z.imag / x1, -0.5 * system.line.length_km, 1.5 * system.line.length_km))

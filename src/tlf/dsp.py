"""Relay signal processing: phasors, symmetrical components, disturbance detection."""
from __future__ import annotations

import numpy as np

A_OP = np.exp(1j * 2 * np.pi / 3)
# Fortescue transform: [x0, x1, x2] = T @ [xa, xb, xc]
T_SEQ = (1.0 / 3.0) * np.array([[1, 1, 1], [1, A_OP, A_OP**2], [1, A_OP**2, A_OP]])
T_SEQ_INV = np.array([[1, 1, 1], [1, A_OP**2, A_OP], [1, A_OP, A_OP**2]])


def dft_phasor(x: np.ndarray, end: int, n: int) -> np.ndarray:
    """Full-cycle DFT phasor of the fundamental from the window ending at sample ``end`` (exclusive).

    ``x`` has shape (samples, channels); ``n`` samples per cycle.  Returns the
    complex peak phasor per channel, referenced to the start of the window:
    x(t) = Re( X e^{jωt} ).  The DC offset and harmonics (2nd, 3rd ...) are
    rejected exactly by the full-cycle window; the decaying DC component of a
    fault current is attenuated but not removed, which is realistic.
    """
    seg = x[end - n:end]
    k = np.arange(n)[:, None]
    return (2.0 / n) * np.sum(seg * np.exp(-1j * 2 * np.pi * k / n), axis=0)


def phasor_track(x: np.ndarray, n: int) -> np.ndarray:
    """Sliding full-cycle DFT phasor for every sample (NaN before the first full window)."""
    out = np.full(x.shape, np.nan, dtype=complex)
    for end in range(n, len(x) + 1):
        out[end - 1] = dft_phasor(x, end, n)
    return out


def sequence_components(ph: np.ndarray) -> np.ndarray:
    """[zero, positive, negative] sequence phasors from phase phasors [a, b, c]."""
    return T_SEQ @ np.asarray(ph, dtype=complex)


def angle_deg(z: complex) -> float:
    return float(np.rad2deg(np.angle(z)))


def wrap_deg(a: float) -> float:
    return float((a + 180.0) % 360.0 - 180.0)


def detect_disturbance(i: np.ndarray, n: int, i_ref: float, threshold: float = 0.15, consecutive: int = 3) -> int | None:
    """Superimposed-current (delta) disturbance detector.

    Compares every sample with the sample one cycle earlier: |i[k] - i[k-n]|.
    Triggers at the first sample where the delta of any phase exceeds
    ``threshold * i_ref`` for ``consecutive`` samples in a row — the same idea
    as the "delta" starting element of a numerical relay.  Returns the trigger
    index or None.
    """
    delta = np.abs(i[n:] - i[:-n])  # index k corresponds to sample k + n
    hit = (delta > threshold * i_ref).any(axis=1)
    run = 0
    for k, h in enumerate(hit):
        run = run + 1 if h else 0
        if run >= consecutive:
            return int(k + n - consecutive + 1)
    return None

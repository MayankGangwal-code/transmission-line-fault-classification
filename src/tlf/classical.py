"""Rule-based fault classifiers.

1. ``magnitude_rule`` — the "overcurrent" intuition every student has: the
   phases whose current jumped are the faulted phases; ground is involved if
   residual current (3 I0) flows.

2. ``sequence_rule`` — the phase-selection logic of numerical distance
   relays, built on symmetrical components (Fortescue, 1918) and the fault
   boundary conditions:

   * angle between negative- and (superimposed) positive-sequence current, ∠(I2/ΔI1):
       AG → 0°,  BG → +120°,  CG → −120°        (single line-to-ground)
       BC, BCG → 180°,  AB, ABG → +60°,  CA, CAG → −60°   (phase-to-phase)
   * residual current 3 I0 tells LL from LLG,
   * the classical ground selector ∠(I2/I0) (AG → 0°, BG → −120°, CG → +120°) is
     kept for the figures: it works for single-phase faults but drifts for
     double-line-to-ground faults with ground resistance,
   * a balanced event (no I2, no I0) is a three-phase fault when the measured
     phase-loop impedances lie inside an extended zone of the line — the
     distance-relay criterion; a load change leaves them at the load impedance.

   The derivation is in docs/theory.md.  The thresholds are relay settings
   derived from the measurement model and the line, not fitted values: six
   60°-wide sectors (the nearest centre wins, i.e. ±30° around each), a
   ground pick-up of 1.5 × the spurious residual that ±2 % ratio errors can
   produce, a zone reach of 160 % of the line impedance.  Each was checked for
   margin on the *training* events only (see the comments below).

Both work on the feature dictionary produced by ``features.analyse_record``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import CLASSES

PHASES = ["a", "b", "c"]
PAIR_CLASS = {("a", "b"): "AB", ("b", "c"): "BC", ("a", "c"): "CA"}
PAIR_CLASS_G = {("a", "b"): "ABG", ("b", "c"): "BCG", ("a", "c"): "CAG"}

# thresholds (per unit of I_NOM / ratios)
DETECT_DI = 0.20  # largest superimposed current must exceed 20 % of I_NOM
SHARE = 0.5  # a phase is "faulted" when its ΔI is at least 50 % of the largest ΔI
# residual current above this fraction of the largest phase current → ground fault.  Two phases with
# opposite ±2 % ratio errors produce a spurious residual of up to 4 % of the phase current; the pick-up is
# 1.5 × that.  (Training events: LL faults reach 3.8 %, LLG faults at the far end through > 30 Ω of
# ground resistance go down to 5.7 % because most of their zero-sequence current flows from the remote end.)
GROUND_RATIO = 0.06
# the six sector centres of ∠(I2/ΔI1) are 60° apart; an angle is assigned to the nearest centre, so each
# sector is ±30° wide.  On the training events single-phase and phase-to-phase faults sit within 3° of
# their centre and double-line-to-ground faults drift by up to 17° (the ground-resistance path rotates I2
# slightly), leaving a margin of 13° to the next sector.
SECTOR = 30.0  # half-width of the angle sectors (degrees) = half the centre spacing
SECTOR_CENTRES = {"AG": 0.0, "BG": 120.0, "CG": -120.0, "BC": 180.0, "AB": 60.0, "CA": -60.0}
# |I2| / |ΔI1| and |I0| / |ΔI1| below this → balanced event.  Unequal decaying DC offsets in the three phases
# leak into the one-cycle DFT and give a three-phase fault a spurious |I2| / |ΔI1| of up to 0.15 (training
# events); the smallest value of any unbalanced fault is 0.46.  The threshold sits in the middle of that gap.
BALANCED_I2 = 0.30
# a balanced event is a three-phase fault when all three phase-loop impedances lie inside 160 % of the line
# impedance (an extended zone, as a distance relay would check); a load change keeps them at the load
# impedance, which is at least 2.3 × the line impedance for the largest load step in the training events
ABC_ZONE = 1.6


def _ang(f: dict, name: str) -> float:
    return float(np.rad2deg(np.arctan2(f[f"sin_{name}"], f[f"cos_{name}"])))


def _angle_distance(a: float, b: float) -> float:
    """Unsigned distance between two angles in degrees (0–180)."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def nearest_sector(angle: float) -> str:
    """The sector centre closest to ``angle``: AG, BG, CG, BC, AB or CA (±30° each)."""
    return min(SECTOR_CENTRES, key=lambda k: _angle_distance(angle, SECTOR_CENTRES[k]))


def magnitude_rule(f: dict) -> str:
    """Phases with a large current change are faulted; residual current means ground."""
    dmax = max(f["di_a"], f["di_b"], f["di_c"])
    if not f["triggered"] or dmax < DETECT_DI:
        return "none"
    faulted = [p for p in PHASES if f[f"dir_{p}"] >= SHARE]
    ground = f["ires"] > GROUND_RATIO
    if len(faulted) == 3:
        return "ABC"
    if len(faulted) == 2:
        key = tuple(sorted(faulted))
        return (PAIR_CLASS_G if ground else PAIR_CLASS)[key]
    if len(faulted) == 1:
        return faulted[0].upper() + "G" if ground else "none"
    return "none"


def sequence_rule(f: dict) -> str:
    """Relay-style phase selection from the angles between sequence currents.

    Primary selector: the angle between the negative-sequence current and the
    superimposed positive-sequence current, ∠(I2/ΔI1), which has six sectors
    60° apart (nearest centre wins):
    0° → AG, +120° → BG, −120° → CG, 180° → BC/BCG, +60° → AB/ABG, −60° → CA/CAG.
    Residual current then separates LL from LLG.  A balanced event (no I2) is
    a three-phase fault if the phase-loop impedances lie inside the extended
    zone of the line, otherwise a load change.
    """
    dmax = max(f["di_a"], f["di_b"], f["di_c"])
    if not f["triggered"] or dmax < DETECT_DI:
        return "none"
    balanced = f["r_i2_di1"] < BALANCED_I2 and f["r_i0_di1"] < BALANCED_I2
    if balanced:
        inside_zone = max(f["zmag_ab"], f["zmag_bc"], f["zmag_ca"]) < ABC_ZONE
        return "ABC" if inside_zone else "none"
    sector = nearest_sector(_ang(f, "a21"))  # ∠(I2 / ΔI1)
    if sector in ("AG", "BG", "CG"):
        return sector  # a single-phase fault always involves ground (its residual may be small for high R_g)
    ground = f["ires"] > GROUND_RATIO
    return sector + "G" if ground else sector


def sequence_angle_i2_i0(f: dict) -> float:
    """The classical ground-fault selector angle ∠(I2/I0): AG ≈ 0°, BG ≈ −120°, CG ≈ +120°
    (used in the figures; double-line-to-ground faults drift away from these sectors as
    the ground resistance grows, which is why ``sequence_rule`` keys on ∠(I2/ΔI1))."""
    return _ang(f, "a20")


RULES = {"Overcurrent rule": magnitude_rule, "Sequence-angle relay logic": sequence_rule}


def apply_rule(rule, table: pd.DataFrame) -> np.ndarray:
    return np.array([rule(row) for row in table.to_dict(orient="records")], dtype=object).astype(str)


def to_group(labels) -> np.ndarray:
    from . import GROUPS

    return np.array([GROUPS[str(l)] for l in labels])

"""The simulator and the relay algorithms checked against textbook results.

Run with:  pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tlf import CLASSES  # noqa: E402
from tlf.classical import magnitude_rule, sequence_rule  # noqa: E402
from tlf.dsp import A_OP, detect_disturbance, dft_phasor, sequence_components, T_SEQ, T_SEQ_INV  # noqa: E402
from tlf.features import analyse_record, k0_factor  # noqa: E402
from tlf.locate import reactance_distance_km, takagi_distance_km  # noqa: E402
from tlf.simulate import FaultSpec, SystemParams, add_noise, fault_conductance, sequence_impedances, simulate  # noqa: E402

SYS = SystemParams()
N = SYS.samples_per_cycle
RADIAL = SystemParams(r_sc_mva=1e-3, e_r_pu=0.0, delta_deg=0.0)  # remote source removed -> single-ended feed


def steady_phasors(rec, cycles_after_fault=6):
    end = rec.fault_index + cycles_after_fault * N
    return dft_phasor(rec.v, end, N), dft_phasor(rec.i, end, N)


# --------------------------------------------------------------------------- #
# signal processing
# --------------------------------------------------------------------------- #
def test_dft_phasor_recovers_amplitude_and_phase():
    t = np.arange(400) / SYS.fs
    x = 100 * np.cos(2 * np.pi * 50 * t + np.deg2rad(37)) + 20 + 15 * np.cos(2 * np.pi * 150 * t)  # fundamental + DC + 3rd harmonic
    X = dft_phasor(x[:, None], N, N)[0]
    assert abs(X) == pytest.approx(100, rel=1e-6)
    assert np.rad2deg(np.angle(X)) == pytest.approx(37, abs=1e-6)  # DC and harmonics rejected exactly


def test_symmetrical_components_roundtrip_and_balanced_set():
    ph = np.array([1.0, A_OP**2, A_OP]) * 230  # balanced abc
    s = sequence_components(ph)
    assert abs(s[0]) < 1e-9 and abs(s[2]) < 1e-9 and abs(s[1] - 230) < 1e-9
    rnd = np.array([3 + 1j, -2 + 0.5j, 0.7 - 2j])
    assert np.allclose(T_SEQ_INV @ (T_SEQ @ rnd), rnd)


def test_disturbance_detector_triggers_only_on_change():
    t = np.arange(10 * N) / SYS.fs
    i = np.stack([700 * np.cos(2 * np.pi * 50 * t - k * 2 * np.pi / 3) for k in range(3)], axis=1)
    assert detect_disturbance(i, N, 1000.0) is None
    i2 = i.copy()
    i2[5 * N:, 0] *= 4.0
    trig = detect_disturbance(i2, N, 1000.0)
    assert trig is not None and 5 * N <= trig <= 5 * N + 6


# --------------------------------------------------------------------------- #
# simulator versus hand calculation
# --------------------------------------------------------------------------- #
def test_prefault_is_balanced_steady_state_with_power_flow():
    rec = simulate(FaultSpec(kind="none"), SYS)
    V, I = dft_phasor(rec.v, N, N), dft_phasor(rec.i, N, N)
    sV, sI = sequence_components(V), sequence_components(I)
    assert abs(sV[2]) / abs(sV[1]) < 1e-3 and abs(sI[2]) / abs(sI[1]) < 1e-3  # balanced
    assert abs(V[0]) == pytest.approx(SYS.v_phase_peak, rel=0.02)
    p_mw = 0.5 * np.real(np.sum(V * np.conj(I))) / 1e6  # peak phasors: P = sum over phases of Re(V I*) / 2
    # E_s E_r sin(delta) / X with X = X_s + X_line + X_r = 9.6 + 39.9 + 16.0 ohm gives ~187 MW
    assert p_mw == pytest.approx(187.0, rel=0.05)
    assert np.abs(rec.v[N:2 * N] - rec.v[:N]).max() < 1e-6  # no start-up transient


@pytest.mark.parametrize("d", [20.0, 60.0, 95.0])
def test_three_phase_fault_current_matches_formula(d):
    rec = simulate(FaultSpec(kind="ABC", distance_km=d, r_arc=1e-4), RADIAL, post_cycles=8)
    _, I = steady_phasors(rec)
    z = sequence_impedances(RADIAL, d)["Z1_total"]
    assert abs(I).mean() == pytest.approx(RADIAL.v_phase_peak / abs(z), rel=0.01)


@pytest.mark.parametrize("d", [20.0, 60.0, 95.0])
def test_single_line_to_ground_fault_current_matches_formula(d):
    """I_a = 3 E / (Z1 + Z2 + Z0) with Z2 = Z1 (radial feed, bolted fault)."""
    rec = simulate(FaultSpec(kind="AG", distance_km=d, r_arc=1e-4, r_ground=1e-4), RADIAL, post_cycles=8)
    _, I = steady_phasors(rec)
    z = sequence_impedances(RADIAL, d)
    expected = 3 * RADIAL.v_phase_peak / abs(2 * z["Z1_total"] + z["Z0_total"])
    assert abs(I[0]) == pytest.approx(expected, rel=0.01)
    assert abs(I[1]) < 0.02 * abs(I[0]) and abs(I[2]) < 0.02 * abs(I[0])  # healthy phases carry only charging current


def test_line_to_line_fault_current_and_no_ground_current():
    rec = simulate(FaultSpec(kind="BC", distance_km=40.0, r_arc=1e-4), RADIAL, post_cycles=8)
    _, I = steady_phasors(rec)
    z = sequence_impedances(RADIAL, 40.0)["Z1_total"]
    assert abs(I[1]) == pytest.approx(np.sqrt(3) * RADIAL.v_phase_peak / abs(2 * z), rel=0.01)
    assert abs(I.sum()) < 1e-3 * abs(I[1])  # no zero-sequence current without ground
    assert abs(I[1] + I[2]) < 0.01 * abs(I[1])  # I_b = -I_c up to the line's charging current


def test_fault_conductance_matrix_structure():
    G = fault_conductance(FaultSpec(kind="BC", r_arc=1.0))
    assert np.allclose(G.sum(axis=0), 0)  # no path to ground
    assert G[1, 1] == pytest.approx(0.5) and G[1, 2] == pytest.approx(-0.5)  # phase-phase resistance 2 r
    G = fault_conductance(FaultSpec(kind="AG", r_arc=1.0, r_ground=2.0))
    assert G[0, 0] == pytest.approx(1 / 3.0) and G[1, 1] == 0  # r + R_g in series
    assert np.allclose(fault_conductance(FaultSpec(kind="none")), 0)


def test_fault_resistance_and_inception_angle_change_the_waveform():
    a = simulate(FaultSpec(kind="AG", distance_km=50, r_arc=0.1, r_ground=0.1, inception_deg=90), SYS)
    b = simulate(FaultSpec(kind="AG", distance_km=50, r_arc=0.1, r_ground=40.0, inception_deg=90), SYS)
    c = simulate(FaultSpec(kind="AG", distance_km=50, r_arc=0.1, r_ground=0.1, inception_deg=0), SYS)
    assert np.abs(a.i[:, 0]).max() > 2 * np.abs(b.i[:, 0]).max()  # ground resistance limits the current
    # inception at the voltage zero crossing (90 deg) gives a DC offset: asymmetric first peaks
    post = slice(a.fault_index, a.fault_index + N)
    asym_a = abs(a.i[post, 0].max() + a.i[post, 0].min()) / np.abs(a.i[post, 0]).max()
    asym_c = abs(c.i[post, 0].max() + c.i[post, 0].min()) / np.abs(c.i[post, 0]).max()
    assert asym_a > 3 * asym_c


# --------------------------------------------------------------------------- #
# relay logic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", [c for c in CLASSES if c != "none"])
def test_sequence_rule_and_magnitude_rule_on_a_clean_fault(kind):
    rec = add_noise(simulate(FaultSpec(kind=kind, distance_km=45.0, r_arc=0.3, r_ground=2.0, inception_deg=30.0), SYS), 40.0, seed=1)
    f = analyse_record(rec, 1.0).features
    assert sequence_rule(f) == kind
    assert magnitude_rule(f) == kind


def test_sequence_angles_match_the_textbook_values():
    """∠(I2/I0) ≈ 0, -120, +120 for AG, BG, CG and ∠(I2/ΔI1) ≈ 180, 60, -60 for BC, AB, CA."""
    for kind, a20 in (("AG", 0), ("BG", -120), ("CG", 120)):
        f = analyse_record(simulate(FaultSpec(kind=kind, distance_km=50, r_arc=0.1, r_ground=0.1), SYS), 1.0).features
        assert np.rad2deg(np.arctan2(f["sin_a20"], f["cos_a20"])) == pytest.approx(a20, abs=15)
    for kind, a21 in (("BC", 180), ("AB", 60), ("CA", -60)):
        f = analyse_record(simulate(FaultSpec(kind=kind, distance_km=50, r_arc=0.1), SYS), 1.0).features
        ang = np.rad2deg(np.arctan2(f["sin_a21"], f["cos_a21"]))
        assert abs((ang - a21 + 180) % 360 - 180) < 15


def test_load_step_is_not_a_fault():
    rec = add_noise(simulate(FaultSpec(kind="none", distance_km=50, load_step_mw=120.0), SYS), 40.0, seed=2)
    f = analyse_record(rec, 1.0).features
    assert f["triggered"] == 1.0  # the detector sees the change ...
    assert sequence_rule(f) == "none"  # ... but the relay logic does not call it a fault


def test_k0_factor_and_locators_exact_for_bolted_faults():
    k0 = k0_factor(SYS)
    line = SYS.line
    z1, z0 = (line.R1 + 1j * SYS.omega * line.L1), (line.R0 + 1j * SYS.omega * line.L0)
    assert k0 == pytest.approx((z0 - z1) / (3 * z1))
    for kind in ("AG", "BC", "ABC"):
        rec = simulate(FaultSpec(kind=kind, distance_km=62.0, r_arc=1e-4, r_ground=1e-4, inception_deg=45.0), SYS, post_cycles=6)
        an3 = analyse_record(rec, 3.0)  # the window the pipeline uses: DC offset mostly decayed
        assert reactance_distance_km(an3, kind, SYS) == pytest.approx(62.0, abs=1.2)
        assert takagi_distance_km(an3, kind, SYS) == pytest.approx(62.0, abs=1.2)
        an5 = analyse_record(rec, 5.0)  # DC offset gone: both locators are exact
        assert reactance_distance_km(an5, kind, SYS) == pytest.approx(62.0, abs=0.2)
        assert takagi_distance_km(an5, kind, SYS) == pytest.approx(62.0, abs=0.2)


def test_takagi_beats_reactance_with_fault_resistance_and_remote_infeed():
    rec = simulate(FaultSpec(kind="AG", distance_km=70.0, r_arc=1.0, r_ground=20.0, inception_deg=10.0), SYS, post_cycles=6)
    an = analyse_record(rec, 3.0)
    e_react = abs(reactance_distance_km(an, "AG", SYS) - 70.0)
    e_takagi = abs(takagi_distance_km(an, "AG", SYS) - 70.0)
    assert e_takagi < e_react

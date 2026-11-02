r"""Time-domain simulator of shunt faults on a two-source transmission line.

System (one line diagram)

    Es ──[Zs]── S ═══ segment 1 (d km) ═══ F ═══ segment 2 (L−d km) ═══ R ──[Zr]── Er
                │  relay measures v_S, i_1   │ fault branch (G_f)            │
               C_S                          C_F                             C_R

* Each line segment is a lumped π-section in the phase domain: 3×3 series
  R and L matrices with mutual coupling, built from the positive- and
  zero-sequence constants (R_s = (R0+2R1)/3, R_m = (R0−R1)/3, same for L and C).
* The two sources are ideal three-phase voltages behind R-L impedances; the
  remote source lags the local one by the load angle δ, which sets the
  pre-fault power flow.
* The fault is a 3×3 conductance matrix at node F: every faulted phase is tied
  to a common fault node through an arc resistance r_arc, and the fault node to
  ground through R_g (ground faults only).
* Between switching instants the whole circuit is a linear time-invariant
  system driven by sinusoidal sources.  The sources are modelled as two
  oscillator states (cos ωt, sin ωt), so the augmented system x' = A x is
  autonomous and is integrated *exactly* with the matrix exponential
  x[k+1] = expm(A Δt) x[k].  The pre-fault steady state is computed in closed
  form, so the record starts without any start-up transient.

Sign convention: currents are positive from S towards R.  Quantities are in
SI units (V, A, s).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.linalg import expm

from . import CLASSES, GROUNDED, PHASE_INDEX, PHASES

TWO_PI = 2.0 * np.pi
A_OP = np.exp(1j * TWO_PI / 3)  # the "a" operator of symmetrical components


# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #
@dataclass
class LineParams:
    """Typical 220 kV single-circuit overhead line (per km)."""

    length_km: float = 100.0
    R1: float = 0.07  # ohm/km positive sequence
    L1: float = 1.27e-3  # H/km
    C1: float = 9.0e-9  # F/km
    R0: float = 0.25  # ohm/km zero sequence
    L0: float = 3.8e-3  # H/km
    C0: float = 5.5e-9  # F/km

    @property
    def X1_per_km(self) -> float:
        return TWO_PI * SystemParams.f0 * self.L1

    def series_matrices(self, km: float) -> tuple[np.ndarray, np.ndarray]:
        """Phase-domain R and L matrices of a segment of ``km`` kilometres."""
        return _seq_to_phase(self.R0, self.R1) * km, _seq_to_phase(self.L0, self.L1) * km

    def shunt_matrix(self, km: float) -> np.ndarray:
        return _seq_to_phase(self.C0, self.C1) * km


@dataclass
class SystemParams:
    f0: float = 50.0  # Hz
    v_ll_kv: float = 220.0  # nominal line-to-line RMS voltage
    fs: float = 4000.0  # sampling rate of the relay (80 samples per cycle at 50 Hz)
    # local source S: short-circuit level and X/R
    s_sc_mva: float = 5000.0
    xr_s: float = 10.0
    # remote source R
    r_sc_mva: float = 3000.0
    xr_r: float = 8.0
    e_s_pu: float = 1.0  # local source voltage magnitude (pu)
    e_r_pu: float = 0.98  # remote source voltage magnitude (pu)
    delta_deg: float = 15.0  # load angle (remote source lags) -> power flows S -> R
    line: LineParams = field(default_factory=LineParams)

    @property
    def v_phase_peak(self) -> float:
        return self.v_ll_kv * 1e3 * np.sqrt(2.0) / np.sqrt(3.0)

    @property
    def omega(self) -> float:
        return TWO_PI * self.f0

    @property
    def samples_per_cycle(self) -> int:
        return int(round(self.fs / self.f0))

    def source_rl(self, which: str) -> tuple[np.ndarray, np.ndarray]:
        """R and L matrices (3×3, decoupled) of a source impedance."""
        mva, xr = (self.s_sc_mva, self.xr_s) if which == "S" else (self.r_sc_mva, self.xr_r)
        z = (self.v_ll_kv * 1e3) ** 2 / (mva * 1e6)  # ohm (positive sequence)
        x = z * xr / np.sqrt(1 + xr**2)
        r = x / xr
        # zero-sequence source impedance taken equal to positive sequence (solidly grounded)
        return np.eye(3) * r, np.eye(3) * x / self.omega


@dataclass
class FaultSpec:
    kind: str = "AG"  # one of CLASSES
    distance_km: float = 50.0  # from the relay (node S)
    r_arc: float = 0.5  # ohm, resistance between each faulted phase and the fault node
    r_ground: float = 5.0  # ohm, fault node to ground (ground faults only)
    inception_deg: float = 0.0  # phase-A voltage angle at the fault instant
    load_step_mw: float = 0.0  # for "none": a balanced resistive load switched on at F instead of a fault
    three_phase_grounded: bool = False  # for "ABC": also tie the fault node to ground (ABCG)


@dataclass
class Record:
    """One simulated event as seen by the relay at S."""

    t: np.ndarray  # s
    v: np.ndarray  # (n, 3) bus voltages at S
    i: np.ndarray  # (n, 3) line currents at S (towards the line)
    fault_index: int  # sample index of the switching instant (the state is continuous there: last pre-fault sample)
    spec: FaultSpec
    system: SystemParams


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _seq_to_phase(z0: float, z1: float) -> np.ndarray:
    """Phase-domain 3×3 matrix from zero- and positive-sequence values."""
    s, m = (z0 + 2 * z1) / 3.0, (z0 - z1) / 3.0
    return np.array([[s, m, m], [m, s, m], [m, m, s]], dtype=float)


def fault_conductance(spec: FaultSpec, system: SystemParams | None = None) -> np.ndarray:
    """3×3 conductance matrix of the fault branch at node F (siemens).

    Faulted phases connect to a common node through r_arc each; that node
    connects to ground through r_ground for ground faults.  The common node is
    eliminated analytically.  For "none" with a load step, a balanced
    resistive load is returned instead.
    """
    G = np.zeros((3, 3))
    if spec.kind == "none":
        if spec.load_step_mw > 0:
            system = system or SystemParams()
            r_load = (system.v_ll_kv * 1e3) ** 2 / (spec.load_step_mw * 1e6)  # per-phase star resistance
            G = np.eye(3) / r_load
        return G
    idx = [PHASE_INDEX[p] for p in PHASES[spec.kind]]
    k = len(idx)
    r = max(spec.r_arc, 1e-4)
    g = 1.0 / r
    grounded = GROUNDED[spec.kind] or (spec.kind == "ABC" and spec.three_phase_grounded)
    # node elimination: I_k = g (V_k - V_N),  V_N = g * sum(V_F) / (k g + g_ground)
    g_ground = 1.0 / max(spec.r_ground, 1e-4) if grounded else 0.0
    denom = k * g + g_ground
    block = g * np.eye(k) - (g * g / denom) * np.ones((k, k))
    for a, ia in enumerate(idx):
        for b, ib in enumerate(idx):
            G[ia, ib] = block[a, b]
    return G


# --------------------------------------------------------------------------- #
# state-space model
# --------------------------------------------------------------------------- #
def _build_system(system: SystemParams, d_km: float, G_f: np.ndarray, phi_s: float):
    """Augmented autonomous state matrix A (23×23) and the output map.

    State x = [i_s(3), v_S(3), i_1(3), v_F(3), i_2(3), v_R(3), i_r(3), c, s]
    with c = cos(ωt), s = sin(ωt).
    """
    line = system.line
    w = system.omega
    L_tot = line.length_km
    d_km = float(np.clip(d_km, 0.5, L_tot - 0.5))
    Rs, Ls = system.source_rl("S")
    Rr, Lr = system.source_rl("R")
    R1, L1 = line.series_matrices(d_km)
    R2, L2 = line.series_matrices(L_tot - d_km)
    C_S = line.shunt_matrix(d_km) / 2.0
    C_F = line.shunt_matrix(d_km) / 2.0 + line.shunt_matrix(L_tot - d_km) / 2.0
    C_R = line.shunt_matrix(L_tot - d_km) / 2.0

    # source phasors: e_k(t) = Vm * Re( E_k e^{jωt} ),  E_k = |E| e^{j(phi - 2πk/3)}
    Vm = system.v_phase_peak
    ang = np.array([0.0, -TWO_PI / 3, TWO_PI / 3])
    E_s = system.e_s_pu * Vm * np.exp(1j * (phi_s + ang))
    E_r = system.e_r_pu * Vm * np.exp(1j * (phi_s - np.deg2rad(system.delta_deg) + ang))

    def src_cols(E: np.ndarray) -> np.ndarray:
        """Map [c, s] -> e(t): e = Re(E) c - Im(E) s."""
        return np.c_[E.real, -E.imag]

    n = 23
    A = np.zeros((n, n))
    sl = {name: slice(3 * k, 3 * k + 3) for k, name in enumerate(["is", "vS", "i1", "vF", "i2", "vR", "ir"])}
    osc = slice(21, 23)
    Ls_i, L1_i, L2_i, Lr_i = (np.linalg.inv(M) for M in (Ls, L1, L2, Lr))
    CS_i, CF_i, CR_i = (np.linalg.inv(M) for M in (C_S, C_F, C_R))

    # Ls di_s/dt = e_s - Rs i_s - v_S
    A[sl["is"], sl["is"]] = -Ls_i @ Rs
    A[sl["is"], sl["vS"]] = -Ls_i
    A[sl["is"], osc] = Ls_i @ src_cols(E_s)
    # C_S dv_S/dt = i_s - i_1
    A[sl["vS"], sl["is"]] = CS_i
    A[sl["vS"], sl["i1"]] = -CS_i
    # L1 di_1/dt = v_S - R1 i_1 - v_F
    A[sl["i1"], sl["vS"]] = L1_i
    A[sl["i1"], sl["i1"]] = -L1_i @ R1
    A[sl["i1"], sl["vF"]] = -L1_i
    # C_F dv_F/dt = i_1 - i_2 - G_f v_F
    A[sl["vF"], sl["i1"]] = CF_i
    A[sl["vF"], sl["i2"]] = -CF_i
    A[sl["vF"], sl["vF"]] = -CF_i @ G_f
    # L2 di_2/dt = v_F - R2 i_2 - v_R
    A[sl["i2"], sl["vF"]] = L2_i
    A[sl["i2"], sl["i2"]] = -L2_i @ R2
    A[sl["i2"], sl["vR"]] = -L2_i
    # C_R dv_R/dt = i_2 - i_r
    A[sl["vR"], sl["i2"]] = CR_i
    A[sl["vR"], sl["ir"]] = -CR_i
    # Lr di_r/dt = v_R - Rr i_r - e_r
    A[sl["ir"], sl["vR"]] = Lr_i
    A[sl["ir"], sl["ir"]] = -Lr_i @ Rr
    A[sl["ir"], osc] = -Lr_i @ src_cols(E_r)
    # oscillator: dc/dt = -ω s, ds/dt = ω c
    A[21, 22] = -w
    A[22, 21] = w
    return A, sl, G_f


def _steady_state(A: np.ndarray, w: float) -> np.ndarray:
    """Exact periodic steady state of x' = A x at t = 0.

    The oscillator states are c = Re(e^{jωt}), s = Re(-j e^{jωt}); the
    remaining states respond as X e^{jωt} with (jω I - A_cc) X = A_co u,
    u = [1, -j].  Returns x(0) = Re([X; u]).
    """
    n = A.shape[0]
    Acc, Aco = A[:21, :21], A[:21, 21:]
    u = np.array([1.0, -1j])
    X = np.linalg.solve(1j * w * np.eye(21) - Acc, Aco @ u)
    x0 = np.empty(n)
    x0[:21] = X.real
    x0[21:] = u.real  # c(0) = 1, s(0) = 0
    return x0


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def simulate(spec: FaultSpec, system: SystemParams | None = None, pre_cycles: float = 2.0, post_cycles: float = 6.0,
             noise_snr_db: float | None = None, seed: int = 0) -> Record:
    """Simulate one event and return what the relay at S measures.

    The record starts in the exact pre-fault steady state; the fault (or load
    step) is switched on after ``pre_cycles`` cycles, at the instant
    t_f = fault_index / fs, when the phase-A source voltage angle equals
    ``spec.inception_deg``.  Sample ``fault_index`` is the (continuous) state
    at the switching instant; the fault acts from the next sample on.
    """
    system = system or SystemParams()
    w, fs = system.omega, system.fs
    dt = 1.0 / fs
    n_pre = int(round(pre_cycles * system.samples_per_cycle))
    n_post = int(round(post_cycles * system.samples_per_cycle))
    n = n_pre + n_post
    t_f = n_pre * dt
    # choose the source phase so that the phase-A voltage angle at t_f equals the inception angle
    phi_s = np.deg2rad(spec.inception_deg) - w * t_f

    A_pre, sl, _ = _build_system(system, spec.distance_km, np.zeros((3, 3)), phi_s)
    A_post, _, _ = _build_system(system, spec.distance_km, fault_conductance(spec, system), phi_s)
    Phi_pre, Phi_post = expm(A_pre * dt), expm(A_post * dt)

    x = _steady_state(A_pre, w)
    X = np.empty((n, A_pre.shape[0]))
    for k in range(n):
        X[k] = x  # state at t = k dt
        x = (Phi_pre if k < n_pre else Phi_post) @ x  # the step from t_f = n_pre dt onwards uses the faulted circuit
    t = np.arange(n) * dt
    v = X[:, sl["vS"]]
    i = X[:, sl["i1"]]

    if noise_snr_db is not None:
        rng = np.random.default_rng(seed)
        for arr in (v, i):
            rms = np.sqrt(np.mean(arr[:n_pre] ** 2, axis=0)).mean() if n_pre > 0 else np.sqrt(np.mean(arr**2))
            sigma = rms / (10 ** (noise_snr_db / 20))
            arr += rng.normal(0.0, sigma, arr.shape)
    return Record(t=t, v=v, i=i, fault_index=n_pre, spec=spec, system=system)


def add_noise(rec: Record, snr_db: float | None, seed: int = 0, ratio_error: float = 0.02) -> Record:
    """Return a copy of the record as a real relay would see it.

    * per-phase CT / VT ratio errors: each channel is scaled by a random gain in
      1 ± ``ratio_error`` (instrument-transformer accuracy class),
    * white measurement noise at ``snr_db`` relative to the pre-fault RMS
      (``None`` or ``inf`` = no noise).
    """
    rng = np.random.default_rng(seed)
    n_pre = rec.fault_index
    v, i = rec.v.copy(), rec.i.copy()
    if ratio_error > 0:
        v *= 1.0 + rng.uniform(-ratio_error, ratio_error, 3)
        i *= 1.0 + rng.uniform(-ratio_error, ratio_error, 3)
    if snr_db is not None and np.isfinite(snr_db):
        for arr in (v, i):
            rms = np.sqrt(np.mean(arr[:n_pre] ** 2, axis=0)).mean()
            arr += rng.normal(0.0, rms / (10 ** (snr_db / 20)), arr.shape)
    return Record(t=rec.t, v=v, i=i, fault_index=rec.fault_index, spec=rec.spec, system=rec.system)


def random_spec(rng: np.random.Generator, kind: str | None = None, system: SystemParams | None = None) -> FaultSpec:
    """Draw a random event: fault type, location, resistances, inception angle."""
    system = system or SystemParams()
    kind = kind or rng.choice(CLASSES)
    L = system.line.length_km
    if kind == "none":
        return FaultSpec(kind="none", distance_km=float(rng.uniform(0.05, 0.95) * L), r_arc=0.0, r_ground=0.0,
                         inception_deg=float(rng.uniform(0, 360)), load_step_mw=float(rng.choice([0.0, rng.uniform(20, 150)])))
    # fault resistance: mostly low, with a tail of high-resistance ground faults
    r_arc = float(10 ** rng.uniform(-2, 0.7))  # 0.01 - 5 ohm
    r_ground = float(10 ** rng.uniform(-1, 1.7)) if GROUNDED[kind] else 0.0  # 0.1 - 50 ohm
    return FaultSpec(kind=kind, distance_km=float(rng.uniform(0.02, 0.98) * L), r_arc=r_arc, r_ground=r_ground,
                     inception_deg=float(rng.uniform(0, 360)), three_phase_grounded=bool(kind == "ABC" and rng.random() < 0.5))


def random_system(rng: np.random.Generator, base: SystemParams | None = None, spread: float = 0.3) -> SystemParams:
    """Perturb the operating point: load angle, source strengths and voltages."""
    base = base or SystemParams()
    return replace(
        base,
        delta_deg=float(rng.uniform(5, 30)),
        s_sc_mva=float(base.s_sc_mva * rng.uniform(1 - spread, 1 + spread)),
        r_sc_mva=float(base.r_sc_mva * rng.uniform(1 - spread, 1 + spread)),
        e_s_pu=float(rng.uniform(0.97, 1.05)),
        e_r_pu=float(rng.uniform(0.95, 1.03)),
    )


def sequence_impedances(system: SystemParams, d_km: float) -> dict:
    """Positive/zero-sequence impedances (ohm) seen from S to the fault point (source + line)."""
    line, w = system.line, system.omega
    Rs, Ls = system.source_rl("S")
    zs = Rs[0, 0] + 1j * w * Ls[0, 0]
    z1 = (line.R1 + 1j * w * line.L1) * d_km
    z0 = (line.R0 + 1j * w * line.L0) * d_km
    return {"Zs1": zs, "Zs0": zs, "Zline1": z1, "Zline0": z0, "Z1_total": zs + z1, "Z0_total": zs + z0}

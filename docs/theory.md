# Theory cheat-sheet — faults, symmetrical components, relaying and fault location

Everything the project uses, in the order a relay uses it. Numbers refer to the project's reference system
(220 kV, 50 Hz, 100 km) unless stated otherwise.

---

## 1. The system

```
 Es ──[Zs]── S ═══════ d km ═══════ F ═══════ (100 − d) km ═══════ R ──[Zr]── Er
              │ relay: v(t), i(t)   │ fault network
```

| item | value | note |
|---|---|---|
| nominal voltage | 220 kV line-to-line → 127 kV phase RMS, 179.6 kV phase peak | the per-unit base of every voltage feature |
| line, positive sequence | R₁ = 0.07 Ω/km, L₁ = 1.27 mH/km (X₁ = 0.399 Ω/km), C₁ = 9 nF/km | |Z₁| = 0.405 Ω/km ∠80°, 40.5 Ω for the whole line |
| line, zero sequence | R₀ = 0.25 Ω/km, L₀ = 3.8 mH/km (X₀ = 1.19 Ω/km), C₀ = 5.5 nF/km | Z₀ ≈ 3 Z₁ — typical for an overhead line with earth return |
| local source S | 5,000 MVA, X/R = 10 → Zₛ = 9.68 Ω | the "strong" end where the relay is |
| remote source R | 3,000 MVA, X/R = 8 → Zᵣ = 16.1 Ω | feeds the fault from the far side (remote infeed) |
| load angle δ | 15° (5–30° in the data) | P ≈ EₛEᵣ sin δ / (Xₛ + X_line + Xᵣ) ≈ 187 MW, ≈ 500 A RMS |
| relay sampling | 4 kHz = 80 samples per cycle | typical numerical-relay rate |
| residual compensation factor | k₀ = (Z₀ − Z₁)/(3 Z₁) = 0.67 ∠−3° | used in the ground-loop impedance |

Surge impedance Z_c = √(L₁/C₁) ≈ 376 Ω, surge-impedance loading ≈ 129 MW; charging current of the whole line
≈ 36 A RMS per phase (that is why the healthy phases of a single-line-to-ground fault are not exactly zero
in the tests).

---

## 2. Symmetrical components (Fortescue, 1918)

Any three unbalanced phasors are the sum of three balanced sets:

```
[I0]       1  [1  1   1 ] [Ia]              [Ia]   [1  1   1 ] [I0]
[I1]  =  ─── · [1  a   a²] [Ib]              [Ib] = [1  a²  a ] [I1]          a = e^{j120°}
[I2]       3  [1  a²  a ] [Ic]              [Ic]   [1  a   a²] [I2]
```

* **zero sequence (I₀)**: three equal, in-phase phasors — can only flow if there is a path through the ground
  (or the neutral), so for a shunt fault **I₀ ≠ 0 ⇔ ground is involved**. The residual current
  I_a + I_b + I_c = 3 I₀. (Unbalanced load or an open conductor also produce some I₀ — that is why the ground
  pick-up is a threshold, not a zero test.)
* **positive sequence (I₁)**: the normal balanced set, abc rotation — present in load and in every fault.
* **negative sequence (I₂)**: a balanced set of the *opposite* rotation (acb) — present in every unbalanced
  fault, **absent in a three-phase fault and in a balanced load change**.

The three sequence networks are decoupled for a balanced system (the line and sources are symmetric); a
fault is the only place where they connect.

---

## 3. Shunt faults and their boundary conditions

With sequence impedances Z₁ = Z₂ (lines are passive and the sources are modelled as fixed impedances, so the
negative-sequence impedance equals the positive-sequence one here; for a real synchronous generator
Z₂ ≈ X″d ≠ Z₁) and Z₀ seen from the fault point, and the pre-fault voltage E at the fault:

| fault | boundary conditions | sequence networks | fault current |
|---|---|---|---|
| **three-phase (ABC)** | V_a = V_b = V_c (symmetrical) | positive only | I_F = E / Z₁ |
| **single line-to-ground (AG)** | I_b = I_c = 0, V_a = 3R_f·I₀ | all three in **series** (I₀ = I₁ = I₂) | I_a = 3E / (Z₁ + Z₂ + Z₀ + 3R_f) |
| **line-to-line (BC)** | I_a = 0, I_b = −I_c, V_b − V_c = R_f·I_b | positive and negative in **parallel**, no zero | I_b = −j√3 E / (Z₁ + Z₂ + R_f) |
| **double line-to-ground (BCG)** | I_a = 0, V_b = V_c (bolted) | all three in **parallel** | I₁ = E / (Z₁ + Z₂ ∥ (Z₀ + 3R_g)) |

The simulator is checked against the first three formulas (radial feed, bolted) within 1 %
(`tests/test_physics.py`). For a fault through resistance the simulator uses an *arc resistance* r per
faulted phase to a common fault node and a *ground resistance* R_g from that node to earth, so
R_f = r + R_g for AG, 2r for BC, and r and R_g enter the LLG network separately.

**Why ground faults are the most common and the most resistive:** 70–80 % of overhead-line faults are
single-line-to-ground (lightning flashover of one insulator string, a tree, a bird); the arc plus tower
footing resistance plus ground return is tens of ohms, while a phase-to-phase arc is usually below a few ohms.

---

## 4. What the relay measures — DFT phasors

A numerical relay samples v(t), i(t) and estimates the fundamental-frequency phasor over a window of one
cycle (N = 80 samples):

```
X = (2/N) · Σ_{k=0}^{N−1} x[k] · e^{−j2πk/N}          (x(t) = Re{ X e^{jωt} }, peak phasor)
```

* exactly rejects the DC component and all harmonics (orthogonality over a full cycle),
* does **not** reject a *decaying* DC offset — the exponential is not periodic — which biases the phasor
  magnitude and angle for the first cycle or two after a fault (τ = L/R of the fault loop ≈ 20 ms here);
  relays filter it with a mimic (digital high-pass) circuit; this project simply uses a later window
  (3 cycles) for location, where it matters, and shows that classification does not care,
* when the window straddles the fault instant the phasor is a mixture of pre- and post-fault signals —
  that is the "accuracy versus decision time" study.

**Inception angle and DC offset.** For an R-L circuit switched on a sinusoidal voltage, the current is
i(t) = Iₘ [sin(ωt + θ − φ) − sin(θ − φ) e^{−t/τ}]. The DC term is largest when θ − φ = ±90°, i.e. — since
φ ≈ 80° for a transmission line — when the fault starts near the voltage **zero crossing**; the first peak
can then approach twice the steady fault current. At the voltage peak there is (almost) no offset.

---

## 5. Disturbance detection — superimposed quantities

The *superimposed* (delta, "pure fault") quantity is the post-fault signal minus the pre-fault signal:
Δi(t) = i(t) − i(t − T). By superposition, the faulted network = pre-fault network + a "pure fault"
network in which the only source is −V_F (the pre-fault voltage at the fault point) applied at the fault,
with all other sources shorted. So:

* Δi is zero until something changes — a perfect starting element (threshold 15 % of nominal current
  for 3 consecutive samples here);
* the superimposed positive-sequence current ΔI₁ is the fault current the relay sees with the load removed —
  the reference against which the sequence angles are measured;
* it is also what the Takagi locator uses (§8).

---

## 6. Phase selection — the sequence-angle logic

In the pure-fault network the sequence currents at the relay are fixed fractions of the fault's sequence
currents (the current-distribution factors C₁ ≈ C₂, C₀ of the line + sources). From the boundary conditions:

| fault | at the fault point | at the relay, ∠(I₂ / ΔI₁) | ∠(I₂ / I₀) |
|---|---|---|---|
| AG | I₁ = I₂ = I₀ | 0° | 0° |
| BG | I₂ = a·I₁, I₀ = a²·I₁ | +120° | −120° |
| CG | I₂ = a²·I₁, I₀ = a·I₁ | −120° | +120° |
| BC, BCG | I₂ = −I₁ (BC exactly; BCG for R_g → 0) | 180° | — |
| AB, ABG | I₂ = −a²·I₁ | +60° | — |
| CA, CAG | I₂ = −a·I₁ | −60° | — |
| ABC, load change | I₂ = 0 | undefined | — |

So one angle, ∠(I₂/ΔI₁), splits the plane into **six 60° sectors** (the measured angle is assigned to the
nearest centre, i.e. ±30° around each), one for each faulted phase or phase pair; it does not depend on the
fault resistance or the location (both scale I₁ and I₂ together), only on the small difference between C₁
and C₂, on the ground-resistance path of LLG faults (which rotates I₂ by up to 17° on the training events)
and on measurement error. This is how numerical distance relays
select the faulted phase: manufacturers use the I₂–I₀ angle for ground faults and/or the I₂–ΔI₁ angle
(the "superimposed" or "delta" quantities) — see Phadke & Thorp and Ziegler in the references.

The remaining decisions:

* **ground involved?** residual current |3I₀| > 6 % of the largest phase current → LLG rather than LL.
  The pick-up is 1.5 × the largest spurious residual that two phases with opposite ±2 % ratio errors can
  produce (4 %). (A single-phase fault always involves ground, so AG/BG/CG need no residual test —
  important for high-resistance faults, whose residual current is small. A double-line-to-ground fault at
  the far end through high ground resistance is the hard case: most of its zero-sequence current flows in
  from the remote end, and the local residual can fall to 5–6 %.)
* **balanced event** (|I₂|/|ΔI₁| and |I₀|/|ΔI₁| both < 0.30 — a three-phase fault shows a spurious I₂ of up
  to 0.15 from unequal decaying DC offsets, the smallest unbalanced fault 0.46): a three-phase fault if all
  three phase-loop impedances lie inside 160 % of the line impedance (the distance-relay zone idea; a
  three-phase fault with up to 5 Ω of arc shows at most 1.2 ×, the largest load step at least 2.3 ×),
  otherwise a load change.
* **why not ∠(I₂/I₀)?** the classical ground selector gives AG 0°, BG −120°, CG +120° and works for
  single-phase faults, but for a double-line-to-ground fault with ground resistance the zero-sequence branch
  (Z₀ + 3R_g) rotates I₀ relative to I₂ by tens of degrees, so BCG events drift out of their sector. The
  project found this empirically (about 94 % → 99.8 % on the test set after switching the primary selector
  to ∠(I₂/ΔI₁), before the other settings were re-derived) — it
  is the kind of thing the simulator exists to expose.

The settings (60°-wide sectors, 6 %, 0.30, 160 %) are derived from the measurement model and the line and
checked for margin on the training events; nothing is fitted. The first version used a 10 % pick-up and a
current-rise test for three-phase faults and scores 99.8 % on the test set (six far-end LLG faults through
27–43 Ω read as LL, two load steps of 129 and 141 MW read as ABC) — the settings above remove both failure
modes.

---

## 7. Distance protection — loop impedances

A distance relay measures the apparent impedance of the faulted loop and compares it with the line
impedance (zone 1 ≈ 80–85 % of the line, zone 2 ≈ 120 %, with time grading). The six loops:

| loop | voltage | current | equals d·Z₁ for a bolted fault at d km because … |
|---|---|---|---|
| A-G (and B-G, C-G) | V_a | I_a + k₀·3I₀ | V_a = d(Z₁ I_a + (Z₀ − Z₁) I₀) and k₀ = (Z₀ − Z₁)/(3Z₁) |
| A-B (and B-C, C-A) | V_a − V_b | I_a − I_b | the zero-sequence terms cancel in the difference |

The project computes all six (per unit of the line impedance) as features; the ML models find them useful
for separating fault types too (an A-G loop impedance inside the line characteristic means an A-G fault).

---

## 8. One-ended fault location

**Reactance method.** For the right loop, Z_loop = d·z₁ + R_f·(I_F / I_loop). If the remote infeed current
is in phase with the local one, the second term is purely resistive and d = Im(Z_loop) / x₁ exactly. With
a load angle (remote infeed out of phase) the R_f term acquires a reactive part → **reactance error**, which
grows with R_f and with distance (more infeed from the far end). Measured here: median error 0.49 km for
bolted faults, 1.7 km above 20 Ω, worst 57 km.

**Takagi method (1982).** Project on the superimposed current ΔI instead of on I_loop:

```
d = Im( V_loop · ΔI* ) / Im( z₁ · I_loop · ΔI* )
```

Because I_F ∝ ΔI in the pure-fault network (same distribution factor along the line if the system is
homogeneous), the resistive term R_f·I_F·ΔI* is (almost) real and drops out of the imaginary part. Measured
here: MAE 0.68 km, 90th percentile 1.55 km, and the error no longer grows with R_f (median ≈ 0.5 km in every
resistance band). Residual error comes from the non-homogeneous system (source angles differ), the lumped
line model, ratio errors and the remaining DC offset.

**Why the ML regressor loses to Takagi:** it must *learn* a relationship that Takagi *computes*; with
6,000 training faults and the same features it reaches 1.1 km MAE — a useful sanity check that
"machine learning" is not automatically the best tool.

Other methods (not implemented): two-ended impedance methods (synchronised phasors from both ends remove
the infeed problem entirely), travelling-wave location (time of arrival of the fault surge, ± 300 m, needs
MHz sampling), and the IEEE C37.114 guide covers all of them.

---

## 9. The machine-learning side

* **features (58)**: per-phase |V|, |I|, |ΔI|, |ΔV|, current rise ratio, share of the largest ΔI; sequence
  magnitudes |V₀,V₁,V₂|, |I₀,I₁,I₂|, ratios I₀/I₁, I₂/I₁, V₀/V₁, V₂/V₁; |ΔI₁|, I₁ rise, I₀/ΔI₁, I₂/ΔI₁,
  residual ratio; cos/sin of ∠(I₂/I₀), ∠(I₂/ΔI₁), ∠(I₀/ΔI₁); |Z|, X and R of the six loops in per unit of
  the line impedance; the trigger flag. Angles enter as cos/sin so that −180° and +180° are neighbours.
* **models**: kNN (k = 5, distance-weighted, standardised), MLP 64-64 (early stopping), Random Forest (400
  trees), histogram gradient boosting (300 iterations, 15 leaves). Fit on the 6,600 training events only;
  the model used for the decision-time study and the confusion matrix is chosen by 5-fold cross-validated
  macro-F1 on the training set, never on the test set.
* **impurity importance** of the forest ranks exactly the relay quantities first: I₂/ΔI₁, I₀/I₁, residual
  ratio, cos ∠(I₂/ΔI₁), I₀/ΔI₁ — the forest rediscovers the sequence-component logic.
* **generalisation**: an unseen system (150 km, constants ±15 %, other sources) — possible only because the
  features are per unit; the Random Forest and gradient boosting stay at 100 %.

---

## 10. Fault statistics worth knowing in an interview

* Overhead lines: ~70–80 % LG, then LLG and LL at roughly 5–15 % each (Blackburn & Domin: LLG 10–17 %,
  LL 8–10 %), three-phase 2–5 %; most are transient (arc extinguishes after the breaker opens → auto-reclose).
* Typical protection: distance (21) as main, directional earth-fault (67N) for high-resistance faults, line
  differential (87L) where a communication channel exists; a numerical relay operates in ¾–1½ cycles, a
  breaker in 2–3 cycles, so total clearing ≈ 60–100 ms at 50 Hz.
* CT ratio errors of 1–3 % (class 5P/10P) and VT ratio errors of 0.5–3 % (measuring class 0.5 to protection
  class 3P) are normal — hence the ±2 % ratio errors in the data; CT saturation under heavy DC offset is the
  classic source of relay misoperation (not modelled).
* Indian 220/400 kV practice (CEA / POWERGRID): main-1 and main-2 distance relays from different makers,
  carrier-aided schemes (PUTT/POTT), fault locators in the relay reported to the control centre.

# Interview preparation — Transmission-Line Fault Detection, Classification and Location

How to present this project to a PSU / core-company panel (POWERGRID, NTPC, NHPC, state transcos, BHEL,
Siemens Energy, ABB/Hitachi Energy, GE Vernova, L&T, Tata Power, Adani, Sterlite …) and the questions they
are likely to ask. All numbers come from `reports/metrics.json`.

---

## 1. The 30-second pitch

> "I built a numerical line-protection relay in software and tested it on thousands of simulated faults.
> I wrote a physics simulator of a 220 kV, 100 km line between two sources — coupled-phase π-sections,
> a fault as a resistance network, solved exactly with a matrix exponential — and checked it against the
> textbook fault-current formulas. The relay works the way real ones do: a delta disturbance detector,
> one-cycle DFT phasors, symmetrical components, phase selection from the angle between the
> negative-sequence and the superimposed positive-sequence current, a residual-current ground test and an
> impedance-zone test. With settings derived from the CT error model and the line — nothing fitted — it
> classifies all 10 fault types plus load switching correctly on 3,300 test events and on 2,200 events from
> a different line, one cycle after the trigger. Then I trained four machine-learning models on the same
> relay quantities: a Random Forest matches the relay logic at 100 % but cannot beat it; it only wins when
> the decision is forced earlier than one cycle. For fault location, Takagi's method gives 0.7 km mean error
> on the 100 km line and beats the ML regressor. There is a Streamlit app where you can create any fault and
> watch the relay decide."

---

## 2. The 2-minute version

**Problem.** When a transmission line faults, the relay at one end has about one cycle — 20 ms — to decide:
is it a fault (and not a load change)? which phases, and is ground involved (10 types: AG, BG, CG, AB, BC,
CA, ABG, BCG, CAG, ABC)? and how far away is it? It only has three voltages and three currents.

**Data.** Real relay records are rare and never labelled by distance and resistance, and every public
dataset is itself a simulation — so I wrote my own simulator and made every assumption testable. Two
sources, lumped π-line with mutual coupling built from the positive- and zero-sequence constants, fault
network with arc and ground resistance, exact integration with `expm`, closed-form steady state. It matches
the hand formulas for three-phase, single-line-to-ground and line-to-line currents within 1 %. I generated
6,600 training and 3,300 test events with random location, resistances, inception angle, load angle and
source strengths, plus 2,200 events on a different line (150 km, constants changed 15 %) to test
generalisation, and the relay sees them through ±2 % CT/VT ratio errors and 40 dB noise.

**Relay.** Delta-current starting element → full-cycle DFT phasors of the window ending one cycle after the
trigger → Fortescue components → 58 per-unit features (sequence ratios and angles, superimposed currents, six
loop impedances). Two rule-based classifiers: the overcurrent intuition (98.2 %, with 58 false trips on load
switching) and the sequence-angle logic of numerical relays, whose settings I derived from the measurement
model rather than fitted: 100 % on the test set and 100 % on the unseen line, one error in 6,600 training
events. Four ML models on the same features: kNN 99.8 %, MLP 99.9 %, Random Forest and gradient boosting
100 %, and 100 % on the unseen line.

**Studies.** Decision time (the rule is 90 % at half a cycle, 99.9 % at three-quarters, 100 % at one; a
forest trained for the early window is 100 % from half a cycle), noise (everything holds to 30 dB, collapses
together at 20 dB), high-resistance faults (above 20 Ω everything except kNN is 100 %), and fault location:
reactance method 1.7 km MAE, Takagi 0.68 km, ML regressor 1.1 km — the physical method with the right
quantity wins.

**What I'd say I learned.** The feature is the engineering: raw samples from the Kaggle dataset give 87 %,
relay quantities from a one-cycle window give 100 % with fixed settings. And settings must come from the
measurement model: my first version used a 10 % residual pick-up and lost six far-end double-line-to-ground
faults through high ground resistance; deriving the pick-up from the CT error model fixed it.

---

## 3. Likely questions and short answers

### A. Faults and symmetrical components

**1. What are the types of shunt faults and how common are they?**
Single line-to-ground (LG, ~70–80 % of overhead-line faults), then double line-to-ground and line-to-line
(roughly 5–15 % each — Blackburn & Domin quote LLG 10–17 % and LL 8–10 %) and three-phase (LLL, 2–5 %).
Three-phase faults are the rarest but usually
the most severe for stability; LG faults are the most common and often high-resistance (arc + tower footing
+ ground return).

**2. What are symmetrical components and why do relays use them?**
Fortescue's theorem: any unbalanced three-phase set equals the sum of a positive-sequence set (abc), a
negative-sequence set (acb) and a zero-sequence set (three in-phase phasors). For a balanced network the
three sequence networks are decoupled; a fault connects them. Negative sequence exists only for unbalanced
faults, zero sequence only when ground is involved — so a relay can tell the fault type from quantities
that do not exist in normal operation.

**3. How are the sequence networks connected for each fault?**
LG: all three in series (I₁ = I₂ = I₀, I_fault = 3E/(Z₁+Z₂+Z₀+3R_f)). LL: positive and negative in parallel,
no zero sequence (I = √3E/(Z₁+Z₂+R_f)). LLG: all three in parallel. LLL: positive sequence only (I = E/Z₁).

**4. Why is a three-phase fault with ground the same class as without ground?**
A balanced fault draws no zero-sequence current — the three fault currents sum to zero — so the ground
connection carries nothing and the relay cannot see it. The project keeps them as one class ("ABC") and
simulates half of them grounded to make sure.

**5. Why is Z₀ about three times Z₁ for an overhead line?**
Zero-sequence current returns through the earth and ground wires; the loop is much larger and its
inductance much higher (Carson's equations). Here Z₁ = 0.405 Ω/km and Z₀ = 1.22 Ω/km. That is also why
the residual-compensation factor k₀ = (Z₀−Z₁)/(3Z₁) ≈ 0.67 is needed in ground-loop impedance.

**6. What is a DC offset in a fault current and when is it largest?**
The current in an R-L circuit cannot jump, so when a fault starts the AC steady state is reached only
through a decaying exponential term, i(t) = Iₘ[sin(ωt+θ−φ) − sin(θ−φ)e^{−t/τ}], τ = L/R. It is largest
when the fault starts near the voltage zero crossing (θ−φ = ±90°, with φ ≈ 80° for a line) and can double
the first peak; it is almost absent at the voltage peak. In my simulator it comes out of the circuit
equations — the test suite checks that the 90° inception gives an asymmetric first cycle and 0° does not.

**7. What is fault resistance made of and what range did you use?**
Arc resistance (Warrington's formula: tens of ohms per metre of arc at low current, falling with current)
plus tower-footing resistance and ground-return resistance for ground faults. I used an arc resistance
per faulted phase of 0.01–5 Ω and a ground resistance of 0.1–50 Ω (log-uniform, so most faults are low
resistance with a realistic tail), and I report results separately for total resistance above 20 Ω.

### B. The simulator

**8. Why simulate instead of using a real or public dataset?**
Real relay records are rare, confidential and not labelled by distance and resistance; the public datasets
(Kaggle's, for instance) are Simulink exports of a *single* run per class with no time column and no
variation of location or resistance — you cannot test generalisation on them. A simulator I wrote myself
makes every assumption explicit and testable, and lets me randomise everything that a relay must be
immune to. I also included the Kaggle set to show what it can and cannot tell you.

**9. How does the simulator work?**
Phase-domain lumped π-sections: 3×3 R, L, C matrices with mutual coupling, built from the sequence
constants (R_self = (R₀+2R₁)/3, R_mutual = (R₀−R₁)/3, same for L and C). Two sources behind R-L
impedances, the remote one lagging by the load angle. The fault is a 3×3 conductance matrix at node F:
each faulted phase through an arc resistance to a common node, that node through a ground resistance to
earth, with the node eliminated analytically. The whole thing is a 23-state linear system driven by the
sources, which I model as two oscillator states (cos ωt, sin ωt), so x' = Ax is autonomous and I step it
*exactly* with x[k+1] = e^{AΔt} x[k]. The pre-fault steady state is solved in closed form, so the record
starts with no start-up transient.

**10. How did you validate it?**
Against hand calculations with a radial feed and bolted faults: three-phase current E/|Z₁|, single-line-
to-ground 3E/|2Z₁+Z₀|, line-to-line √3E/|2Z₁| at three distances each, all within 1 %; no zero-sequence
current for a phase-phase fault; healthy phases carrying only charging current for an LG fault; a balanced
steady state with the expected power flow (≈ 187 MW at δ = 15°); and the k₀ factor and both locators
exact for bolted faults. 27 physics tests in all (31 with the data and model checks).

**11. What are the limitations of the model?**
Lumped (not distributed) line — no travelling waves, which is fine for 50 Hz phasor relaying but not for
travelling-wave protection; no CT saturation; no frequency dependence of the line parameters; the remote
end is a simple source behind an impedance; no evolving faults, power swings, series compensation or
parallel lines with mutual coupling. Each of these is a known next step and I can say what it would change.

**12. What does the relay "see"?**
Voltages and currents at the sending end sampled at 4 kHz (80 samples per cycle), each channel multiplied
by a random CT/VT ratio error within ±2 % (the order of a protection-class CT's ratio error), plus white
noise at 40 dB SNR relative to the pre-fault RMS. The noise study then lowers that to 20 dB.

### C. Relay signal processing

**13. How does a numerical relay get phasors from samples?**
A full-cycle DFT: X = (2/N) Σ x[k] e^{−j2πk/N} over the last N = 80 samples. It rejects DC and every
harmonic exactly, but not a *decaying* DC offset, which biases the phasor for the first cycle or two. A
half-cycle DFT is faster but passes even harmonics; real relays add a mimic filter for the DC offset.

**14. How is the fault detected (the starting element)?**
A delta or superimposed-current detector: every sample is compared with the sample one cycle earlier,
|i[k] − i[k−N]|; three consecutive samples above 15 % of nominal current trigger it. In steady state the
delta is zero whatever the load, so it is far more sensitive than an overcurrent pick-up — but it also
triggers on load switching, which is why classification must then decide "fault or not".

**15. What are superimposed ("pure fault") quantities and why do they matter?**
Post-fault minus pre-fault signals. By superposition the faulted network equals the pre-fault network plus
a network whose only source is −V_pre-fault at the fault point. The superimposed positive-sequence current
ΔI₁ is therefore "the fault current without the load current" — the right reference for the sequence
angles, and the quantity Takagi's locator projects on.

**16. Why a decision window of one cycle, and what happens if you decide earlier?**
A DFT needs a full cycle of post-fault data to be clean; at half a cycle the window is half pre-fault and
the phasors are a mixture. Measured: the sequence rule is 76 % at ¼ cycle, 90 % at ½, 99.9 % at ¾, 100 % at
1 cycle (3,299 of 3,300 at 1.5 cycles, all at 2); a Random Forest retrained for each decision time is 99.9 %
at ¼ cycle and 100 % from ½ cycle — a learned model can exploit the mixed window, a fixed rule cannot. One
cycle is the standard compromise between speed and security.

**17. What is the effect of measurement noise?**
Settings and models fixed (derived/trained at 40 dB), test noise varied: everything ≥ 98 % down to 30 dB;
at 25 dB the rule is 99.5 %, the forest 99.3 %, the other ML models 98–99 % and the overcurrent rule 97.7 %;
at 20 dB (noise RMS = 10 % of the load current, which is a terrible measurement) all methods fall to 68–74 %
together. The limit is the
measurement, not the algorithm — a relay engineer would answer by filtering or a longer window, not a
different classifier.

### D. Classification

**18. Explain the sequence-angle rule.**
The angle between the negative-sequence current and the superimposed positive-sequence current, ∠(I₂/ΔI₁),
follows from the boundary conditions: AG 0°, BG +120°, CG −120°, BC/BCG 180°, AB/ABG +60°, CA/CAG −60° —
six 60°-wide sectors (the nearest centre wins; LLG faults drift up to 17° from their centre, ground and
phase-phase faults stay within 3°), independent of fault resistance and distance. Residual current (3I₀ above 6 % of the
largest phase current) separates LLG from LL. No negative and no zero sequence means a balanced event: a
three-phase fault if all three phase-loop impedances lie inside 160 % of the line impedance — the
distance-relay zone idea — otherwise a load change, whose impedance is the load impedance, several times the
line. Nothing is fitted, and it scores 100 % on the test set.

**19. How did you choose the thresholds, and is 100 % on your own test set not suspicious?**
Each threshold comes from the measurement model or the line, and I only *checked* its margin on the
training events. The ground pick-up: two phases with opposite ±2 % ratio errors produce a spurious residual
of up to 4 % of the phase current, so the pick-up is 1.5 × that, 6 % (line-to-line faults in training reach
3.8 %). The balanced-event threshold: unequal decaying DC offsets leak into the one-cycle DFT and give a
three-phase fault a spurious |I₂|/|ΔI₁| of up to 0.15, the smallest unbalanced fault has 0.46, so the
threshold is 0.30, in the middle. The zone: three-phase faults anywhere on the line with up to 5 Ω of arc
show at most 1.2 × the line impedance, the largest load step at least 2.3 ×, so the reach is 1.6 ×. 100 %
means the margins held on 3,300 new events and on a different line; the one training error — a CAG fault at
97.7 km through 32 Ω with a 5.7 % residual — shows where the margin ends, and the closest test event (an
LLG fault with a 6.6 % residual) shows how thin it is for far-end double-line-to-ground faults through high
ground resistance, where the remote end carries most of the zero-sequence current — in practice the remote
relay and a directional earth-fault element cover that case. The honest statement is "zero errors on 5,500
events it was not tuned on", not "perfect".

**20. Where did your first version fail?**
With a 10 % residual pick-up and a current-rise test for three-phase faults it scores 99.8 % on the test
set: six double-line-to-ground faults in the last 10 km of the line through 27–43 Ω of ground resistance are
reported as line-to-line, because most of their zero-sequence current flows in from the remote end and the
local residual is only 6.6–9.2 %; and two load switchings of 129 and 141 MW at a light pre-fault load more than
doubled the positive-sequence current and tripped the current-rise test. Both are physics, not bugs — and
both disappeared when the settings were derived properly (CT error model; impedance zone instead of current
rise). A good example of why protection settings are calculated, not guessed.

**21. Why not use the classical ∠(I₂/I₀) ground selector?**
I started with it (AG 0°, BG −120°, CG +120°) and got 93.7 %: for a double-line-to-ground fault the
zero-sequence path contains 3R_g, which rotates I₀ relative to I₂ by tens of degrees as R_g grows, so BCG
events drifted out of their sector and were misread as LL. ∠(I₂/ΔI₁) does not have that problem because
both currents flow through the same positive/negative-sequence impedances. The map in
`reports/figures/sequence_angle_map.png` shows both angles for every test event.

**22. What is the overcurrent rule and why is it worse?**
"The phases whose current jumped by at least half of the largest jump are faulted; residual current means
ground." It gets every fault type right but trips on 58 of the 300 no-fault events — all of them load
switchings (half of the no-fault events switch 20–150 MW, the other half switch nothing and never trigger
the detector; the false trips are steps of 77–148 MW, more likely when the pre-fault load is light), because
"the current jumped on all three phases"
is also what a load change looks like. The sequence rule survives those because a load change has no
negative sequence and leaves the loop impedance far outside the line.

**23. Which ML models, how trained, and how do you know they did not overfit?**
kNN (k = 5, distance-weighted, standardised features), an MLP with two layers of 64 (early stopping), a
Random Forest (400 trees) and histogram gradient boosting (300 iterations, 15 leaves). Trained only on the
6,600 training events; the 3,300 test events are new random draws; and the 2,200 unseen-system events
come from a different line (150 km, constants ±15 %, different sources) — the forest and the boosting stay
at 100 % there, kNN at 99.7 % and the MLP at 99.6 %. That is the overfitting test: if the models had learned
"this line" instead of "fault physics", the unseen set would show it. The model used for the follow-up
studies is picked by 5-fold cross-validation on the training set (RF and GB both 1.000, MLP 0.999,
kNN 0.997), never on the test set.

**24. Why can the models transfer to a different line?**
Because the features are per-unit relay quantities — voltages in per unit of nominal, currents of a fixed
base, loop impedances in per unit of the line impedance, angles between sequence currents — not raw samples
or absolute amperes. The angles in particular are the same on any line (they depend only on the boundary
conditions). It is the relay-engineering equivalent of feature normalisation.

**25. What does the forest consider important?**
Impurity importance ranks I₂/ΔI₁ (0.056), I₀/I₁ (0.055), the residual ratio (0.047), cos ∠(I₂/ΔI₁)
(0.041) and I₀/ΔI₁ (0.040) first, then the per-phase superimposed currents. In other words it rediscovers
the sequence-component logic. Permutation importance was not informative because the forest is 100 % on
the test set and the features are redundant — shuffling any one of them changes nothing.

**26. What about high-resistance faults?**
268 of the test events have more than 20 Ω total fault resistance. Everything except kNN (98.9 %) is 100 %
there. They were the hard group for my first version of the rule (six far-end LLG faults lost to a 10 %
residual pick-up), which is what led me to derive the pick-up from the CT error model.

**27. What is the Kaggle dataset and why do you say its 99.9 % is an illusion?**
7,861 rows, six classes, every row one time sample of one Simulink run per class — a single location and
resistance, no time column, irregular sampling. The published notebooks shuffle rows and split randomly,
so neighbouring samples of the same waveform land on both sides; even then a Random Forest on the raw
samples only gets 86.6 % for me, and 88.2 % with a time-block split, because an instantaneous sample is an
ambiguous description of a fault (every waveform passes through zero). A relay works on a window — that
is what this project does — and a single run per class cannot test generalisation at all.

**28. Precision, recall, macro-F1 — why those metrics?**
Classes are balanced here so accuracy is fine, but for protection the asymmetric errors matter: a *missed
fault* (fault classified as "none") is a failure to trip, a *false trip* (load switching classified as a
fault) is an unwanted outage. I report both — every method has zero missed faults; the overcurrent rule has
58 false trips, everything else none. Macro-F1 picks the best ML model without favouring any class.

### E. Fault location

**29. How does the reactance method work?**
A distance relay measures the apparent impedance of the faulted loop — V_a/(I_a + k₀·3I₀) for an A-G
fault, (V_a−V_b)/(I_a−I_b) for A-B — which for a bolted fault equals d·Z₁. Taking only the imaginary part,
d = Im(Z)/X₁, removes a purely resistive arc. With remote infeed the fault current is not in phase with the
local current, the R_f drop acquires a reactive component, and the method over- or under-reaches: median
error 0.49 km for bolted faults, 1.7 km above 20 Ω, worst case 57 km.

**30. What does Takagi's method change?**
It projects on the superimposed current ΔI instead of the total current: d = Im(V·ΔI*)/Im(z₁·I_loop·ΔI*).
In the pure-fault network I_fault ∝ ΔI, so the resistive term is (nearly) real and drops out of the
imaginary part regardless of the load angle. Measured: MAE 0.68 km, 90th percentile 1.55 km, and the median
error stays ≈ 0.5 km in every resistance band. Residual error comes from the non-homogeneous system (the
two sources' angles differ), the lumped-line model, ratio errors and the remaining DC offset.

**31. Why does the ML regressor lose to Takagi?**
It has to *learn* a relation that Takagi *computes* exactly; with 6,000 training faults and the same 58
features plus the predicted class it reaches 1.1 km MAE. Its MAE is better than the reactance method's
(1.7 km) because it avoids the large errors above 20 Ω (median 1.3 km against 1.7 km there), so it learns
part of the infeed correction — but its median error (0.79 km) is worse than the reactance method's (0.70 km)
and far from Takagi's 0.47 km. A closed-form physical method with the right quantity is simply better here.
A useful lesson: ML is not automatically the best tool.

**32. Why do you locate 3 cycles after the trigger but classify at 1?**
The decaying DC offset biases the one-cycle phasors by up to about 10 % — a bolted 60 km fault reads
anywhere from 53 to 65 km depending on the inception angle — but not enough to change a sector of 60°. By
three cycles (τ ≈ 20 ms) it has mostly decayed (59–61 km);
tests check both locators within 1.2 km at 3 cycles and 0.2 km at 5. A real relay filters the offset with
a mimic circuit and can locate earlier; location is not time-critical anyway (it is reported after the trip).

**33. What other location methods exist?**
Two-ended impedance methods with synchronised (PMU) measurements eliminate the infeed and R_f problems;
travelling-wave locators time the arrival of the fault surge (hundreds of metres, needs MHz sampling and
GPS time); IEEE C37.114 is the guide. In India, POWERGRID uses the fault locator in the distance relay
and increasingly travelling-wave units on long 400/765 kV lines.

### F. Protection practice they may jump to

**34. What is distance protection and what are its zones?**
A relay that measures loop impedance and trips if it lies inside a characteristic (mho or quadrilateral).
Zone 1 covers ~80–85 % of the line instantaneously (not 100 %, because of measurement error and the
infeed/under-reach problem), zone 2 covers the whole line plus ~20 % of the next with ~0.3–0.4 s delay, zone 3
is back-up with ~1 s. Carrier-aided schemes (PUTT/POTT) let zone 2 trip instantly when the remote end
also sees the fault. My three-phase-fault test is exactly a zone check with a 160 % reach.

**35. What is the difference between under-reach and over-reach?**
Under-reach: the relay sees a fault inside its zone as outside (e.g. fault resistance with infeed from the
far end makes the impedance look larger). Over-reach: it sees a fault beyond the zone as inside (e.g.
mutual coupling of a parallel line, or a decaying DC offset enlarging the measured current). Zone 1 is set
short to stay secure against over-reach.

**36. How fast must protection be and why?**
A numerical relay decides in ¾–1½ cycles, the breaker opens in 2–3 cycles, so a 220 kV fault is cleared in
60–100 ms. Faster clearing improves transient stability (equal-area criterion: less accelerating energy)
and reduces equipment damage; the critical clearing time of a stressed system can be ~100–150 ms.

**37. What are CT saturation and its effect on relays?**
A large DC offset or a high fault current drives the CT core into saturation; the secondary current
collapses within each half cycle, phasors are under-estimated and distance relays can under-reach or
mis-select the phase. It is the classic reason for mis-operation that this project does not model — I'd
add a CT model (magnetising-curve) and an anti-saturation algorithm as the next step.

**38. Which relays are used on a 220 kV line in India?**
Two main distance relays (main-1 and main-2, often different makes), directional earth-fault back-up,
auto-reclosing (single-phase auto-reclosing is standard on 400 kV lines; at 220 kV practice varies between
single-phase and three-phase schemes), and a fault locator and disturbance recorder in the relay. Communication-aided
schemes over PLCC or OPGW fibre.

### G. Engineering judgement

**39. What is the difference between a fault and a power swing for a distance relay?**
A power swing is a slow, balanced oscillation of the apparent impedance after a disturbance; a fault is a
sudden jump. Relays block on swings by measuring the rate of change of impedance (blinders / ΔZ/Δt) and
the absence of negative-sequence current. My balanced-event test (no I₂, no I₀, impedance inside the zone)
is the first half of that logic; the rate-of-change part is not implemented.

**40. If you had to deploy one method, which one and why — and what next?**
The sequence-angle relay logic for the trip decision: it is explainable, its settings are calculated from
the CT class and the line data, it has no training data to maintain, and a protection engineer must be able
to say *why* a relay tripped — "the forest said so" is not an acceptable relay setting. A model trained for
the half-cycle window is the one place ML buys something (100 % against 90 %), so I would use it as a
fast-trip accelerator supervised by the rule, and Takagi for the location report. With another month: a CT
saturation model and the decaying-DC mimic filter; a distributed-parameter (Bergeron) line for
travelling-wave location; double-circuit lines with zero-sequence mutual coupling; evolving and
cross-country faults; power swings; and real disturbance-recorder COMTRADE files from a utility (the CLI
already accepts a six-channel CSV, so a COMTRADE reader is a small step).

---

## 4. Three things not to say

* **"The model achieved 100 % accuracy."** Lead with the physics: the relay logic gets 100 % because its
  settings are derived from the measurement model and the margins held; say immediately that the
  unseen-line test, the training-set error and the noise study are what make the number mean something.
  100 % on a test set from the same simulator is expected once the physics is right, not impressive.
* **"Machine learning is better than conventional relays."** It is not, in this project: the forest ties
  with the rule on classification and Takagi beats the ML locator. ML earns its place only at early
  decision times — say exactly that.
* **"It is real data."** It is a validated physics simulation; the Kaggle set is also a simulation. Be the
  person who explains *why* simulated data was the right choice and what the tests check.

---

## 5. If they ask you to draw something

* **The one-line diagram** (two sources, relay at S, fault at d km) and the sequence networks for an LG
  fault (three in series) and an LL fault (positive and negative in parallel).
* **The six sectors** of ∠(I₂/ΔI₁) on a circle: AG at 0°, AB at +60°, BG at +120°, BC at 180°, CG at −120°,
  CA at −60° — then say "ground faults are at the phase angles, phase faults in between, residual current
  tells LL from LLG, and a balanced event is a fault only if the impedance is inside the zone".
* **A fault current with DC offset**: a sine wave shifted up by a decaying exponential, first peak nearly
  doubled; mark the one-cycle DFT window and say why the first cycle's phasor is biased.
* **The reactance error**: V = d·Z₁·I + R_f·I_F with I_F not in phase with I; show the R_f·I_F phasor
  tilting off the real axis — "Takagi projects on ΔI to make it real again".

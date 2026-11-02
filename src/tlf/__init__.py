"""
tlf — transmission-line fault detection, classification and location.

Sub-modules
-----------
simulate   : physics-based time-domain simulator of a two-source transmission
             line with shunt faults (exact matrix-exponential integration)
dsp        : full-cycle DFT phasors, symmetrical components, disturbance detector
features   : the relay-style feature vector computed from one decision window
classical  : rule-based fault classifiers (overcurrent rule, sequence-angle relay logic)
locate     : impedance-based fault locators (reactance method, Takagi method)
models     : scikit-learn classifiers / regressors used in the comparison
evaluate   : metrics helpers
plots      : figures for the report
"""

__version__ = "1.0.0"

# The 11 classes.  Phases are named A, B, C; "G" means ground is involved.
# A three-phase fault with or without ground is indistinguishable at the relay
# (no zero-sequence current flows in a balanced fault), so both are "ABC".
CLASSES = ["none", "AG", "BG", "CG", "AB", "BC", "CA", "ABG", "BCG", "CAG", "ABC"]

GROUPS = {  # coarse fault type
    "none": "no fault", "AG": "LG", "BG": "LG", "CG": "LG", "AB": "LL", "BC": "LL", "CA": "LL",
    "ABG": "LLG", "BCG": "LLG", "CAG": "LLG", "ABC": "LLL",
}
GROUP_ORDER = ["no fault", "LG", "LL", "LLG", "LLL"]

# faulted phases and ground flag of each class
PHASES = {
    "none": (), "AG": ("A",), "BG": ("B",), "CG": ("C",), "AB": ("A", "B"), "BC": ("B", "C"), "CA": ("C", "A"),
    "ABG": ("A", "B"), "BCG": ("B", "C"), "CAG": ("C", "A"), "ABC": ("A", "B", "C"),
}
GROUNDED = {c: c.endswith("G") for c in CLASSES}

PHASE_INDEX = {"A": 0, "B": 1, "C": 2}

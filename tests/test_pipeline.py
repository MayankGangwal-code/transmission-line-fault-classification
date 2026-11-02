"""Checks on the generated data and saved models (skipped until the pipeline has run)."""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tlf import CLASSES  # noqa: E402
from tlf.classical import apply_rule, sequence_rule  # noqa: E402
from tlf.evaluate import summarise  # noqa: E402
from tlf.features import feature_names  # noqa: E402
from tlf.models import decode  # noqa: E402

DATA, MODELS = ROOT / "data", ROOT / "models"
needs_data = pytest.mark.skipif(not (DATA / "events_test.csv").exists(), reason="run scripts/run_experiments.py first")
needs_models = pytest.mark.skipif(not (MODELS / "classifiers.joblib").exists(), reason="run scripts/run_experiments.py first")


def _split(df: pd.DataFrame):
    names = feature_names()
    return df[names], df["label"].to_numpy()


@needs_data
def test_event_tables_are_balanced_and_complete():
    for name, n in (("train", 600), ("test", 300), ("unseen", 200)):
        df = pd.read_csv(DATA / f"events_{name}.csv")
        assert set(df["label"]) == set(CLASSES)
        assert (df["label"].value_counts() == n).all()
        assert not df[feature_names()].isna().any().any()
        assert df["distance_km"].between(0, df["line_km"]).all()


@needs_data
def test_sequence_rule_accuracy_on_saved_test_features():
    X, y = _split(pd.read_csv(DATA / "events_test.csv"))
    s = summarise(y, apply_rule(sequence_rule, X))
    assert s["accuracy"] > 0.98 and s["missed_faults"] == 0


@needs_models
def test_saved_classifier_and_locator():
    bundle = joblib.load(MODELS / "classifiers.joblib")
    X, y = _split(pd.read_csv(DATA / "events_test.csv"))
    best = bundle["models"][bundle["best"]]
    assert summarise(y, decode(best.predict(X)))["accuracy"] > 0.99
    loc = joblib.load(MODELS / "locator.joblib")
    assert len(loc["columns"]) == len(feature_names()) + len(CLASSES)


@needs_data
def test_waveform_sample_file():
    z = np.load(DATA / "waveforms_sample.npz", allow_pickle=False)
    assert z["v"].shape[1:] == (640, 3) and z["i"].shape == z["v"].shape
    assert set(z["labels"]) == set(CLASSES)

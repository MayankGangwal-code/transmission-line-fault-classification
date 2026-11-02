"""Metrics helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from . import CLASSES, GROUP_ORDER, GROUPS


def to_group(labels) -> np.ndarray:
    return np.array([GROUPS[str(l)] for l in labels])


def summarise(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str)
    fault_true, fault_pred = y_true != "none", y_pred != "none"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0)),
        "group_accuracy": float(np.mean(to_group(y_true) == to_group(y_pred))),  # LG / LL / LLG / LLL / none
        "detection_accuracy": float(np.mean(fault_true == fault_pred)),  # fault vs no fault
        "missed_faults": int(np.sum(fault_true & ~fault_pred)),
        "false_trips": int(np.sum(~fault_true & fault_pred)),
        "n": int(len(y_true)),
    }


def confusion(y_true, y_pred, classes=CLASSES) -> pd.DataFrame:
    cm = confusion_matrix(np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str), labels=classes)
    return pd.DataFrame(cm, index=classes, columns=classes)


def per_class(y_true, y_pred, classes=CLASSES) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(np.asarray(y_true).astype(str), np.asarray(y_pred).astype(str), labels=classes, zero_division=0)
    return pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=classes).round(3)


def location_errors(d_true, d_hat) -> dict:
    e = np.abs(np.asarray(d_hat, float) - np.asarray(d_true, float))
    e = e[np.isfinite(e)]
    return {"mae_km": float(np.mean(e)), "median_km": float(np.median(e)), "p90_km": float(np.quantile(e, 0.9)), "max_km": float(np.max(e)), "n": int(len(e))}

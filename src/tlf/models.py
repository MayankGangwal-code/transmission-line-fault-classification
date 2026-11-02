"""scikit-learn classifiers and regressors used in the comparison."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import CLASSES

RANDOM_STATE = 42
CLASS_TO_INT = {c: i for i, c in enumerate(CLASSES)}


def encode(labels) -> np.ndarray:
    return np.array([CLASS_TO_INT[str(l)] for l in labels], dtype=int)


def decode(codes) -> np.ndarray:
    return np.array([CLASSES[int(c)] for c in codes], dtype=object).astype(str)


def build_classifiers(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    return {
        "k-Nearest Neighbours": Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance"))]),
        "Neural Network": Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=400, early_stopping=True, n_iter_no_change=15, random_state=random_state)),
            ]
        ),
        "Random Forest": Pipeline([("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=1, n_jobs=-1, random_state=random_state))]),
        "Gradient Boosting": Pipeline(
            [("clf", HistGradientBoostingClassifier(learning_rate=0.08, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0, random_state=random_state))]
        ),
    }


def build_locator(random_state: int = RANDOM_STATE) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(learning_rate=0.05, max_iter=600, max_leaf_nodes=31, l2_regularization=1.0, random_state=random_state)

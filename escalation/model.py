from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import HISTORY, KEYS, semantic, signals, structured


def row(messages, context, use_embeddings=False):
    return (list(structured(messages, context).values()),
            " ".join(f"{m.role} {m.text}" for m in messages[-5:]),
            semantic(messages) if use_embeddings else None)


class RiskModel:
    def __init__(self, kind, strength, dropped=()):
        self.kind = kind
        self.strength = strength
        self.columns = [i for i, key in enumerate(KEYS) if key not in dropped]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=3000)
        self.estimator = (
            HistGradientBoostingClassifier(max_leaf_nodes=strength, max_iter=100,
                                           class_weight="balanced", random_state=42)
            if kind == "structured" else make_pipeline(
                StandardScaler(with_mean=False),
                LogisticRegression(C=strength, class_weight="balanced", max_iter=500),
            )
        )
        self.calibrator = LogisticRegression()

    def matrix(self, rows, fitting=False):
        numerical = np.asarray([r[0] for r in rows])[:, self.columns]
        if self.kind == "baseline":
            text = [r[1] for r in rows]
            vectors = self.vectorizer.fit_transform(text) if fitting else self.vectorizer.transform(text)
            return hstack([csr_matrix(numerical), vectors], format="csr")
        if self.kind == "hybrid":
            return np.column_stack([numerical, np.stack([r[2] for r in rows])])
        return numerical

    def fit(self, rows, labels, weights):
        args = {"sample_weight" if self.kind == "structured" else "logisticregression__sample_weight": weights}
        self.estimator.fit(self.matrix(rows, fitting=True), labels, **args)
        self.reference = np.median(np.asarray([r[0] for r in rows]), axis=0)
        return self

    def raw(self, rows):
        return self.estimator.decision_function(self.matrix(rows)).reshape(-1, 1)

    def calibrate(self, rows, labels, weights):
        self.calibrator.fit(self.raw(rows), labels, sample_weight=weights)

    def probabilities(self, rows):
        return self.calibrator.predict_proba(self.raw(rows))[:, 1]

    def explain(self, current, risk):
        # Replace one observed feature at a time with its training median.
        variations = []
        for column in self.columns:
            values = current[0].copy()
            values[column] = self.reference[column]
            variations.append((values, current[1], current[2]))
        deltas = risk - self.probabilities(variations)
        ranked = sorted(zip(self.columns, deltas), key=lambda item: abs(item[1]), reverse=True)
        return [{"feature": KEYS[i], "delta": round(float(delta), 4)}
                for i, delta in ranked[:4] if abs(delta) >= 0.0001]


class Predictor:
    def __init__(self, path: Path):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.info = bundle["info"]

    def score(self, messages, context):
        start = perf_counter()
        current = row(messages, context, self.model.kind == "hybrid")
        risk = float(self.model.probabilities([current])[0])
        return {"risk": risk, "signals": signals(dict(zip(KEYS, current[0]))),
                "contributions": self.model.explain(current, risk),
                "latency_ms": (perf_counter() - start) * 1000}

"""Prediction models: selective-feature logistic blend with the simulator.

``GameModel`` predicts P(home team wins) from:
    * the Monte Carlo simulator's probability (the physical prior), and
    * the selectively-filtered feature subset.

``TotalsModel`` predicts the expected total runs the same way (linear),
used for over/under evaluation.

Both are plain, inspectable scikit-learn objects — calibrated
probabilities matter more than exotic learners for betting evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

SIM_COLS = ["sim_p_home"]
SIM_COLS_TOTAL = ["sim_mean_total"]
EPS = 1e-6


@dataclass
class FitReport:
    kept_features: list[str]
    coeficients: dict[str, float]
    intercept: float


class _ScaledLinear:
    def __init__(self, model: BaseEstimator):
        self.model = model
        self.scaler = StandardScaler()

    def _prep(self, X: np.ndarray) -> np.ndarray:
        return np.nan_to_num(X, nan=0.0)

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xs = self.scaler.fit_transform(self._prep(X))
        self.model.fit(Xs, y)
        return self

    def predict_proba_home(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(self._prep(X))
        return self.model.predict_proba(Xs)[:, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(self._prep(X))
        return self.model.predict(Xs)


class GameModel:
    """P(home win) from simulator prior + selected features."""

    def __init__(self, feature_cols: list[str]):
        self.feature_cols = ["sim_p_home"] + list(feature_cols)
        self._inner = _ScaledLinear(
            LogisticRegression(max_iter=2000, C=1.0))

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.feature_cols].to_numpy(dtype=float)

    def fit(self, train: pd.DataFrame) -> FitReport:
        self._inner.fit(self._X(train), train["home_win"].to_numpy())
        coefs = {c: float(v) for c, v in
                 zip(self.feature_cols, self._inner.model.coef_[0])}
        return FitReport(kept_features=list(self.feature_cols),
                         coeficients=coefs,
                         intercept=float(self._inner.model.intercept_[0]))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        p = self._inner.predict_proba_home(self._X(df))
        return np.clip(p, EPS, 1 - EPS)


class TotalsModel:
    """Expected total runs from simulator mean total + selected features."""

    def __init__(self, feature_cols: list[str]):
        self.feature_cols = ["sim_mean_total"] + list(feature_cols)
        self._inner = _ScaledLinear(LinearRegression())

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.feature_cols].to_numpy(dtype=float)

    def fit(self, train: pd.DataFrame):
        self._inner.fit(self._X(train), train["total_runs"].to_numpy(dtype=float))
        resid = train["total_runs"].to_numpy(dtype=float) - self.predict(train)
        self.resid_sigma = float(np.std(resid)) or 1.0

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self._inner.predict(self._X(df))

    def prob_over(self, df: pd.DataFrame, line: float) -> np.ndarray:
        from scipy.stats import norm  # local import: optional dependency
        mu = self.predict(df)
        return norm.cdf((mu - line) / self.resid_sigma)

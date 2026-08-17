"""Selective input filtering — the project's stated edge.

From the wide candidate pool we keep only inputs that show a real,
*stable* linear correlation with the betting outcome **on past data
only**:

1. Point-biserial correlation of each candidate with the label on the
   training window; keep |r| >= tau.
2. Sign-stability check: the training window is split in two halves and
   the correlation sign must agree in both (kills features whose effect
   flipped mid-history).
3. Redundancy pruning: among near-duplicates (|r_ff'| >= redundancy),
   keep the one most correlated with the label.

The selector is re-run inside every walk-forward retrain, so the set of
inputs is allowed to evolve — but never using future games.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SelectionResult:
    kept: list[str]
    dropped_low_corr: list[str] = field(default_factory=list)
    dropped_unstable: list[str] = field(default_factory=list)
    dropped_redundant: list[str] = field(default_factory=list)
    correlations: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kept": self.kept,
            "dropped_low_corr": self.dropped_low_corr,
            "dropped_unstable": self.dropped_unstable,
            "dropped_redundant": self.dropped_redundant,
            "correlations": {k: round(v, 5) for k, v in self.correlations.items()},
        }


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if x.size < 30 or x.std() < 1e-9 or y.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def select_features(train: pd.DataFrame, candidates: list[str], label: str,
                    tau: float = 0.03, redundancy: float = 0.96,
                    min_obs: int = 200) -> SelectionResult:
    """Correlation + stability + redundancy filter on strictly-past data."""
    res = SelectionResult(kept=[])
    y = train[label].to_numpy(dtype=float)
    if len(train) < min_obs:
        return res  # not enough history -> keep nothing beyond defaults

    corrs: dict[str, float] = {}
    half = len(train) // 2
    for col in candidates:
        if col not in train.columns:
            continue
        x = train[col].to_numpy(dtype=float)
        r = _corr(x, y)
        if np.isnan(r):
            res.dropped_low_corr.append(col)
            continue
        corrs[col] = r
        if abs(r) < tau:
            res.dropped_low_corr.append(col)
            continue
        r1 = _corr(x[:half], y[:half])
        r2 = _corr(x[half:], y[half:])
        if np.isnan(r1) or np.isnan(r2) or np.sign(r1) != np.sign(r2):
            res.dropped_unstable.append(col)
            continue
        res.kept.append(col)

    res.correlations = corrs

    # Redundancy pruning, strongest-first.
    kept_sorted = sorted(res.kept, key=lambda c: abs(corrs[c]), reverse=True)
    final: list[str] = []
    for c in kept_sorted:
        xc = train[c].to_numpy(dtype=float)
        redundant = False
        for k in final:
            xk = train[k].to_numpy(dtype=float)
            mask = ~(np.isnan(xc) | np.isnan(xk))
            if mask.sum() < 30:
                continue
            rr = np.corrcoef(xc[mask], xk[mask])[0, 1]
            if abs(rr) >= redundancy:
                redundant = True
                break
        if redundant:
            res.dropped_redundant.append(c)
        else:
            final.append(c)
    res.kept = sorted(final)
    return res

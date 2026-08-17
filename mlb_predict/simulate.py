"""Monte Carlo game simulation.

Run scoring is drawn from a Negative Binomial whose mean blends a team's
recent offense with the opponent's recent defense, with a data-estimated
home advantage.  The simulation distribution is the model's *prior*;
``model.py`` blends it with the selectively-filtered feature model.

Nothing here is tuned by eye on future games: the dispersion and home
advantage constants are re-estimated inside the walk-forward loop from
strictly-past games (``estimate_globals``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RUN_DISPERSION = 1.22       # var/runs-per-game for MLB teams (~empirical)
FLOOR, CEIL = 1.5, 8.0      # sanity clamp on per-team expected runs


@dataclass(frozen=True)
class Globals:
    home_adv: float = 1.024          # multiplicative on home team run rate
    dispersion: float = RUN_DISPERSION


def estimate_globals(history_runs_home: np.ndarray,
                     history_runs_away: np.ndarray) -> Globals:
    """Estimate home advantage + scoring dispersion from past games only."""
    hh = np.asarray(history_runs_home, dtype=float)
    ha = np.asarray(history_runs_away, dtype=float)
    hh, ha = hh[~np.isnan(hh)], ha[~np.isnan(ha)]
    if hh.size < 50:
        return Globals()
    home_adv = float(np.clip(hh.mean() / max(ha.mean(), 1e-6), 0.98, 1.08))
    pooled = np.concatenate([hh, ha])
    dispersion = float(np.clip(pooled.var() / max(pooled.mean(), 1e-6),
                               1.05, 1.45))
    return Globals(home_adv=home_adv, dispersion=dispersion)


def expected_runs(rf_team: float, ra_opp: float, is_home: bool,
                   g: Globals) -> float:
    """Blend own recent offense with opponent recent defense."""
    if np.isnan(rf_team):
        rf_team = 4.5
    if np.isnan(ra_opp):
        ra_opp = 4.5
    lam = 0.5 * (rf_team + ra_opp)
    if is_home:
        lam *= g.home_adv
    return float(np.clip(lam, FLOOR, CEIL))


def _nb_sample(lam: float, dispersion: float, n: int,
               rng: np.random.Generator) -> np.ndarray:
    """Negative binomial via the exact Gamma-Poisson mixture."""
    p = 1.0 / dispersion
    r = max(lam * p / (1.0 - p), 1e-3)
    rates = rng.gamma(shape=r, scale=(1.0 - p) / p, size=n)
    return rng.poisson(rates).astype(np.int32)


def simulate_game(rf_home: float, ra_home: float,
                  rf_away: float, ra_away: float,
                  g: Globals | None = None, n_sims: int = 4000,
                  seed: int | None = None) -> dict:
    """Return the simulated outcome distribution for one matchup.

    Keys: p_home, p_total_over per integer line, mean_total, p_home_ml_runline.
    """
    g = g or Globals()
    rng = np.random.default_rng(seed)
    lam_h = expected_runs(rf_home, ra_away, True, g)
    lam_a = expected_runs(rf_away, ra_home, False, g)
    runs_h = _nb_sample(lam_h, g.dispersion, n_sims, rng)
    runs_a = _nb_sample(lam_a, g.dispersion, n_sims, rng)
    total = runs_h + runs_a
    margin = runs_h - runs_a
    return {
        "sim_p_home": float((runs_h > runs_a).mean()),
        "sim_mean_total": float(total.mean()),
        "sim_mean_margin": float(margin.mean()),
        "sim_p_home_cover": float((margin >= 2).mean()),  # home -1.5 run line
        "sim_lam_home": lam_h,
        "sim_lam_away": lam_a,
    }


def simulate_frame(df, g: Globals | None = None, n_sims: int = 2000,
                   seed: int = 7) -> dict:
    """Fully vectorised Monte Carlo for a frame with rf10/ra10 columns."""
    g = g or Globals()
    n = len(df)
    lam_h = np.array([expected_runs(rf, ra_opp, True, g)
                      for rf, ra_opp in zip(df["rf10_home"], df["rf10_away"])])
    lam_a = np.array([expected_runs(rf, ra_opp, False, g)
                      for rf, ra_opp in zip(df["rf10_away"], df["ra10_home"])])
    rng = np.random.default_rng(seed)
    p = 1.0 / g.dispersion

    def draws(lams: np.ndarray) -> np.ndarray:
        # Negative binomial via the exact Gamma-Poisson mixture so the
        # success-count parameter can stay fractional per team.
        r = np.maximum(lams * p / (1.0 - p), 1e-3)
        rates = rng.gamma(shape=np.repeat(r[:, None], n_sims, axis=1),
                          scale=(1.0 - p) / p)
        return rng.poisson(rates).astype(np.int32)

    runs_h = draws(lam_h)
    runs_a = draws(lam_a)
    total = runs_h + runs_a
    margin = runs_h - runs_a
    return {
        "sim_p_home": (runs_h > runs_a).mean(axis=1),
        "sim_mean_total": total.mean(axis=1),
        "sim_mean_margin": margin.mean(axis=1),
        "sim_p_home_cover": (margin >= 2).mean(axis=1),
    }

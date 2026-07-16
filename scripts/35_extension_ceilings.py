"""Price every proposed expressiveness/estimation upgrade BEFORE building it,
then measure what realizable versions actually collect.

Extends the paper's error budget from "where the error lives" to "what each
fix is worth", on the same 13 rolling monthly windows as RQ3 (train
cumulative < T, test [T, T+1mo), decisive votes, production half_tie fits).

Oracles (upper bounds, computed with test-month knowledge):
  O1 recalibration : best single scale s on sigma(s*g) per test month
                     (the paper's drift line; should reproduce ~12.4x MPD)
  O2 month refit   : BT refit on the test month itself, scored in-sample
                     (caps ANY dynamic-ability scheme, drift + shape)
  O3 cold start    : unseen-model votes, p=0.5 baseline vs month-refit oracle
                     (caps ANY metadata prior / early-shrinkage scheme)
  O4 tie drift     : per-month tie share vs cumulative-train tie share
                     (caps a time-varying tie parameter, global level only)

Realizable (walk-forward, hyperparameters chosen on the PREVIOUS window):
  R1 time-decay BT     : vote weight 0.5^(age/half_life)
  R2 recalibration     : scale fitted on previous month's held-out preds
  R3 decay + recal     : both
  R4 ridge (global l2) : shrinkage toward the mean ability

Output: results/tables/extension_budget_windows.csv + pooled summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.fit import fit_gaplink  # noqa: E402

MPD_DEC = 4e-4   # paper threshold, decisive channel (nats/vote)
MPD_TIE = 3e-4   # paper threshold, tie channel


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class BT:
    """log F and dlogF/dg for the Bradley-Terry decisive link."""

    @staticmethod
    def log_f_decisive(g):
        # log sigma(g) computed stably; derivative is sigma(-g)
        return -np.logaddexp(0.0, -g), _sig(-g)


def aggregate(battles: pd.DataFrame, decay_weights: np.ndarray | None = None) -> pd.DataFrame:
    """Collapse identical (a, b, winner) votes to weighted rows (exact likelihood)."""
    df = battles[["model_a", "model_b", "winner"]].copy()
    df["weight"] = 1.0 if decay_weights is None else decay_weights
    return (df.groupby(["model_a", "model_b", "winner"], observed=True, as_index=False)
              ["weight"].sum())


def fit_bt(train: pd.DataFrame, decay_weights=None, l2=1e-6) -> pd.Series:
    agg = aggregate(train, decay_weights)
    return fit_gaplink(agg, BT, mode="half_tie", include_both_bad=True, l2=l2)


def nll_decisive(theta: pd.Series, test: pd.DataFrame, scale: float = 1.0) -> tuple[float, int]:
    """Mean NLL (nats/vote) of decisive test votes; both models must be in theta."""
    g = theta.reindex(test["model_a"]).to_numpy() - theta.reindex(test["model_b"]).to_numpy()
    won_a = (test["winner"] == "model_a").to_numpy()
    z = np.where(won_a, g, -g) * scale
    return float(np.mean(np.logaddexp(0.0, -z))), len(z)


def oracle_scale(theta: pd.Series, test: pd.DataFrame) -> tuple[float, float]:
    g = theta.reindex(test["model_a"]).to_numpy() - theta.reindex(test["model_b"]).to_numpy()
    z = np.where((test["winner"] == "model_a").to_numpy(), g, -g)

    def f(s):
        return np.mean(np.logaddexp(0.0, -s * z))

    r = minimize_scalar(f, bounds=(0.2, 5.0), method="bounded",
                        options={"xatol": 1e-6})
    return float(r.x), float(r.fun)


def kl_bern(p, q):
    p = np.clip(p, 1e-12, 1 - 1e-12); q = np.clip(q, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def main():
    df = pd.read_parquet(ROOT / "data/processed/clean_battle_20240814.parquet")
    df = df[df.dedup_sampled].copy()
    df["dt"] = pd.to_datetime(df["tstamp"], unit="s")
    checkpoints = pd.date_range("2023-07-01", "2024-07-01", freq="MS")

    HL_GRID = [30.0, 60.0, 90.0, 180.0, 365.0, np.inf]     # days
    L2_GRID = [1e-6, 0.03, 0.1, 0.3, 1.0]

    prev_best_hl, prev_best_l2, prev_recal_s = np.inf, 1e-6, 1.0
    rows = []
    for T in checkpoints:
        T1 = T + pd.DateOffset(months=1)
        train = df[df.dt < T]
        month = df[(df.dt >= T) & (df.dt < T1)]
        known = set(train.model_a) | set(train.model_b)
        dec = month[month.winner.isin(["model_a", "model_b"])]
        scor = dec[dec.model_a.isin(known) & dec.model_b.isin(known)]
        cold = dec[dec.model_a.isin(known) ^ dec.model_b.isin(known)]
        if len(scor) < 1000:
            continue

        age = (T - train.dt).dt.total_seconds().to_numpy() / 86400.0

        # ---------- baseline ----------
        theta0 = fit_bt(train)
        base, n = nll_decisive(theta0, scor)

        # ---------- oracles ----------
        s_star, o1 = oracle_scale(theta0, scor)                       # O1
        theta_m = fit_bt(month)                                        # month refit
        o2, _ = nll_decisive(theta_m, scor)                            # O2 (in-sample cap)
        # O3 cold start: p=0.5 vs oracle month refit, on unseen-vs-seen votes
        cold_ok = cold[cold.model_a.isin(theta_m.index) & cold.model_b.isin(theta_m.index)]
        if len(cold_ok) >= 200:
            o3_oracle, n_cold = nll_decisive(theta_m, cold_ok)
            o3_gain = np.log(2.0) - o3_oracle
        else:
            o3_gain, n_cold = np.nan, len(cold_ok)
        # O4 tie drift (global level): month tie share vs cumulative share
        def tie_share(d):
            nb = d[d.winner != "tie (bothbad)"]
            return (nb.winner == "tie").mean()
        o4 = kl_bern(tie_share(month), tie_share(train))

        # ---------- realizable, walk-forward ----------
        # R1: time-decay picked on the PREVIOUS window
        hl_scores = {}
        for hl in HL_GRID:
            wts = np.power(0.5, age / hl) if np.isfinite(hl) else None
            th = fit_bt(train, wts)
            hl_scores[hl] = (th, nll_decisive(th, scor)[0])
        r1 = hl_scores[prev_best_hl][1]
        # R2: recalibration with the PREVIOUS window's fitted scale
        r2, _ = nll_decisive(theta0, scor, scale=prev_recal_s)
        # R3: decay theta + previous scale
        r3, _ = nll_decisive(hl_scores[prev_best_hl][0], scor, scale=prev_recal_s)
        # R4: ridge picked on the PREVIOUS window
        l2_scores = {}
        for l2 in L2_GRID:
            th = fit_bt(train, l2=l2)
            l2_scores[l2] = (th, nll_decisive(th, scor)[0])
        r4 = l2_scores[prev_best_l2][1]

        rows.append(dict(
            checkpoint=str(T.date()), n_test=n, n_cold=len(cold),
            base=base,
            o1_recal=(base - o1) / MPD_DEC, o1_scale=s_star,
            o2_refit=(base - o2) / MPD_DEC,
            o3_cold_gain=(o3_gain / MPD_DEC if np.isfinite(o3_gain) else np.nan),
            cold_share=len(cold) / max(len(dec), 1),
            o4_tie_drift=o4 / MPD_TIE,
            r1_decay=(base - r1) / MPD_DEC, r1_hl=prev_best_hl,
            r2_recal=(base - r2) / MPD_DEC, r2_s=prev_recal_s,
            r3_both=(base - r3) / MPD_DEC,
            r4_ridge=(base - r4) / MPD_DEC, r4_l2=prev_best_l2,
        ))
        print(f"{T.date()}  n={n:6d}  base={base:.5f}  "
              f"O1={rows[-1]['o1_recal']:+6.2f}x  O2={rows[-1]['o2_refit']:+6.2f}x  "
              f"O3={rows[-1]['o3_cold_gain']:+7.1f}x(share {rows[-1]['cold_share']:.2f})  "
              f"O4={rows[-1]['o4_tie_drift']:+6.2f}x  "
              f"R1={rows[-1]['r1_decay']:+5.2f}x  R2={rows[-1]['r2_recal']:+5.2f}x  "
              f"R3={rows[-1]['r3_both']:+5.2f}x  R4={rows[-1]['r4_ridge']:+5.2f}x", flush=True)

        # update walk-forward selections for the NEXT window
        prev_best_hl = min(hl_scores, key=lambda h: hl_scores[h][1])
        prev_best_l2 = min(l2_scores, key=lambda l: l2_scores[l][1])
        prev_recal_s, _ = oracle_scale(theta0, scor)

    out = pd.DataFrame(rows)
    dest = ROOT / "results/tables/extension_budget_windows.csv"
    out.to_csv(dest, index=False)
    w = out.n_test.to_numpy(float)

    def pooled(col):
        v = out[col].to_numpy(float)
        m = np.isfinite(v)
        return float(np.average(v[m], weights=w[m]))

    print("\n=== POOLED (vote-weighted, x MPD) ===")
    for col, label in [("o1_recal", "O1 oracle recalibration (paper drift line ~12.4)"),
                       ("o2_refit", "O2 oracle month refit (cap on any dynamic scheme)"),
                       ("o3_cold_gain", "O3 cold-start oracle gain (on unseen votes only)"),
                       ("o4_tie_drift", "O4 tie-drift ceiling (tie MPD units)"),
                       ("r1_decay", "R1 time-decay BT (realizable)"),
                       ("r2_recal", "R2 walk-forward recalibration (realizable)"),
                       ("r3_both", "R3 decay + recalibration (realizable)"),
                       ("r4_ridge", "R4 global ridge (realizable)")]:
        print(f"  {label:55s} {pooled(col):+8.2f}")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()

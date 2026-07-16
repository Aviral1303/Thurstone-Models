"""GENERALIZATION RUN: transport the extension budget (script 35) and the
cold-start pedigree prior (script 36) to the 2025 Arena release
(arena-human-preference-140k: 135,634 battles, 53 models, 14 weeks).

Logic is imported unchanged from scripts 35/36 via importlib; only the
window grid changes: WEEKLY checkpoints (7-day steps from the data's min
date, first checkpoint at min_date + 14 days so every window has >= 2
weeks of training data), test window [T, T+7d). Same filters as the 2024
runs (>= 1000 scoreable decisive test votes for the budget, >= 200 cold
votes for the prior), same MPD constants (4e-4 decisive, 3e-4 tie), same
half-life/ridge grids, same walk-forward hyperparameter rule, and the
same ML-kappa temper rule (kappa maximizes the likelihood of all
previously observed cold votes; kappa = 1.0 for the first window).

The only 2025-specific addition is a LOCAL list of family keywords for
new-era model names (grok, o3, nova, ...) that script 36's 2024-era list
maps to "other". It is APPENDED to the imported list, so every 2024-era
name resolves exactly as before. Scripts 35/36 are not modified.

Outputs: results/tables/arena2025_extension_budget.csv
         results/tables/arena2025_coldstart_prior.csv
"""

from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ext35 = _load("ext35", ROOT / "scripts/35_extension_ceilings.py")
_ext36 = _load("ext36", ROOT / "scripts/36_coldstart_prior.py")

fit_bt, nll_decisive, oracle_scale, kl_bern = (
    _ext35.fit_bt, _ext35.nll_decisive, _ext35.oracle_scale, _ext35.kl_bern)
MPD_DEC, MPD_TIE = _ext35.MPD_DEC, _ext35.MPD_TIE
features, size_b, ridge_fit, ridge_predict = (
    _ext36.features, _ext36.size_b, _ext36.ridge_fit, _ext36.ridge_predict)

# 2025-era family keywords absent from script 36's 2024-era list (those
# names would all land in "other"). Appended AFTER the imported keywords
# so every 2024-era resolution is unchanged. Built by inspecting the 53
# model names in arena2025_140k.parquet.
EXTRA_FAMILY_KEYWORDS = [
    "nova",       # amazon-nova-*, amazon.nova-*
    "grok",       # grok-3-*, grok-4-*
    "hunyuan",    # hunyuan-turbos-*
    "kimi",       # kimi-k2-*
    "magistral",  # magistral-medium-* (before generic checks; no clash with "mistral")
    "minimax",    # minimax-m1
    "o3",         # o3-2025-*, o3-mini
    "o4",         # o4-mini-*
    "qwq",        # qwq-32b
]
_ext36.FAMILY_KEYWORDS = _ext36.FAMILY_KEYWORDS + EXTRA_FAMILY_KEYWORDS

WEEK = pd.Timedelta(days=7)


def load_battles() -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    df = pd.read_parquet(ROOT / "data/processed/arena2025_140k.parquet")
    # no dedup_sampled column in this release: use all rows
    df["dt"] = pd.to_datetime(df["tstamp"], unit="s")
    start, end = df.dt.min(), df.dt.max()
    checkpoints = []
    T = start + pd.Timedelta(days=14)   # two weeks of initial training
    while T < end:
        checkpoints.append(T)
        T = T + WEEK
    return df, checkpoints


def run_budget(df: pd.DataFrame, checkpoints: list[pd.Timestamp]) -> pd.DataFrame:
    """Script 35's main loop, verbatim, on weekly windows."""
    HL_GRID = [30.0, 60.0, 90.0, 180.0, 365.0, np.inf]     # days
    L2_GRID = [1e-6, 0.03, 0.1, 0.3, 1.0]

    prev_best_hl, prev_best_l2, prev_recal_s = np.inf, 1e-6, 1.0
    rows = []
    for T in checkpoints:
        T1 = T + WEEK
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
        theta_m = fit_bt(month)                                        # week refit
        o2, _ = nll_decisive(theta_m, scor)                            # O2 (in-sample cap)
        # O3 cold start: p=0.5 vs oracle week refit, on unseen-vs-seen votes
        cold_ok = cold[cold.model_a.isin(theta_m.index) & cold.model_b.isin(theta_m.index)]
        if len(cold_ok) >= 200:
            o3_oracle, n_cold = nll_decisive(theta_m, cold_ok)
            o3_gain = np.log(2.0) - o3_oracle
        else:
            o3_gain, n_cold = np.nan, len(cold_ok)
        # O4 tie drift (global level): week tie share vs cumulative share
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
    dest = ROOT / "results/tables/arena2025_extension_budget.csv"
    out.to_csv(dest, index=False)
    w = out.n_test.to_numpy(float)

    def pooled(col):
        v = out[col].to_numpy(float)
        m = np.isfinite(v)
        return float(np.average(v[m], weights=w[m]))

    print("\n=== POOLED (vote-weighted, x MPD) ===")
    for col, label in [("o1_recal", "O1 oracle recalibration"),
                       ("o2_refit", "O2 oracle week refit (cap on any dynamic scheme)"),
                       ("o3_cold_gain", "O3 cold-start oracle gain (on unseen votes only)"),
                       ("o4_tie_drift", "O4 tie-drift ceiling (tie MPD units)"),
                       ("r1_decay", "R1 time-decay BT (realizable)"),
                       ("r2_recal", "R2 walk-forward recalibration (realizable)"),
                       ("r3_both", "R3 decay + recalibration (realizable)"),
                       ("r4_ridge", "R4 global ridge (realizable)")]:
        print(f"  {label:55s} {pooled(col):+8.2f}")
    print(f"\nwrote {dest}")
    return out


def run_coldstart(df: pd.DataFrame, checkpoints: list[pd.Timestamp]) -> pd.DataFrame:
    """Script 36's main loop, verbatim, on weekly windows."""
    rows = []
    past_predictions: dict[str, float] = {}   # model -> prior predicted at its entry
    past_z: list[np.ndarray] = []             # signed prior gaps of past cold votes
    for T in checkpoints:
        T1 = T + WEEK
        train = df[df.dt < T]
        month = df[(df.dt >= T) & (df.dt < T1)]
        known = set(train.model_a) | set(train.model_b)
        dec = month[month.winner.isin(["model_a", "model_b"])]
        cold = dec[dec.model_a.isin(known) ^ dec.model_b.isin(known)]
        if len(cold) < 200:
            continue

        theta = fit_bt(train)
        counts = pd.concat([train.model_a, train.model_b]).value_counts()
        established = [m for m in theta.index if counts.get(m, 0) >= 200]
        th_est = theta[established]
        mean_th, med_sz = float(th_est.mean()), float(np.nanmedian(
            [size_b(m) for m in established if np.isfinite(size_b(m))]))

        X = np.array([features(m, th_est, mean_th, med_sz, exclude_self=True)
                      for m in established])
        y = th_est.to_numpy()
        beta, mu, sd = ridge_fit(X, y)

        new_models = sorted((set(month.model_a) | set(month.model_b)) - known)
        prior = {m: ridge_predict(features(m, th_est, mean_th, med_sz), beta, mu, sd)
                 for m in new_models}

        # ML-kappa temper rule, exactly as script 36: kappa maximizes the
        # likelihood of ALL previously observed cold votes; 1.0 first window.
        past_err = [(past_predictions[m] - theta[m]) ** 2
                    for m in past_predictions if m in theta.index and counts.get(m, 0) >= 50]
        sigma2 = float(np.mean(past_err)) if past_err else np.nan
        if past_z:
            zp = np.concatenate(past_z)
            kappa = float(minimize_scalar(
                lambda k: np.mean(np.logaddexp(0.0, -k * zp)),
                bounds=(0.0, 1.5), method="bounded").x)
        else:
            kappa = 1.0
        past_predictions.update(prior)

        # score cold votes: prior theta for the new side, trained theta for known
        th_all = pd.concat([theta, pd.Series(prior)])
        g = th_all.reindex(cold.model_a).to_numpy() - th_all.reindex(cold.model_b).to_numpy()
        z = np.where((cold.winner == "model_a").to_numpy(), g, -g)
        nll_prior = float(np.mean(np.logaddexp(0.0, -z)))
        nll_temper = float(np.mean(np.logaddexp(0.0, -kappa * z)))
        nll_half = float(np.log(2.0))
        past_z.append(z)

        # oracle: week refit (same definition as script 35's O3)
        theta_m = fit_bt(month)
        ok = cold[cold.model_a.isin(theta_m.index) & cold.model_b.isin(theta_m.index)]
        nll_oracle, _ = nll_decisive(theta_m, ok)

        gain = (nll_half - nll_prior) / MPD_DEC
        tgain = (nll_half - nll_temper) / MPD_DEC
        ceil = (nll_half - nll_oracle) / MPD_DEC
        rows.append(dict(checkpoint=str(T.date()), n_cold=len(cold),
                         n_new_models=len(new_models),
                         cold_share=len(cold) / max(len(dec), 1),
                         sigma2=sigma2, kappa=kappa,
                         nll_half=nll_half, nll_prior=nll_prior,
                         nll_temper=nll_temper, nll_oracle=nll_oracle,
                         prior_gain=gain, temper_gain=tgain, oracle_gain=ceil,
                         collected=tgain / ceil if ceil > 0 else np.nan))
        print(f"{T.date()}  cold={len(cold):6d} new={len(new_models):2d}  "
              f"raw {gain:+7.1f}x  temper(k={kappa:.2f}) {tgain:+7.1f}x  "
              f"oracle {ceil:+7.1f}x  collected {tgain/ceil if ceil>0 else float('nan'):.0%}",
              flush=True)

    out = pd.DataFrame(rows)
    dest = ROOT / "results/tables/arena2025_coldstart_prior.csv"
    out.to_csv(dest, index=False)
    w = out.n_cold.to_numpy(float)
    pg = np.average(out.prior_gain, weights=w)
    tg = np.average(out.temper_gain, weights=w)
    og = np.average(out.oracle_gain, weights=w)
    print(f"\nPOOLED (cold-vote weighted): raw prior {pg:+.1f}x MPD, "
          f"tempered {tg:+.1f}x, oracle {og:+.1f}x, "
          f"tempered collects {tg/og:.0%} of the ceiling")
    print(f"windows beating coin flip: raw {(out.prior_gain > 0).sum()}/{len(out)}, "
          f"tempered {(out.temper_gain > 0).sum()}/{len(out)}")
    print(f"wrote {dest}")
    return out


def main():
    df, checkpoints = load_battles()
    print(f"battles={len(df)}  models={pd.concat([df.model_a, df.model_b]).nunique()}  "
          f"span {df.dt.min()} -> {df.dt.max()}  weekly checkpoints={len(checkpoints)}")
    print("\n=== EXTENSION BUDGET (weekly, 2025) ===")
    run_budget(df, checkpoints)
    print("\n=== COLD-START PRIOR (weekly, 2025) ===")
    run_coldstart(df, checkpoints)


if __name__ == "__main__":
    main()

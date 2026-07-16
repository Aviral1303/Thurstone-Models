"""Collect the cold-start line of the extension budget (script 35): a
pedigree prior that scores unseen-model votes from the model's NAME alone.

Script 35 priced the oracle at ~+117x MPD on unseen-model votes (26-55% of
each month's decisive votes). Here we build the realizable version: at each
checkpoint, regress established models' fitted abilities on name-derived
features (predecessor checkpoint, family best, parameter count, frontier
variant), predict each entering model's ability before it has a single vote,
and score the month's new-vs-known votes with that prior.

Walk-forward, no leakage: the regression at checkpoint T uses only models
and abilities known at T; features are computed leave-one-out.

Output: results/tables/coldstart_prior_windows.csv
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("ext35", ROOT / "scripts/35_extension_ceilings.py")
_ext = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ext)
fit_bt, nll_decisive = _ext.fit_bt, _ext.nll_decisive

MPD_DEC = 4e-4

FAMILY_KEYWORDS = [
    "gpt4all", "chatgpt", "gpt", "claude", "gemini", "bard", "palm", "gemma",
    "codellama", "llama", "mixtral", "mistral", "qwen", "reka", "command",
    "deepseek", "chatglm", "glm", "phi", "vicuna", "wizardlm", "zephyr",
    "starling", "openchat", "openhermes", "mpt", "pythia", "dolly", "koala",
    "alpaca", "falcon", "snowflake", "solar", "stablelm", "rwkv", "tulu",
    "olmo", "dbrx", "nemotron", "pplx", "athene", "guanaco", "hyena", "yi",
]
FRONTIER = re.compile(r"large|max|opus|core|advanced|plus|405b|ultra|next")


def family(name: str) -> str:
    low = name.lower()
    for kw in FAMILY_KEYWORDS:
        if kw in low:
            return kw
    return "other"


def size_b(name: str) -> float:
    low = name.lower()
    m = re.search(r"(\d+)x(\d+(?:\.\d+)?)b", low)
    if m:
        return float(m.group(1)) * float(m.group(2))
    m = re.search(r"(\d+(?:\.\d+)?)b\b", low)
    return float(m.group(1)) if m else np.nan


def longest_prefix_theta(name: str, theta: pd.Series, exclude: str | None = None,
                         min_len: int = 6):
    best_len, best_th = 0, np.nan
    for other, th in theta.items():
        if other == name or other == exclude:
            continue
        k = 0
        for a, b in zip(name.lower(), other.lower()):
            if a != b:
                break
            k += 1
        if k >= min_len and k > best_len:
            best_len, best_th = k, th
    return best_th, best_len


def features(name: str, theta: pd.Series, mean_theta: float, med_size: float,
             exclude_self: bool = False) -> np.ndarray:
    excl = name if exclude_self else None
    fam = family(name)
    fam_thetas = [th for m, th in theta.items()
                  if m != excl and family(m) == fam]
    fam_best = max(fam_thetas) if fam_thetas else mean_theta
    pre_th, pre_len = longest_prefix_theta(name, theta, exclude=excl)
    if not np.isfinite(pre_th):
        pre_th = fam_best
    sz = size_b(name)
    logsz = np.log10(sz if np.isfinite(sz) else med_size)
    return np.array([1.0, pre_th, fam_best, logsz,
                     1.0 if FRONTIER.search(name.lower()) else 0.0])


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    # standardize non-intercept columns; penalize them only
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    Xs[:, 0] = 1.0
    pen = alpha * np.eye(X.shape[1])
    pen[0, 0] = 0.0
    beta = np.linalg.solve(Xs.T @ Xs + pen, Xs.T @ y)
    return beta, mu, sd


def ridge_predict(x: np.ndarray, beta, mu, sd) -> float:
    xs = (x - mu) / sd
    xs[0] = 1.0
    return float(xs @ beta)


def main():
    df = pd.read_parquet(ROOT / "data/processed/clean_battle_20240814.parquet")
    df = df[df.dedup_sampled].copy()
    df["dt"] = pd.to_datetime(df["tstamp"], unit="s")
    checkpoints = pd.date_range("2023-07-01", "2024-07-01", freq="MS")

    rows = []
    past_predictions: dict[str, float] = {}   # model -> prior predicted at its entry
    past_z: list[np.ndarray] = []             # signed prior gaps of past cold votes
    for T in checkpoints:
        T1 = T + pd.DateOffset(months=1)
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

        # Prior-uncertainty temper, walk-forward rule: kappa maximizes the
        # likelihood of ALL previously observed cold votes (an estimated
        # calibration parameter, not a tuned one — same logic as the
        # recalibration line of script 35). A Gaussian posterior-predictive
        # rule (kappa = 1/sqrt(1 + pi*sigma^2/8) from squared prior errors)
        # tempers too little because prior errors are heavy-tailed: frontier
        # releases are leaps, not jitter. sigma2 is kept as a diagnostic.
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

        # oracle: month refit (same definition as script 35's O3)
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
    out.to_csv(ROOT / "results/tables/coldstart_prior_windows.csv", index=False)
    w = out.n_cold.to_numpy(float)
    pg = np.average(out.prior_gain, weights=w)
    tg = np.average(out.temper_gain, weights=w)
    og = np.average(out.oracle_gain, weights=w)
    print(f"\nPOOLED (cold-vote weighted): raw prior {pg:+.1f}x MPD, "
          f"tempered {tg:+.1f}x, oracle {og:+.1f}x, "
          f"tempered collects {tg/og:.0%} of the ceiling")
    print(f"windows beating coin flip: raw {(out.prior_gain > 0).sum()}/{len(out)}, "
          f"tempered {(out.temper_gain > 0).sum()}/{len(out)}")


if __name__ == "__main__":
    main()

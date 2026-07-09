"""RQ3 evaluation machinery — held-out calibration of decisive-outcome
probabilities. Written and synthetically validated BEFORE any real-data
number is computed (logs/RQ3_PREANALYSIS.md). The same functions run the
synthetic validation and, after user go-ahead, the real experiment.

Design (pre-committed):
- Rolling-origin: train on votes with tstamp <= T_k, test on votes in
  (T_k, T_{k+1}]. Scoring is decisive-only, two-outcome, conditional on a
  decisive vote: p = F_decisive(theta_A - theta_B) from each method's own
  fitted link and abilities. Ties (both kinds) are excluded from scoring
  (tie prediction is RQ4's question); their counts are reported.
- Test votes involving a model absent from training are unscoreable
  (no ability estimate) — dropped and counted.
- Recent-entrant stratum: a test vote where >=1 of the two models first
  appeared in the training data within RECENT_DAYS before T_k. Complement
  stratum ("established") reported alongside. Headline = full population.
- Delta convention: delta = ll_bt - ll_lattice per vote, so POSITIVE means
  lattice better (lower loss).
- Inference: per-window mean deltas are the units (13 windows on real
  data). Pooled estimate = vote-weighted mean; CI by window-cluster
  bootstrap (resample windows). Effective N == number of windows, stated
  wherever the CI is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fit import VoteData, _rows_half_tie

RECENT_DAYS = 28
SECONDS_PER_DAY = 86400.0
DECISIVE = ("model_a", "model_b")


# ---------------- scoring ----------------

def predict_and_score(theta: pd.Series, link, test: pd.DataFrame) -> pd.DataFrame:
    """Score decisive test votes under one method. Returns per-vote frame
    with logloss/brier and scoreability flags (unscoreable rows kept for
    accounting, with NaN losses)."""
    dec = test[test["winner"].isin(DECISIVE)].copy()
    known = dec["model_a"].isin(theta.index) & dec["model_b"].isin(theta.index)
    g = (theta.reindex(dec["model_a"]).to_numpy()
         - theta.reindex(dec["model_b"]).to_numpy())
    p = np.where(known, link.f_decisive(np.nan_to_num(g)), np.nan)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    y = (dec["winner"] == "model_a").to_numpy(dtype=float)
    dec["scoreable"] = known.to_numpy()
    dec["logloss"] = np.where(known, -(y * np.log(p) + (1 - y) * np.log(1 - p)), np.nan)
    dec["brier"] = np.where(known, (p - y) ** 2, np.nan)
    return dec


def stratum_labels(test: pd.DataFrame, first_seen: pd.Series, t_k: float) -> pd.Series:
    """'recent' if either model entered training within RECENT_DAYS before T_k."""
    thresh = t_k - RECENT_DAYS * SECONDS_PER_DAY
    fa = first_seen.reindex(test["model_a"]).to_numpy()
    fb = first_seen.reindex(test["model_b"]).to_numpy()
    recent = (fa >= thresh) | (fb >= thresh)
    return pd.Series(np.where(recent, "recent", "established"), index=test.index)


# ---------------- window records ----------------

@dataclass
class WindowResult:
    label: str
    n_test_votes: int
    n_decisive: int
    n_scoreable: int
    n_ties_excluded: int
    per_vote: pd.DataFrame  # columns: stratum, d_logloss, d_brier


def evaluate_window(theta_bt: pd.Series, link_bt,
                    theta_lat: pd.Series, link_lat,
                    train: pd.DataFrame, test: pd.DataFrame,
                    t_k: float, label: str) -> WindowResult:
    first_seen = pd.concat([
        train.groupby("model_a")["tstamp"].min(),
        train.groupby("model_b")["tstamp"].min(),
    ], axis=1).min(axis=1)

    s_bt = predict_and_score(theta_bt, link_bt, test)
    s_lat = predict_and_score(theta_lat, link_lat, test)
    both = s_bt["scoreable"] & s_lat["scoreable"]
    strata = stratum_labels(s_bt, first_seen, t_k)
    per_vote = pd.DataFrame({
        "stratum": strata[both],
        "d_logloss": (s_bt["logloss"] - s_lat["logloss"])[both],
        "d_brier": (s_bt["brier"] - s_lat["brier"])[both],
        "ll_bt": s_bt["logloss"][both],
        "ll_lat": s_lat["logloss"][both],
    })
    return WindowResult(
        label=label,
        n_test_votes=len(test),
        n_decisive=len(s_bt),
        n_scoreable=int(both.sum()),
        n_ties_excluded=int((~test["winner"].isin(DECISIVE)).sum()),
        per_vote=per_vote,
    )


# ---------------- aggregation & inference ----------------

def window_table(results: list[WindowResult], stratum: str | None = None) -> pd.DataFrame:
    rows = []
    for r in results:
        pv = r.per_vote if stratum is None else r.per_vote[r.per_vote["stratum"] == stratum]
        if len(pv) == 0:
            rows.append({"window": r.label, "n": 0, "mean_d_logloss": np.nan,
                         "mean_d_brier": np.nan, "mean_ll_bt": np.nan, "mean_ll_lat": np.nan})
            continue
        rows.append({
            "window": r.label, "n": len(pv),
            "mean_d_logloss": float(pv["d_logloss"].mean()),
            "mean_d_brier": float(pv["d_brier"].mean()),
            "mean_ll_bt": float(pv["ll_bt"].mean()),
            "mean_ll_lat": float(pv["ll_lat"].mean()),
        })
    return pd.DataFrame(rows)


def pooled_estimate(wt: pd.DataFrame, col: str = "mean_d_logloss",
                    n_boot: int = 10_000, seed: int = 20260710) -> dict:
    """Vote-weighted pooled mean + window-cluster bootstrap percentile CI.
    Effective N = number of windows with data (stated in output)."""
    w = wt.dropna(subset=[col])
    n_windows = len(w)
    weights = w["n"].to_numpy(dtype=float)
    vals = w[col].to_numpy()
    pooled = float(np.average(vals, weights=weights))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_windows, size=(n_boot, n_windows))
    bs = np.average(vals[idx], weights=weights[idx], axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"pooled": pooled, "ci_lo": float(lo), "ci_hi": float(hi),
            "n_windows": n_windows, "n_votes": int(weights.sum())}


# ---------------- pre-committed pattern classifier (section 4) ----------------

MPD_LOGLOSS = 4e-4   # nats/vote; derivation in RQ3_PREANALYSIS.md section 3
SIGN_CONSISTENCY = 10  # of 13 windows (one-sided binomial p=0.046)


def classify(wt: pd.DataFrame, pooled: dict, mpd: float = MPD_LOGLOSS,
             consistency_needed: int | None = None) -> dict:
    """Three-way call per pre-committed section-4 rules. delta>0 = lattice
    better. Returns dict with verdict in {'equivalence','lattice_positive',
    'bt_positive','inconclusive'} plus the evidence trail. The mechanism
    grade (b2: recent-entrant concentration) is assessed separately by the
    caller — this classifier is direction/effect-size/consistency only (b1)."""
    w = wt.dropna(subset=["mean_d_logloss"])
    n = len(w)
    need = consistency_needed if consistency_needed is not None else int(np.ceil(n * SIGN_CONSISTENCY / 13))
    n_lat = int((w["mean_d_logloss"] > 0).sum())
    n_bt = int((w["mean_d_logloss"] < 0).sum())
    med = float(w["mean_d_logloss"].median())
    pooled_val, lo, hi = pooled["pooled"], pooled["ci_lo"], pooled["ci_hi"]

    ev = {"n_windows": n, "n_lattice_better": n_lat, "n_bt_better": n_bt,
          "median_delta": med, "pooled_delta": pooled_val,
          "ci": (lo, hi), "mpd": mpd, "consistency_needed": need}

    ci_excludes_equiv_band = lo > mpd or hi < -mpd
    ci_inside_band = (-mpd < lo) and (hi < mpd)

    if pooled_val >= mpd and n_lat >= need and lo > 0:
        ev["verdict"] = "lattice_positive"
    elif pooled_val <= -mpd and n_bt >= need and hi < 0:
        ev["verdict"] = "bt_positive"
    elif abs(pooled_val) < mpd and abs(med) < mpd and ci_inside_band:
        ev["verdict"] = "equivalence"
        # a real but sub-practical directional difference is reported as such,
        # not hidden inside "equivalence" (validated: both synthetic worlds
        # land here with the CORRECT direction)
        if lo > 0:
            ev["directional_note"] = "sub-practical directional lean: lattice"
        elif hi < 0:
            ev["directional_note"] = "sub-practical directional lean: bt"
    elif not ci_excludes_equiv_band and not ci_inside_band:
        ev["verdict"] = "inconclusive"
    else:
        # e.g. large pooled effect without sign consistency (few big windows)
        ev["verdict"] = "inconclusive"
        ev["note"] = "heterogeneous: effect size and consistency disagree"
    return ev


# ---------------- Fisher SEs (section 4.1 machinery) ----------------

# Fisher SEs underestimate the published bootstrap SDs by ~20% (validated
# full-regime cross-check, scripts/12: spearman 0.983, median ratio 0.80).
# The confound-correction procedure uses bootstrap-CALIBRATED SEs as primary
# (divide Fisher by this factor); raw Fisher SEs are the labeled sensitivity
# variant.
FISHER_TO_BOOTSTRAP_CALIBRATION = 0.80


def fisher_se(train: pd.DataFrame, link, theta: pd.Series) -> pd.Series:
    """Per-model SEs from expected information of the half-tie objective.
    I_ij = sum over weighted rows of F'(g)^2/(F(1-F)) with +/- design; gauge
    fixed by pseudo-inverse (mean-zero convention, matches our fits)."""
    data = VoteData.from_battles(train)
    w_idx, l_idx, wts = _rows_half_tie(data, include_both_bad=True)
    order = {m: i for i, m in enumerate(data.models)}
    th = theta.reindex(data.models).to_numpy()
    g = th[w_idx] - th[l_idx]
    eps = 1e-4
    F = np.clip(link.f_decisive(g), 1e-9, 1 - 1e-9)
    dF = (link.f_decisive(g + eps) - link.f_decisive(g - eps)) / (2 * eps)
    contrib = wts * dF ** 2 / (F * (1 - F))
    n = len(data.models)
    info = np.zeros((n, n))
    np.add.at(info, (w_idx, w_idx), contrib)
    np.add.at(info, (l_idx, l_idx), contrib)
    np.add.at(info, (w_idx, l_idx), -contrib)
    np.add.at(info, (l_idx, w_idx), -contrib)
    # The information matrix has a KNOWN null space (the constant vector —
    # translation gauge). Deflate it explicitly: add c*(11'/n) with huge c,
    # whose inverse contributes a negligible 1/(c*n) to the diagonal and
    # leaves all gauge-orthogonal directions exact. (A bare pinv inverted
    # the numerically-near-null direction and produced garbage.)
    c = 1e6 * np.trace(info) / n
    cov = np.linalg.inv(info + c * np.ones((n, n)) / n)
    return pd.Series(np.sqrt(np.clip(np.diag(cov), 0, None)), index=data.models)


def fisher_se_calibrated(train, link, theta) -> pd.Series:
    """Bootstrap-calibrated per-model SEs — the primary uncertainty input
    for the section 4.1 confound procedure."""
    return fisher_se(train, link, theta) / FISHER_TO_BOOTSTRAP_CALIBRATION

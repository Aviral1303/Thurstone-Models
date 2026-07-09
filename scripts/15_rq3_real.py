"""RQ3 REAL-DATA experiment — run exactly per logs/RQ3_PREANALYSIS.md
(pre-committed sections 1-5.1; user go-ahead 2026-07-11).

13 rolling windows on the RQ1 checkpoint grid; BT + lattice at units
0.5855 (primary) / 0.1 / 0.8; decisive-only conditional scoring;
full-population verdict per section 4 classifier (vote-weighted pooling,
window-cluster bootstrap); recent/established strata; noise-based bin
pooling (section 5.1, inverse-variance); reliability-slope diagnostic.
Section 4.1 correction runs only if a positive verdict fires.

Outputs:
  results/tables/rq3_window_table_{unit}.csv
  results/tables/rq3_pooled_verdicts.csv
  results/tables/rq3_bin_pooled.csv
  results/tables/rq3_reliability.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fit import fit_gaplink  # noqa: E402
from lattice_link import LatticeLink, LogisticLink  # noqa: E402
from rq3_eval import (  # noqa: E402
    MPD_LOGLOSS,
    classify,
    evaluate_window,
    pooled_estimate,
    window_table,
)

FINAL_CUTOFF = 1723479651.547
SEED = 20260712

battles = pd.read_parquet(ROOT / "data" / "processed" / "clean_battle_20240814.parquet")
RENAMES = {"chatgpt-4o-latest": "chatgpt-4o-latest-2024-08-08"}
battles = battles.assign(model_a=battles["model_a"].replace(RENAMES),
                         model_b=battles["model_b"].replace(RENAMES))
dedup = battles[battles["dedup_sampled"]]

month_ends = pd.date_range("2023-08-01", "2024-08-01", freq="MS", tz="UTC")
cutoffs = [t.timestamp() for t in month_ends] + [FINAL_CUTOFF]
labels = [str((t - pd.Timedelta(days=1)).date()) for t in month_ends] + ["2024-08-12"]

# Section 5.1 bins (fixed pre-run) by window STARTING checkpoint
BINS = {
    "2023-08-31": "high", "2023-09-30": "high",
    "2023-07-31": "moderate", "2023-10-31": "moderate", "2024-01-31": "moderate",
    "2024-02-29": "moderate", "2024-07-31": "moderate",
}  # everything else: low

UNITS = {"u0.5855": 0.5855, "u0.1": 0.1, "u0.8": 0.8}
bt_link = LogisticLink()
lat_links = {name: LatticeLink(unit=u) for name, u in UNITS.items()}

# ---------------- fits and window evaluation ----------------
results: dict[str, list] = {name: [] for name in UNITS}
for k in range(len(cutoffs) - 1):
    train = dedup[dedup["tstamp"] <= cutoffs[k]]
    test = dedup[(dedup["tstamp"] > cutoffs[k]) & (dedup["tstamp"] <= cutoffs[k + 1])]
    th_bt = fit_gaplink(train, bt_link, mode="half_tie", include_both_bad=True)
    for name, link in lat_links.items():
        th_lat = fit_gaplink(train, link, mode="half_tie", include_both_bad=True)
        wr = evaluate_window(th_bt, bt_link, th_lat, link, train, test,
                             cutoffs[k], labels[k])
        results[name].append(wr)
    r0 = results["u0.5855"][-1]
    print(f"[{labels[k]}] test votes={r0.n_test_votes:,} decisive={r0.n_decisive:,} "
          f"scoreable={r0.n_scoreable:,} ties_excluded={r0.n_ties_excluded:,}")

# ---------------- per-window tables + per-vote variance for IV pooling ----------------
def window_table_with_var(res_list, stratum=None):
    wt = window_table(res_list, stratum=stratum)
    variances, bins = [], []
    for r in res_list:
        pv = r.per_vote if stratum is None else r.per_vote[r.per_vote["stratum"] == stratum]
        variances.append(float(pv["d_logloss"].var(ddof=1)) if len(pv) > 1 else np.nan)
        bins.append(BINS.get(r.label, "low"))
    wt["var_per_vote"] = variances
    wt["bin"] = bins
    return wt


def iv_pooled(wt, n_boot=10_000, seed=SEED):
    """Inverse-variance pooled mean + window-cluster bootstrap CI (section 5.1)."""
    w = wt.dropna(subset=["mean_d_logloss", "var_per_vote"])
    w = w[w["n"] > 0]
    if len(w) == 0:
        return None
    wts = w["n"].to_numpy() / w["var_per_vote"].to_numpy()
    vals = w["mean_d_logloss"].to_numpy()
    pooled = float(np.average(vals, weights=wts))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(w), size=(n_boot, len(w)))
    bs = np.average(vals[idx], weights=wts[idx], axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"pooled": pooled, "ci_lo": float(lo), "ci_hi": float(hi),
            "n_windows": len(w), "n_votes": int(w["n"].sum())}


verdicts, bin_rows = [], []
for name in UNITS:
    res = results[name]
    wt_full = window_table_with_var(res)
    wt_rec = window_table_with_var(res, "recent")
    wt_est = window_table_with_var(res, "established")
    wt_full.to_csv(ROOT / "results" / "tables" / f"rq3_window_table_{name}.csv", index=False)
    wt_rec.to_csv(ROOT / "results" / "tables" / f"rq3_window_table_{name}_recent.csv", index=False)

    pooled_full = pooled_estimate(wt_full)         # section 3: vote-weighted
    verdict = classify(wt_full, pooled_full)
    pooled_rec_all = iv_pooled(wt_rec)             # primary stratum read (5.1 weights)
    pooled_est_all = iv_pooled(wt_est)

    print(f"\n===== {name} =====")
    print(wt_full[["window", "n", "mean_d_logloss", "mean_d_brier", "bin"]]
          .round(6).to_string(index=False))
    print(f"full-pop pooled: {pooled_full['pooled']:+.3e} "
          f"CI=({pooled_full['ci_lo']:+.3e},{pooled_full['ci_hi']:+.3e}) "
          f"[{pooled_full['n_windows']} windows, {pooled_full['n_votes']:,} votes]")
    print(f"VERDICT: {verdict['verdict']} {verdict.get('directional_note', '')}")
    print(f"recent stratum pooled (IV): {pooled_rec_all}")
    verdicts.append({"unit": name, **{k: v for k, v in verdict.items() if k != 'ci'},
                     "ci_lo": verdict["ci"][0], "ci_hi": verdict["ci"][1],
                     "recent_pooled": pooled_rec_all["pooled"] if pooled_rec_all else np.nan,
                     "recent_ci_lo": pooled_rec_all["ci_lo"] if pooled_rec_all else np.nan,
                     "recent_ci_hi": pooled_rec_all["ci_hi"] if pooled_rec_all else np.nan,
                     "recent_n": pooled_rec_all["n_votes"] if pooled_rec_all else 0,
                     "established_pooled": pooled_est_all["pooled"] if pooled_est_all else np.nan})

    for bin_name in ("high", "moderate", "low"):
        for strat_name, wt_s in (("recent", wt_rec), ("established", wt_est), ("full", wt_full)):
            sub = wt_s[wt_s["bin"] == bin_name]
            p = iv_pooled(sub)
            if p is None:
                continue
            meets = p["n_votes"] >= 1500
            bin_rows.append({"unit": name, "bin": bin_name, "stratum": strat_name,
                             "pooled_x_mpd": p["pooled"] / MPD_LOGLOSS,
                             "ci_lo_x_mpd": p["ci_lo"] / MPD_LOGLOSS,
                             "ci_hi_x_mpd": p["ci_hi"] / MPD_LOGLOSS,
                             "n_windows": p["n_windows"], "n_votes": p["n_votes"],
                             "ci_bearing": meets})

pd.DataFrame(verdicts).to_csv(ROOT / "results" / "tables" / "rq3_pooled_verdicts.csv", index=False)
bins_df = pd.DataFrame(bin_rows)
bins_df.to_csv(ROOT / "results" / "tables" / "rq3_bin_pooled.csv", index=False)
print("\n===== bin-pooled (x MPD) =====")
print(bins_df.round(3).to_string(index=False))

# ---------------- reliability-slope diagnostic (section 2; diagnostic only) ----------------
from sklearn.linear_model import LogisticRegression  # noqa: E402

rel_rows = []
for k in range(len(cutoffs) - 1):
    train = dedup[dedup["tstamp"] <= cutoffs[k]]
    test = dedup[(dedup["tstamp"] > cutoffs[k]) & (dedup["tstamp"] <= cutoffs[k + 1])]
    th_bt = fit_gaplink(train, bt_link, mode="half_tie", include_both_bad=True)
    fits_ = {"bt": (th_bt, bt_link)}
    for name, link in lat_links.items():
        fits_[name] = (fit_gaplink(train, link, mode="half_tie", include_both_bad=True), link)
    dec = test[test["winner"].isin(("model_a", "model_b"))]
    for mname, (th, link) in fits_.items():
        known = dec["model_a"].isin(th.index) & dec["model_b"].isin(th.index)
        d2 = dec[known]
        g = th.reindex(d2["model_a"]).to_numpy() - th.reindex(d2["model_b"]).to_numpy()
        p = np.clip(link.f_decisive(g), 1e-9, 1 - 1e-9)
        x = np.log(p / (1 - p)).reshape(-1, 1)
        y = (d2["winner"] == "model_a").to_numpy(dtype=int)
        lr = LogisticRegression(C=np.inf, max_iter=1000).fit(x, y)
        rel_rows.append({"window": labels[k], "method": mname,
                         "reliability_slope": float(lr.coef_[0][0]), "n": len(d2)})
rel = pd.DataFrame(rel_rows)
rel_sum = rel.groupby("method").apply(
    lambda d: np.average(d["reliability_slope"], weights=d["n"]), include_groups=False)
print("\nreliability slopes (vote-weighted mean; 1.0 = calibrated, <1 overconfident):")
print(rel_sum.round(4).to_string())
rel.to_csv(ROOT / "results" / "tables" / "rq3_reliability.csv", index=False)
print("\nDONE")

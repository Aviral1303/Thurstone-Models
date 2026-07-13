"""POST-HOC (reviewer-driven): is lattice steepness just implicit
regularization? Test whether explicit ridge on BT reproduces the cold-start
protection of RQ3's pplx episode — separating link shape from shrinkage.

*** Labeled post-hoc. Uses the frozen RQ3 design (same windows, same
*** scoring); only the new ridge penalty is varied. No verdict is revised.

Part A (episode anatomy): on the 2023-11-30 training window, sweep the L2
strength lambda and record (i) BT's fitted theta for pplx-7b-online,
(ii) the held-out window mean delta of ridge-BT vs the lattice (u=0.5855),
(iii) collateral shift on established (>=1000-vote) models.

Part B (full grid): rerun all 13 RQ3 windows with ridge-BT at two
pre-chosen strengths (lambda = 1 and 10, fixed before seeing Part B
results; chosen from Part A's episode anatomy only) and classify vs the
lattice with the pre-registered classifier.

Output: results/tables/ridge_sweep_episode.csv, ridge_windows.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fit import fit_gaplink  # noqa: E402
from lattice_link import LatticeLink, LogisticLink  # noqa: E402
from rq3_eval import MPD_LOGLOSS, classify, evaluate_window, pooled_estimate, window_table  # noqa: E402

FINAL_CUTOFF = 1723479651.547
battles = pd.read_parquet(ROOT / "data" / "processed" / "clean_battle_20240814.parquet")
RENAMES = {"chatgpt-4o-latest": "chatgpt-4o-latest-2024-08-08"}
battles = battles.assign(model_a=battles["model_a"].replace(RENAMES),
                         model_b=battles["model_b"].replace(RENAMES))
dedup = battles[battles["dedup_sampled"]]

month_ends = pd.date_range("2023-08-01", "2024-08-01", freq="MS", tz="UTC")
cutoffs = [t.timestamp() for t in month_ends] + [FINAL_CUTOFF]
labels = [str((t - pd.Timedelta(days=1)).date()) for t in month_ends] + ["2024-08-12"]

bt = LogisticLink()
lat = LatticeLink(unit=0.5855)

# ---------------- Part A: episode anatomy ----------------
k = labels.index("2023-11-30")
train = dedup[dedup["tstamp"] <= cutoffs[k]]
test = dedup[(dedup["tstamp"] > cutoffs[k]) & (dedup["tstamp"] <= cutoffs[k + 1])]
th_lat = fit_gaplink(train, lat, mode="half_tie", include_both_bad=True)
votes = pd.concat([train["model_a"], train["model_b"]]).value_counts()
established = votes[votes >= 1000].index

rows = []
th0 = None
for lam in [1e-6, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
    th = fit_gaplink(train, bt, mode="half_tie", include_both_bad=True, l2=lam)
    if th0 is None:
        th0 = th
    wr = evaluate_window(th, bt, th_lat, lat, train, test, cutoffs[k], "ep")
    d = float(wr.per_vote["d_logloss"].mean())
    est_shift = float((th.reindex(established) - th0.reindex(established)
                       - (th.reindex(established) - th0.reindex(established)).median()
                       ).abs().max())
    rows.append({"lambda": lam,
                 "theta_pplx": float(th.get("pplx-7b-online", np.nan)
                                     - th.median()),
                 "theta_pplx_lat_ref": float(th_lat.get("pplx-7b-online", np.nan)
                                             - th_lat.median()),
                 "window_delta_x_mpd": d / MPD_LOGLOSS,
                 "max_established_shift": est_shift,
                 "n_scoreable": wr.n_scoreable})
    print(f"lam={lam:>7}: theta_pplx(med-centered)={rows[-1]['theta_pplx']:+.3f} "
          f"window ridgeBT-vs-lattice delta={d/MPD_LOGLOSS:+.3f}xMPD "
          f"max est. shift={est_shift:.4f}")
ep = pd.DataFrame(rows)
ep.to_csv(ROOT / "results" / "tables" / "ridge_sweep_episode.csv", index=False)

# ---------------- Part B: all windows at lambda in {1, 10} ----------------
out = []
for lam in [1.0, 10.0]:
    results = []
    for kk in range(len(cutoffs) - 1):
        tr = dedup[dedup["tstamp"] <= cutoffs[kk]]
        te = dedup[(dedup["tstamp"] > cutoffs[kk]) & (dedup["tstamp"] <= cutoffs[kk + 1])]
        th_r = fit_gaplink(tr, bt, mode="half_tie", include_both_bad=True, l2=lam)
        th_l = fit_gaplink(tr, lat, mode="half_tie", include_both_bad=True)
        results.append(evaluate_window(th_r, bt, th_l, lat, tr, te, cutoffs[kk], labels[kk]))
    wt = window_table(results)
    pooled = pooled_estimate(wt)
    v = classify(wt, pooled)
    wt["lambda"] = lam
    out.append(wt)
    print(f"\nlambda={lam}: pooled {pooled['pooled']/MPD_LOGLOSS:+.3f}xMPD "
          f"CI ({pooled['ci_lo']/MPD_LOGLOSS:+.3f},{pooled['ci_hi']/MPD_LOGLOSS:+.3f}) "
          f"verdict={v['verdict']} {v.get('directional_note','')}")
    print(wt[["window", "n", "mean_d_logloss"]].round(6).to_string(index=False))
pd.concat(out).to_csv(ROOT / "results" / "tables" / "ridge_windows.csv", index=False)
print("DONE")

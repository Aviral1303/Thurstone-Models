"""POST-HOC (reviewer-driven): does uncertainty-integrated
(posterior-predictive-style) prediction change the RQ3 picture?

Uses the pre-specified section-4.1 machinery (never triggered by a
verdict): replace plugin p = F(ghat) with ptilde = E_eps[F(ghat + eps)],
eps ~ N(0, SE_A^2 + SE_B^2), SEs = bootstrap-calibrated Fisher from the
SAME training fit; identical correction both methods; Gauss-Hermite
15-point quadrature. Recomputes per-window deltas and reliability slopes.

*** Labeled post-hoc. Answers: would posterior-predictive prediction
*** remove the steepness/plugin effects, and does it fix the shared
*** underconfidence (reliability slope 1.455)? No verdict is revised.

Output: results/tables/pp_windows.csv, pp_reliability.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fit import fit_gaplink  # noqa: E402
from lattice_link import LatticeLink, LogisticLink  # noqa: E402
from rq3_eval import MPD_LOGLOSS, fisher_se_calibrated, pooled_estimate  # noqa: E402

FINAL_CUTOFF = 1723479651.547
battles = pd.read_parquet(ROOT / "data" / "processed" / "clean_battle_20240814.parquet")
RENAMES = {"chatgpt-4o-latest": "chatgpt-4o-latest-2024-08-08"}
battles = battles.assign(model_a=battles["model_a"].replace(RENAMES),
                         model_b=battles["model_b"].replace(RENAMES))
dedup = battles[battles["dedup_sampled"]]

month_ends = pd.date_range("2023-08-01", "2024-08-01", freq="MS", tz="UTC")
cutoffs = [t.timestamp() for t in month_ends] + [FINAL_CUTOFF]
labels = [str((t - pd.Timedelta(days=1)).date()) for t in month_ends] + ["2024-08-12"]

gh_x, gh_w = np.polynomial.hermite_e.hermegauss(15)  # probabilists' Hermite
gh_w = gh_w / gh_w.sum()

links = {"bt": LogisticLink(), "lattice": LatticeLink(unit=0.5855)}

def pp_probs(link, g, sd):
    """E_eps~N(0,sd^2)[F(g+eps)] via 15-pt Gauss-Hermite, vectorized."""
    G = g[:, None] + sd[:, None] * gh_x[None, :]
    return np.clip((link.f_decisive(G) * gh_w[None, :]).sum(axis=1), 1e-12, 1 - 1e-12)

win_rows, rel_rows = [], []
for k in range(len(cutoffs) - 1):
    train = dedup[dedup["tstamp"] <= cutoffs[k]]
    test = dedup[(dedup["tstamp"] > cutoffs[k]) & (dedup["tstamp"] <= cutoffs[k + 1])]
    dec = test[test["winner"].isin(("model_a", "model_b"))]
    lls, preds = {}, {}
    for name, link in links.items():
        th = fit_gaplink(train, link, mode="half_tie", include_both_bad=True)
        se = fisher_se_calibrated(train, link, th)
        known = dec["model_a"].isin(th.index) & dec["model_b"].isin(th.index)
        d2 = dec[known]
        g = (th.reindex(d2["model_a"]).to_numpy()
             - th.reindex(d2["model_b"]).to_numpy())
        sd = np.sqrt(se.reindex(d2["model_a"]).to_numpy() ** 2
                     + se.reindex(d2["model_b"]).to_numpy() ** 2)
        y = (d2["winner"] == "model_a").to_numpy(dtype=float)
        for mode, p in (("plugin", np.clip(link.f_decisive(g), 1e-12, 1 - 1e-12)),
                        ("pp", pp_probs(link, g, sd))):
            ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
            lls[(name, mode)] = pd.Series(ll, index=d2.index)
            x = np.log(p / (1 - p)).reshape(-1, 1)
            lr = LogisticRegression(C=np.inf, max_iter=1000).fit(x, y.astype(int))
            rel_rows.append({"window": labels[k], "method": name, "mode": mode,
                             "reliability_slope": float(lr.coef_[0][0]), "n": len(d2)})
    common = lls[("bt", "plugin")].index.intersection(lls[("lattice", "plugin")].index)
    for mode in ("plugin", "pp"):
        d = (lls[("bt", mode)].reindex(common) - lls[("lattice", mode)].reindex(common))
        win_rows.append({"window": labels[k], "mode": mode, "n": len(common),
                         "mean_d_logloss": float(d.mean())})
    print(f"[{labels[k]}] plugin delta "
          f"{win_rows[-2]['mean_d_logloss']/MPD_LOGLOSS:+.3f} | pp "
          f"{win_rows[-1]['mean_d_logloss']/MPD_LOGLOSS:+.3f} xMPD")

wt = pd.DataFrame(win_rows)
wt.to_csv(ROOT / "results" / "tables" / "pp_windows.csv", index=False)
rel = pd.DataFrame(rel_rows)
rel.to_csv(ROOT / "results" / "tables" / "pp_reliability.csv", index=False)
for mode in ("plugin", "pp"):
    sub = wt[wt["mode"] == mode].rename(columns={"mean_d_logloss": "mean_d_logloss"})
    pooled = pooled_estimate(sub.assign(mean_d_logloss=sub["mean_d_logloss"]))
    print(f"{mode}: pooled {pooled['pooled']/MPD_LOGLOSS:+.3f}xMPD "
          f"CI ({pooled['ci_lo']/MPD_LOGLOSS:+.3f},{pooled['ci_hi']/MPD_LOGLOSS:+.3f})")
print("\nreliability (vote-weighted mean):")
print(rel.groupby(["method", "mode"]).apply(
    lambda d: np.average(d.reliability_slope, weights=d.n), include_groups=False).round(4).to_string())
print("DONE")

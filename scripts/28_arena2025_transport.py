"""GENERALIZATION RUN: transport the pre-registered RQ3/RQ4 designs,
verbatim in structure, to the 2025 Arena release (135,634 battles, 53
models, 14 weeks). Weekly rolling windows replace monthly (the data span
is 14 weeks; this yields the same 13-window grid). Labeled: the DESIGNS
are the pre-registered ones; their application to this dataset is a
post-hoc generalization check, not itself pre-registered.

Parts:
  A. Full-population side-by-side consistency (BT vs lattice half-tie).
  B. Tie-parameter profiles: unit on first 3 weeks (look-ahead-free
     primary, mirroring the first-3-months rule) and on the full sample;
     per-window profiled nu/unit trajectories.
  C. RQ3 transport: weekly windows, BT vs lattice at the three units,
     pre-registered classifier (MPD 4e-4), same sign convention.
  D. RQ4 transport: weekly windows, native trinomial with per-window
     profiled tie parameters, classifier (MPD 3e-4), band widths.

Outputs: results/tables/arena2025_*.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from davidson_link import DavidsonLink  # noqa: E402
from fit import fit_gaplink, profile_lattice_unit  # noqa: E402
from lattice_link import LatticeLink, LogisticLink  # noqa: E402
from rq3_eval import MPD_LOGLOSS, classify, evaluate_window, pooled_estimate, window_table  # noqa: E402
from rq4_eval import MPD_RQ4, evaluate_window_trinomial, relabel_verdict, window_table_trinomial  # noqa: E402

T = ROOT / "results" / "tables"
df = pd.read_parquet(ROOT / "data" / "processed" / "arena2025_140k.parquet")
t0 = float(df["tstamp"].min())
WEEK = 7 * 86400.0
cutoffs = [t0 + (w + 1) * WEEK for w in range(13)]
labels = [f"wk{w+1:02d}" for w in range(13)]
bt = LogisticLink()

# ---------- A: consistency ----------
th_bt = fit_gaplink(df, bt, mode="half_tie", include_both_bad=True)
th_lat = fit_gaplink(df, LatticeLink(unit=0.5855), mode="half_tie", include_both_bad=True)
common = th_bt.index
sp = spearmanr(th_bt.reindex(common), th_lat.reindex(common)).statistic
kt = kendalltau(th_bt.reindex(common), th_lat.reindex(common)).statistic
r_bt = th_bt.rank(ascending=False)
r_lat = th_lat.reindex(th_bt.index).rank(ascending=False)
maxmove = int((r_bt - r_lat).abs().max())
print(f"[A] consistency: spearman={sp:.6f} kendall={kt:.5f} max rank move={maxmove}")
pd.DataFrame([{"spearman": sp, "kendall": kt, "max_rank_move": maxmove,
               "n_models": len(common)}]).to_csv(T / "arena2025_consistency.csv", index=False)

# ---------- B: tie-parameter profiles ----------
UNITS_GRID = np.geomspace(0.2, 1.4, 8)
NUS_GRID = np.geomspace(0.15, 1.5, 8)
early = df[df["tstamp"] <= t0 + 3 * WEEK]
u_early, _, _ = profile_lattice_unit(early, UNITS_GRID,
                                     make_link=lambda v: LatticeLink(unit=float(v), g_step=0.005),
                                     mode="native")
u_full, _, _ = profile_lattice_unit(df, UNITS_GRID,
                                    make_link=lambda v: LatticeLink(unit=float(v), g_step=0.005),
                                    mode="native")
print(f"[B] profiled unit: first-3-weeks {u_early:.4f} | full-sample {u_full:.4f}")

# ---------- C: RQ3 transport ----------
UNITS = {"u0.1": 0.1, f"u{u_early:.4f}_early": u_early, f"u{u_full:.4f}_full": u_full}
rq3_rows, verdicts = [], []
fits_cache = {}
for name, u in UNITS.items():
    lat = LatticeLink(unit=u)
    results = []
    for k in range(len(cutoffs) - 1):
        tr = df[df["tstamp"] <= cutoffs[k]]
        te = df[(df["tstamp"] > cutoffs[k]) & (df["tstamp"] <= cutoffs[k + 1])]
        key = ("bt", k)
        if key not in fits_cache:
            fits_cache[key] = fit_gaplink(tr, bt, mode="half_tie", include_both_bad=True)
        th_b = fits_cache[key]
        th_l = fit_gaplink(tr, lat, mode="half_tie", include_both_bad=True)
        results.append(evaluate_window(th_b, bt, th_l, lat, tr, te, cutoffs[k], labels[k]))
    wt = window_table(results)
    pooled = pooled_estimate(wt)
    v = classify(wt, pooled)
    wt["unit"] = name
    rq3_rows.append(wt)
    verdicts.append({"experiment": "rq3", "unit": name,
                     "pooled_x_mpd": pooled["pooled"] / MPD_LOGLOSS,
                     "ci_lo_x_mpd": pooled["ci_lo"] / MPD_LOGLOSS,
                     "ci_hi_x_mpd": pooled["ci_hi"] / MPD_LOGLOSS,
                     "n_bt_better": v["n_bt_better"], "verdict": v["verdict"],
                     "note": v.get("directional_note", "")})
    print(f"[C] {name}: pooled {pooled['pooled']/MPD_LOGLOSS:+.3f}xMPD "
          f"CI ({pooled['ci_lo']/MPD_LOGLOSS:+.3f},{pooled['ci_hi']/MPD_LOGLOSS:+.3f}) "
          f"BT better {v['n_bt_better']}/13 verdict={v['verdict']} {v.get('directional_note','')}")
pd.concat(rq3_rows).to_csv(T / "arena2025_rq3_windows.csv", index=False)

# ---------- D: RQ4 transport ----------
def band_half_width(link):
    gs = np.linspace(0, 8, 4001)
    pt = link.p_tie(gs)
    p0 = float(pt[0])
    below = np.where(pt <= 0.5 * p0)[0]
    return float(gs[below[0]]) if len(below) else np.nan

results4, traj = [], []
for k in range(len(cutoffs) - 1):
    tr = df[df["tstamp"] <= cutoffs[k]]
    te = df[(df["tstamp"] > cutoffs[k]) & (df["tstamp"] <= cutoffs[k + 1])]
    nu_hat, th_d, _ = profile_lattice_unit(tr, NUS_GRID,
                                           make_link=lambda v: DavidsonLink(nu=float(v)),
                                           mode="native")
    u_hat, th_l, _ = profile_lattice_unit(tr, UNITS_GRID,
                                          make_link=lambda v: LatticeLink(unit=float(v), g_step=0.005),
                                          mode="native")
    lk_d, lk_l = DavidsonLink(nu=nu_hat), LatticeLink(unit=u_hat, g_step=0.005)
    results4.append(evaluate_window_trinomial(th_d, lk_d, th_l, lk_l, te, labels[k]))
    traj.append({"window": labels[k], "nu_hat": nu_hat, "unit_hat": u_hat,
                 "dav_halfwidth": band_half_width(lk_d),
                 "lat_halfwidth": band_half_width(lk_l)})
    print(f"[D {labels[k]}] nu={nu_hat:.4f} u={u_hat:.4f} "
          f"d={results4[-1].per_vote['d_logloss'].mean()/MPD_RQ4:+.3f}xMPD")
wt4 = window_table_trinomial(results4)
pooled4 = pooled_estimate(wt4)
v4 = classify(wt4, pooled4, mpd=MPD_RQ4)
v4["verdict"] = relabel_verdict(v4["verdict"])
verdicts.append({"experiment": "rq4", "unit": "profiled",
                 "pooled_x_mpd": pooled4["pooled"] / MPD_RQ4,
                 "ci_lo_x_mpd": pooled4["ci_lo"] / MPD_RQ4,
                 "ci_hi_x_mpd": pooled4["ci_hi"] / MPD_RQ4,
                 "n_bt_better": v4["n_bt_better"], "verdict": v4["verdict"],
                 "note": v4.get("directional_note", "")})
print(f"[D] pooled {pooled4['pooled']/MPD_RQ4:+.3f}xMPD "
      f"CI ({pooled4['ci_lo']/MPD_RQ4:+.3f},{pooled4['ci_hi']/MPD_RQ4:+.3f}) "
      f"verdict={v4['verdict']} {v4.get('directional_note','')}")
wt4.to_csv(T / "arena2025_rq4_windows.csv", index=False)
pd.DataFrame(traj).to_csv(T / "arena2025_rq4_trajectories.csv", index=False)
pd.DataFrame(verdicts).to_csv(T / "arena2025_verdicts.csv", index=False)
print("DONE")

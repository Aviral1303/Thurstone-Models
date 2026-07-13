"""Stress-test the profile boundary that the both-bad entanglement
measurement leaned on (punch-list item, labeled post-hoc): rerun the
both-bad variant's per-window lattice profile with the grid extended from
1.4 to 2.6. If the fitted width comes interior and the decisive-channel
degradation persists, the entanglement cost stands on firmer ground; if
the width still pins, the boundary was hiding the true optimum.

Output: results/tables/bothbad_boundary_stress.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fit import fit_gaplink, profile_lattice_unit  # noqa: E402
from lattice_link import LatticeLink  # noqa: E402

MPD3 = 4e-4
FINAL_CUTOFF = 1723479651.547
battles = pd.read_parquet(ROOT / "data" / "processed" / "clean_battle_20240814.parquet")
RENAMES = {"chatgpt-4o-latest": "chatgpt-4o-latest-2024-08-08"}
battles = battles.assign(model_a=battles["model_a"].replace(RENAMES),
                         model_b=battles["model_b"].replace(RENAMES))
dedup = battles[battles["dedup_sampled"]]

month_ends = pd.date_range("2023-08-01", "2024-08-01", freq="MS", tz="UTC")
cutoffs = [t.timestamp() for t in month_ends] + [FINAL_CUTOFF]
labels = [str((t - pd.Timedelta(days=1)).date()) for t in month_ends] + ["2024-08-12"]

GRID = np.geomspace(0.2, 2.6, 10)
old = pd.read_csv(ROOT / "results" / "tables" / "rq4_param_trajectories.csv")
old_bb = old[old.variant == "robustness_bothbad"].set_index("window")

rows = []
for k in range(len(cutoffs) - 1):
    train = dedup[dedup["tstamp"] <= cutoffs[k]]
    test = dedup[(dedup["tstamp"] > cutoffs[k]) & (dedup["tstamp"] <= cutoffs[k + 1])]
    u_hat, th_l, prof = profile_lattice_unit(
        train, GRID, make_link=lambda v: LatticeLink(unit=float(v), g_step=0.005),
        mode="native", include_both_bad=True)
    lk = LatticeLink(unit=u_hat, g_step=0.005)
    # decisive-channel held-out log loss under the both-bad fit
    dec = test[test["winner"].isin(("model_a", "model_b"))]
    known = dec["model_a"].isin(th_l.index) & dec["model_b"].isin(th_l.index)
    d2 = dec[known]
    g = th_l.reindex(d2["model_a"]).to_numpy() - th_l.reindex(d2["model_b"]).to_numpy()
    p = np.clip(lk.f_decisive(g), 1e-12, 1 - 1e-12)
    y = (d2["winner"] == "model_a").to_numpy(dtype=float)
    ll = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
    # main-variant fit for the same window (tie-only), for the degradation delta
    u_m, th_m, _ = profile_lattice_unit(
        train, np.geomspace(0.2, 1.4, 8),
        make_link=lambda v: LatticeLink(unit=float(v), g_step=0.005), mode="native")
    lk_m = LatticeLink(unit=u_m, g_step=0.005)
    g_m = th_m.reindex(d2["model_a"]).to_numpy() - th_m.reindex(d2["model_b"]).to_numpy()
    p_m = np.clip(lk_m.f_decisive(g_m), 1e-12, 1 - 1e-12)
    ll_m = float(np.mean(-(y * np.log(p_m) + (1 - y) * np.log(1 - p_m))))
    rows.append({"window": labels[k],
                 "u_bothbad_old_grid": float(old_bb.loc[labels[k], "unit_hat"]),
                 "u_bothbad_extended": u_hat,
                 "interior": bool(u_hat < 2.55),
                 "u_main": u_m,
                 "decisive_ll_bothbad": ll, "decisive_ll_main": ll_m,
                 "degradation_x_mpd": (ll - ll_m) / MPD3})
    print(f"[{labels[k]}] old u={rows[-1]['u_bothbad_old_grid']:.3f} -> extended "
          f"u={u_hat:.4f} ({'interior' if rows[-1]['interior'] else 'AT BOUND'}) "
          f"decisive degradation {(ll-ll_m)/MPD3:+.2f}xMPD")

out = pd.DataFrame(rows)
out.to_csv(ROOT / "results" / "tables" / "bothbad_boundary_stress.csv", index=False)
was_ceiling = out[out.u_bothbad_old_grid >= 1.399]
print(f"\nformerly-pinned windows (n={len(was_ceiling)}): "
      f"now interior {int(was_ceiling.interior.sum())}/{len(was_ceiling)}; "
      f"mean degradation {was_ceiling.degradation_x_mpd.mean():+.2f}xMPD "
      f"(old measurement lattice-side: +4.75)")
print("DONE")

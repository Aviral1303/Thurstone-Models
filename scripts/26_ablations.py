"""POST-HOC ablations (reviewer-driven): sensitivity of headline machinery
to (i) bootstrap resample count, (ii) lattice grid resolution,
(iii) tie-parameter profile grid. Labeled; no verdict revised.

Outputs: results/tables/ablations.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fit import fit_gaplink, profile_lattice_unit  # noqa: E402
from lattice_link import LatticeLink, LogisticLink  # noqa: E402
from rq3_eval import MPD_LOGLOSS, pooled_estimate  # noqa: E402

T = ROOT / "results" / "tables"
rows = []

# ---- (i) bootstrap size on the RQ3 window table (u0.5855) ----
wt = pd.read_csv(T / "rq3_window_table_u0.5855.csv")
for B in (1_000, 5_000, 10_000, 50_000):
    p = pooled_estimate(wt, n_boot=B)
    rows.append({"ablation": "bootstrap_B", "setting": B,
                 "value": f"CI ({p['ci_lo']/MPD_LOGLOSS:+.3f},{p['ci_hi']/MPD_LOGLOSS:+.3f})xMPD"})
    print(rows[-1])

# ---- (ii) lattice resolution: curve agreement + full-pop refit ----
gs = np.linspace(-3, 3, 601)
base = LatticeLink(unit=0.5855)  # L=500, g_step=0.02
for L, g_step in ((250, 0.02), (1000, 0.02), (500, 0.005)):
    alt = LatticeLink(unit=0.5855, L=L, g_step=g_step)
    supF = float(np.max(np.abs(base.f_decisive(gs) - alt.f_decisive(gs))))
    rows.append({"ablation": "lattice_resolution_curve",
                 "setting": f"L={L},g_step={g_step}",
                 "value": f"sup|dF|={supF:.2e}"})
    print(rows[-1])

battles = pd.read_parquet(ROOT / "data" / "processed" / "clean_battle_20240814.parquet")
battles = battles.assign(model_a=battles["model_a"].replace({"chatgpt-4o-latest": "chatgpt-4o-latest-2024-08-08"}),
                         model_b=battles["model_b"].replace({"chatgpt-4o-latest": "chatgpt-4o-latest-2024-08-08"}))
dedup = battles[battles["dedup_sampled"]]
th_500 = fit_gaplink(dedup, LatticeLink(unit=0.5855, L=500), mode="half_tie", include_both_bad=True)
th_250 = fit_gaplink(dedup, LatticeLink(unit=0.5855, L=250), mode="half_tie", include_both_bad=True)
sp = spearmanr(th_500, th_250.reindex(th_500.index)).statistic
mx = float((th_500 - th_250.reindex(th_500.index)
            - (th_500 - th_250.reindex(th_500.index)).median()).abs().max())
rows.append({"ablation": "lattice_resolution_fullfit", "setting": "L=250 vs 500",
             "value": f"spearman={sp:.6f}, max|dtheta|={mx:.5f}"})
print(rows[-1])

# ---- (iii) profile grid density on the final training window ----
FINAL_CUTOFF = 1723479651.547
tr = dedup[dedup["tstamp"] <= FINAL_CUTOFF]
for pts in (8, 12, 16):
    grid = np.geomspace(0.2, 1.4, pts)
    u_hat, _, _ = profile_lattice_unit(tr, grid,
                                       make_link=lambda v: LatticeLink(unit=float(v), g_step=0.005),
                                       mode="native")
    rows.append({"ablation": "profile_grid", "setting": f"{pts}pts",
                 "value": f"u_hat={u_hat:.4f}"})
    print(rows[-1])

pd.DataFrame(rows).to_csv(T / "ablations.csv", index=False)
print("DONE")

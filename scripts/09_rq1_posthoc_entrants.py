"""POST-HOC EXPLORATORY: entrant-intensity slice of the RQ1 windows.

*** NOT part of the pre-registered RQ1_SPEC.md (approved 2026-07-10). This
*** check was requested after seeing the null result. Whatever it shows is
*** at most a lead for future pre-registered work, not a finding of this
*** paper, and does not modify the confirmatory RQ1 result.

Question: does the (null) between-method stability difference, or stability
itself, co-vary with how many new models entered the window / how much vote
share they absorbed?

Method: Spearman correlations between per-window covariates (n_entrants,
entrant_vote_share) and (a) per-method tau_b, (b) the paired lattice-minus-BT
differences in tau_b and in mean|dtheta|_med. delta=1, incumbents>=1000.
n=13 windows — treat every number below as descriptive.

Output: results/tables/rq1_posthoc_entrant_intensity.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
ELO = 400 / np.log(10)

m = pd.read_csv(ROOT / "results" / "tables" / "rq1_metrics.csv")
cov = pd.read_csv(ROOT / "results" / "tables" / "rq1_window_covariates.csv")

sub = m[(m.delta == 1) & (m.incumbents == "votes>=1000")]
cov1 = cov[cov.delta == 1].set_index("T")
k = sub.pivot(index="T", columns="method", values="kendall")
mag = sub.pivot(index="T", columns="method", values="mean_abs_dtheta_med") * ELO

tbl = pd.DataFrame({
    "n_entrants": cov1["n_entrants"],
    "entrant_vote_share": cov1["entrant_vote_share"],
    "tau_bt": k["bt"],
    "tau_lat01": k["lattice_u0.1"],
    "d_tau_lat01_bt": k["lattice_u0.1"] - k["bt"],
    "d_mag_lat01_bt_elo": mag["lattice_u0.1"] - mag["bt"],
})
print(tbl.round(4).to_string())

rows = []
for target in ["tau_bt", "tau_lat01", "d_tau_lat01_bt", "d_mag_lat01_bt_elo"]:
    for covar in ["n_entrants", "entrant_vote_share"]:
        r = spearmanr(tbl[covar], tbl[target])
        rows.append({"target": target, "covariate": covar,
                     "spearman": r.statistic, "p_descriptive": r.pvalue, "n": len(tbl)})
res = pd.DataFrame(rows)
print("\n", res.round(4).to_string(index=False))
res.to_csv(ROOT / "results" / "tables" / "rq1_posthoc_entrant_intensity.csv", index=False)

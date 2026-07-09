"""RQ3 pre-analysis: plausible-effect ceiling of the lattice family vs BT.

For each candidate lattice generating link (unit, skew) and ability-gap
spread, compute the minimum expected per-vote KL divergence between the
lattice decisive link and the BEST-FIT (scale-free) logistic — i.e. the
largest calibration advantage the lattice family could possibly have over
BT if the world were exactly lattice-Thurstone. This bounds RQ3's plausible
effect size BEFORE any real data is scored (used in RQ3_PREANALYSIS.md §3/§4).

Note the best-fit logistic here has one free parameter (scale); the real BT
fit has a free ability per model, which can only absorb MORE of the
difference — so these numbers are upper bounds on the in-family advantage.

Output: results/tables/rq3_family_effect_ceiling.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lattice_link import LatticeLink  # noqa: E402

MPD = 4e-4
gs = np.linspace(-8, 8, 401)


def min_kl_vs_logistic(gen_link, sigma_g: float):
    w = np.exp(-gs ** 2 / (2 * sigma_g ** 2))
    w /= w.sum()
    q = np.clip(gen_link.f_decisive(gs), 1e-12, 1 - 1e-12)

    def kl_at(s):
        p = np.clip(1 / (1 + np.exp(-gs / s)), 1e-12, 1 - 1e-12)
        return float(np.sum(w * (q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p)))))

    r = minimize_scalar(kl_at, bounds=(0.2, 5.0), method="bounded")
    return r.fun, r.x


rows = []
for (u, a) in [(0.1, 0.0), (0.4, 0.0), (0.5855, 0.0), (0.8, 0.0), (1.2, 0.0),
               (0.8, 4.0), (0.8, 8.0), (1.2, 6.0), (0.4, 8.0)]:
    lk = LatticeLink(unit=u, skew_a=a)
    for sg in [2.1, 3.0, 4.0]:
        kl, s = min_kl_vs_logistic(lk, sg)
        rows.append({"unit": u, "skew_a": a, "sigma_gap": sg,
                     "min_kl_nats": kl, "x_mpd": kl / MPD, "best_logistic_scale": s})

out = pd.DataFrame(rows)
print(out.round(6).to_string(index=False))
print(f"\nfamily ceiling: {out.min_kl_nats.max():.2e} nats "
      f"({out.x_mpd.max():.2f}x MPD) at "
      f"{out.loc[out.min_kl_nats.idxmax(), ['unit', 'skew_a', 'sigma_gap']].to_dict()}")
out.to_csv(ROOT / "results" / "tables" / "rq3_family_effect_ceiling.csv", index=False)

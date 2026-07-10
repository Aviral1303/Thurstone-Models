"""RQ4 pre-analysis: analytic ceiling on the trinomial calibration
difference between lattice dead-heat and Davidson tie mechanisms.

Same boundary as RQ3's real-input grounding: uses ONLY the empirical gap
distribution (from completed training fits) and link mathematics — no
held-out outcomes, no Davidson fitting on real votes.

For lattice-truth at unit u (trinomial W,D,L over gaps), find the Davidson
(scale s, nu) minimizing expected KL over the empirical gap distribution.
That minimum is the largest trinomial log-loss advantage the lattice tie
mechanism could have over a best-fit Davidson if the world were exactly
lattice — the RQ4 analog of RQ3's effect ceiling. Also computed in the
reverse direction (Davidson-truth, best-fit lattice over units).

Output: results/tables/rq4_tie_ceiling.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lattice_link import LatticeLink  # noqa: E402

MPD_RQ4 = 3e-4

battles = pd.read_parquet(ROOT / "data" / "processed" / "clean_battle_20240814.parquet")
dedup = battles[battles["dedup_sampled"]]
fits_full = pd.read_csv(ROOT / "results" / "tables" / "full_fits_20240814.csv", index_col="model")
g_emp = (fits_full["lattice_theta"].reindex(dedup["model_a"]).to_numpy()
         - fits_full["lattice_theta"].reindex(dedup["model_b"]).to_numpy())
hist, edges = np.histogram(g_emp, bins=400, range=(-8, 8))
w = hist / hist.sum()
centers = 0.5 * (edges[:-1] + edges[1:])


def davidson_probs(g, s, nu):
    e = np.exp(g / (2 * s))
    denom = e + 1 / e + nu
    return e / denom, nu / denom, (1 / e) / denom  # win, tie, loss


def lattice_trinomial(unit):
    lk = LatticeLink(unit=unit)
    W = lk.p_win(centers)
    D = lk.p_tie(centers)
    L = lk.p_loss(centers)
    s = W + D + L
    return W / s, D / s, L / s


def kl_trinomial(P, Q):
    out = 0.0
    for p, q in zip(P, Q):
        p_ = np.clip(p, 1e-12, 1)
        q_ = np.clip(q, 1e-12, 1)
        out += np.sum(w * p_ * np.log(p_ / q_))
    return float(out)


rows = []
for unit in (0.5855, 0.8):
    P = lattice_trinomial(unit)

    def obj(x):
        s, nu = np.exp(x)  # positive params
        return kl_trinomial(P, davidson_probs(centers, s, nu))

    best = min((minimize(obj, x0, method="Nelder-Mead") for x0 in
                ([0.0, -1.0], [-0.5, 0.0], [0.3, -2.0])), key=lambda r: r.fun)
    s_hat, nu_hat = np.exp(best.x)
    rows.append({"direction": f"lattice_truth_u{unit}", "min_kl_nats": best.fun,
                 "x_mpd": best.fun / MPD_RQ4,
                 "bestfit_scale": s_hat, "bestfit_nu": nu_hat})
    print(rows[-1])

# reverse: Davidson truth (nu matching ~20% tie share at small gaps), best lattice unit
for nu_true in (0.5, 0.55):
    Q_true = davidson_probs(centers, 0.7, nu_true)  # scale ~ empirical best-fit logistic
    tie_at_0 = nu_true / (2 + nu_true)
    best_kl, best_u = np.inf, None
    for u in np.geomspace(0.2, 1.4, 25):
        kl = kl_trinomial(Q_true, lattice_trinomial(u))
        if kl < best_kl:
            best_kl, best_u = kl, u
    rows.append({"direction": f"davidson_truth_nu{nu_true}(tie0={tie_at_0:.3f})",
                 "min_kl_nats": best_kl, "x_mpd": best_kl / MPD_RQ4,
                 "bestfit_scale": np.nan, "bestfit_nu": best_u})
    print(rows[-1])

pd.DataFrame(rows).to_csv(ROOT / "results" / "tables" / "rq4_tie_ceiling.csv", index=False)
print("DONE")

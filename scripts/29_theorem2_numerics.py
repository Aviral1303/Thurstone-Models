"""Numerics for Theorem 2 (approximation-limited regime), computed on the
empirical vote-weighted gap distribution from the completed full fits.
Part (i): assumption-free bound E[|F-sigma|/m]. Part (ii): sharp in-family
ceiling (must reproduce rq3_ceiling_empirical.csv row).
Output: results/tables/theorem2_numerics.csv
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from lattice_link import LatticeLink

MPD = 4e-4
fits = pd.read_csv(ROOT / "results/tables/full_fits_20240814.csv", index_col="model")
b = pd.read_parquet(ROOT / "data/processed/clean_battle_20240814.parquet")
b = b[b.dedup_sampled]
b = b.assign(model_a=b.model_a.replace({"chatgpt-4o-latest": "chatgpt-4o-latest-2024-08-08"}),
             model_b=b.model_b.replace({"chatgpt-4o-latest": "chatgpt-4o-latest-2024-08-08"}))
g = (fits.lattice_theta.reindex(b.model_a).to_numpy()
     - fits.lattice_theta.reindex(b.model_b).to_numpy())
assert not np.isnan(g).any()
ceil = pd.read_csv(ROOT / "results/tables/rq3_ceiling_empirical.csv")
row = ceil[ceil.unit == 0.5855].iloc[0]
lk = LatticeLink(unit=0.5855)
F = np.clip(lk.f_decisive(g), 1e-9, 1 - 1e-9)
S = 1 / (1 + np.exp(-g / row.best_logistic_scale))
eps = np.abs(F - S)
m = np.minimum.reduce([F, 1 - F, S, 1 - S])
KL = F * np.log(F / S) + (1 - F) * np.log((1 - F) / (1 - S))
out = pd.DataFrame([{
    "unit": 0.5855, "best_logistic_scale": row.best_logistic_scale,
    "sup_abs_dp": float(eps.max()), "mean_abs_dp": float(eps.mean()),
    "part_i_bound_nats": float(np.mean(eps / m)),
    "part_i_bound_x_mpd": float(np.mean(eps / m) / MPD),
    "part_ii_vote_kl_nats": float(KL.mean()),
    "part_ii_vote_kl_x_mpd": float(KL.mean() / MPD),
    "stored_ceiling_x_mpd": float(row.x_mpd),
}])
print(out.round(6).to_string(index=False))
out.to_csv(ROOT / "results/tables/theorem2_numerics.csv", index=False)

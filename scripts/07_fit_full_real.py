"""Phase 3 real-data step: full-population lattice + BT fits, side by side.

Both fits use identical data (dedup_sampled battles, full file through
2024-08-14), identical tie treatment (half-tie mode = fastchat-equivalent),
identical per-vote MLE machinery, and the common anchor (gpt-4-0613 = 0).
Only the link differs. Consistency check (user-gated before RQ1): decisive
rank-ordering agreement between the two fits, full population and the
>=1000-votes subset.

Also runs the RQ4-groundwork unit profile on real data (native mode, 'tie'
only) and verifies the half-tie fit is insensitive to the unit choice.

Outputs:
  results/tables/full_fits_20240814.csv
  results/tables/full_fit_consistency.csv
  results/tables/unit_profile_real_20240814.csv
  results/figures/full_fit_scatter.png
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau, spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchoring import ANCHOR_MODEL, anchor  # noqa: E402
from fit import fit_gaplink, profile_lattice_unit  # noqa: E402
from lattice_link import LatticeLink, LogisticLink  # noqa: E402

battles = pd.read_parquet(ROOT / "data" / "processed" / "clean_battle_20240814.parquet")
battles = battles[battles["dedup_sampled"]]
RENAMES = {"chatgpt-4o-latest": "chatgpt-4o-latest-2024-08-08"}
battles = battles.assign(model_a=battles["model_a"].replace(RENAMES),
                         model_b=battles["model_b"].replace(RENAMES))
print(f"dedup battles, full file: {len(battles):,}")

votes_per_model = pd.concat([battles["model_a"], battles["model_b"]]).value_counts()

# ---- side-by-side half-tie fits (identical treatment, only the link differs)
t0 = time.time()
theta_bt = anchor(fit_gaplink(battles, LogisticLink(), mode="half_tie"))
print(f"BT (logistic) fit: {time.time()-t0:.1f}s, {len(theta_bt)} models")

t0 = time.time()
lat_link_default = LatticeLink()  # unit 0.1; insensitivity verified below
theta_lat = anchor(fit_gaplink(battles, lat_link_default, mode="half_tie"))
print(f"lattice fit: {time.time()-t0:.1f}s, {len(theta_lat)} models")

fits = pd.DataFrame({
    "bt_theta": theta_bt,
    "lattice_theta": theta_lat,
    "bt_elo_equiv": theta_bt * 400 / np.log(10),  # display only
    "num_votes": votes_per_model,
})
fits.index.name = "model"
fits = fits.sort_values("bt_theta", ascending=False)
fits["bt_rank"] = fits["bt_theta"].rank(ascending=False).astype(int)
fits["lattice_rank"] = fits["lattice_theta"].rank(ascending=False).astype(int)
fits["rank_diff"] = fits["bt_rank"] - fits["lattice_rank"]
fits.to_csv(ROOT / "results" / "tables" / "full_fits_20240814.csv")


def consistency(sub: pd.DataFrame, name: str) -> dict:
    sp = spearmanr(sub["bt_theta"], sub["lattice_theta"]).statistic
    kt = kendalltau(sub["bt_theta"], sub["lattice_theta"]).statistic
    rec = {"subset": name, "n_models": len(sub), "spearman": sp, "kendall": kt,
           "max_abs_rank_diff": int(sub["rank_diff"].abs().max()),
           "mean_abs_rank_diff": float(sub["rank_diff"].abs().mean())}
    print(rec)
    worst = sub.reindex(sub["rank_diff"].abs().sort_values(ascending=False).head(5).index)
    print(worst[["bt_rank", "lattice_rank", "num_votes"]].to_string())
    return rec


print("\n--- consistency: full population ---")
c1 = consistency(fits, "full")
print("\n--- consistency: >=1000 votes ---")
c2 = consistency(fits[fits["num_votes"] >= 1000], "votes>=1000")
pd.DataFrame([c1, c2]).to_csv(ROOT / "results" / "tables" / "full_fit_consistency.csv", index=False)

# ---- scatter figure
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(fits["bt_theta"], fits["lattice_theta"], s=14, alpha=0.7)
ax.set_xlabel(f"BT ability (log units, {ANCHOR_MODEL}=0)")
ax.set_ylabel(f"lattice ability ({ANCHOR_MODEL}=0)")
ax.set_title("Full-population fits, identical half-tie treatment")
for m in fits.reindex(fits["rank_diff"].abs().sort_values(ascending=False).head(3).index).index:
    ax.annotate(m, (fits.loc[m, "bt_theta"], fits.loc[m, "lattice_theta"]), fontsize=7)
fig.tight_layout()
fig.savefig(ROOT / "results" / "figures" / "full_fit_scatter.png", dpi=150)

# ---- RQ4 groundwork: profile the unit on real data (native mode, 'tie' only)
print("\n--- unit profile on real data (RQ4 groundwork) ---")
t0 = time.time()
units = np.geomspace(0.1, 1.4, 12)
best_u, theta_nat, prof = profile_lattice_unit(
    battles, units, make_link=lambda u: LatticeLink(unit=u), mode="native")
print(prof.to_string(index=False))
link_fit = LatticeLink(unit=best_u)
print(f"fitted unit: {best_u:.4f} ({time.time()-t0:.0f}s); D(0)={link_fit.p_tie(0.0):.4f}; "
      f"observed quality-tie share among non-bothbad votes: "
      f"{(battles.winner == 'tie').sum() / battles.winner.isin(['model_a','model_b','tie']).sum():.4f}")
prof["fitted_unit"] = best_u
prof.to_csv(ROOT / "results" / "tables" / "unit_profile_real_20240814.csv", index=False)

# ---- verify half-tie fit is insensitive to unit (claimed in RQ4_DESIGN.md)
theta_lat_fitu = anchor(fit_gaplink(battles, link_fit, mode="half_tie"))
common = theta_lat.index
sp_u = spearmanr(theta_lat.reindex(common), theta_lat_fitu.reindex(common)).statistic
max_shift = float((theta_lat.reindex(common) - theta_lat_fitu.reindex(common)).abs().max())
print(f"half-tie fit, unit 0.1 vs fitted {best_u:.3f}: spearman={sp_u:.6f}, "
      f"max|dtheta|={max_shift:.4f} (native units)")

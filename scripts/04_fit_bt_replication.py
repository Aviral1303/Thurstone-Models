"""Phase 2 gating checkpoint: replicate the published Arena BT ratings.

Replicates fastchat's `compute_mle_elo` (the algorithm behind the leaderboard
since Dec 2023): BT MLE via weighted logistic regression where each decisive
battle contributes weight 2 in its direction and each tie (both 'tie' and
'tie (bothbad)') contributes weight 1 in EACH direction; ratings scaled by
SCALE=400/log(10), anchored at mixtral-8x7b-instruct-v0.1 = 1114.

Battles are truncated to the pickle's last_updated_tstamp so both fits see the
same data. We fit two variants — with and without the dedup_sampled filter —
and compare each against the published ratings (Spearman, Kendall, Pearson,
MAE / max|Δ| after identical anchoring).

Also fits choix's vanilla BT (decisive votes only) as a cross-check that an
independent implementation agrees up to affine transform.

Output: results/tables/bt_replication_20240813.csv + printed report.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260709

published = pd.read_csv(ROOT / "results" / "tables" / "published_bt_20240813.csv", index_col="model")
meta = json.loads((ROOT / "results" / "tables" / "published_bt_20240813_meta.json").read_text())
battles_all = pd.read_parquet(ROOT / "data" / "processed" / "clean_battle_20240814.parquet")

cutoff = meta["last_updated_tstamp"]
battles_all = battles_all[battles_all["tstamp"] <= cutoff]

# The published board renamed this model (battle log uses the unsuffixed name).
RENAMES = {"chatgpt-4o-latest": "chatgpt-4o-latest-2024-08-08"}
battles_all = battles_all.assign(
    model_a=battles_all["model_a"].replace(RENAMES),
    model_b=battles_all["model_b"].replace(RENAMES),
)
print(f"battles at/before published cutoff ({meta['last_updated_datetime']}): {len(battles_all):,}")


import sys
sys.path.insert(0, str(ROOT / "src"))
from bt_baseline import compute_mle_elo  # noqa: E402


def compare(ours: pd.Series, name: str) -> dict:
    common = published.index.intersection(ours.index)
    a = published.loc[common, "rating"]
    b = ours.loc[common]
    res = {
        "variant": name,
        "n_common": len(common),
        "n_ours_only": len(ours.index.difference(published.index)),
        "n_pub_only": len(published.index.difference(ours.index)),
        "spearman": spearmanr(a, b).statistic,
        "kendall": kendalltau(a, b).statistic,
        "pearson": pearsonr(a, b).statistic,
        "mae": float(np.mean(np.abs(a - b))),
        "max_abs": float(np.max(np.abs(a - b))),
    }
    worst = (a - b).abs().sort_values(ascending=False).head(5)
    print(f"\n--- {name} ---")
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in res.items()})
    print("largest deviations (rating points):")
    print(worst.to_string())
    return res


results = []

# Variant 1: dedup_sampled filter (what the 2024 pipeline used)
dedup = battles_all[battles_all["dedup_sampled"]]
print(f"dedup_sampled battles: {len(dedup):,}")
bt_dedup = compute_mle_elo(dedup)
results.append(compare(bt_dedup, "fastchat-LR, dedup_sampled"))

# Variant 2: no dedup filter
bt_nodedup = compute_mle_elo(battles_all)
results.append(compare(bt_nodedup, "fastchat-LR, all battles"))

# Cross-check: choix BT on decisive votes only (independent implementation).
import choix  # noqa: E402

best = dedup if True else battles_all
models_sorted = sorted(set(best["model_a"]) | set(best["model_b"]))
midx = {m: i for i, m in enumerate(models_sorted)}
dec = best[best["winner"].isin(["model_a", "model_b"])]
pairs = [
    (midx[r.model_a], midx[r.model_b]) if r.winner == "model_a" else (midx[r.model_b], midx[r.model_a])
    for r in dec.itertuples()
]
theta = choix.opt_pairwise(len(models_sorted), pairs, alpha=1e-6)
choix_elo = pd.Series(theta, index=models_sorted) * 400 / math.log(10)
if "mixtral-8x7b-instruct-v0.1" in choix_elo.index:
    choix_elo += 1114 - choix_elo["mixtral-8x7b-instruct-v0.1"]
results.append(compare(choix_elo.sort_values(ascending=False), "choix BT, decisive only, dedup"))

out = pd.DataFrame(results)
out_path = ROOT / "results" / "tables" / "bt_replication_20240813.csv"
out.to_csv(out_path, index=False)

# Also persist our best-variant ratings for later phases
fits = pd.DataFrame({
    "bt_dedup": bt_dedup,
    "bt_nodedup": bt_nodedup,
    "choix_decisive_dedup": choix_elo,
    "published": published["rating"],
})
fits.index.name = "model"
fits.to_csv(ROOT / "results" / "tables" / "bt_fitted_ratings_20240813.csv")
print(f"\nwrote {out_path}")

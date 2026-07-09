"""Fix the era-bin rule for RQ3 pooled stratum reporting (pre-fit, training
quantities only): per-checkpoint recent-entrant cohort SE vs era-average SE,
plus stratum vote counts (last-28-training-days proxy).

The bin rule is tied to the mechanism found in scripts/13: pooled stratum
reads are needed exactly where the recent cohort is noisy AND small.

Output: results/tables/rq3_bin_rule.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fit import fit_gaplink  # noqa: E402
from lattice_link import LogisticLink  # noqa: E402
from rq3_eval import RECENT_DAYS, SECONDS_PER_DAY, fisher_se_calibrated  # noqa: E402

ELO_PER_NAT = 400 / np.log(10)
FINAL_CUTOFF = 1723479651.547

battles = pd.read_parquet(ROOT / "data" / "processed" / "clean_battle_20240814.parquet")
RENAMES = {"chatgpt-4o-latest": "chatgpt-4o-latest-2024-08-08"}
battles = battles.assign(model_a=battles["model_a"].replace(RENAMES),
                         model_b=battles["model_b"].replace(RENAMES))
dedup = battles[battles["dedup_sampled"]]

month_ends = pd.date_range("2023-08-01", "2024-08-01", freq="MS", tz="UTC")
cutoffs = [t.timestamp() for t in month_ends] + [FINAL_CUTOFF]
labels = [str((t - pd.Timedelta(days=1)).date()) for t in month_ends] + ["2024-08-12"]

rows = []
for k, cut in enumerate(cutoffs):
    train = dedup[dedup["tstamp"] <= cut]
    first_seen = pd.concat([
        train.groupby("model_a")["tstamp"].min(),
        train.groupby("model_b")["tstamp"].min(),
    ], axis=1).min(axis=1)
    recent = first_seen[first_seen >= cut - RECENT_DAYS * SECONDS_PER_DAY].index
    th = fit_gaplink(train, LogisticLink(), mode="half_tie", include_both_bad=True)
    se = fisher_se_calibrated(train, LogisticLink(), th)
    last28 = train[train["tstamp"] > cut - RECENT_DAYS * SECONDS_PER_DAY]
    strat = last28[last28["model_a"].isin(recent) | last28["model_b"].isin(recent)]
    strat_dec = strat["winner"].isin(["model_a", "model_b"]).sum()
    ratio = float(se.reindex(recent.intersection(se.index)).median() / se.median()) if len(recent) else np.nan
    rows.append({
        "checkpoint": labels[k],
        "n_recent_models": len(recent),
        "cohort_median_se_elo": float(se.reindex(recent.intersection(se.index)).median() * ELO_PER_NAT) if len(recent) else np.nan,
        "era_median_se_elo": float(se.median() * ELO_PER_NAT),
        "se_ratio": ratio,
        "stratum_votes_28d": len(strat),
        "stratum_decisive_28d": int(strat_dec),
    })
    print(rows[-1])

out = pd.DataFrame(rows)
out.to_csv(ROOT / "results" / "tables" / "rq3_bin_rule.csv", index=False)

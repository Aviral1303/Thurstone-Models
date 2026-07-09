"""POST-HOC EXPLORATORY decomposition of RQ3 (user-directed, 2026-07-12).

*** NOT pre-registered. Labeled per the RQ1-entrant-slice precedent. ***

Answers a SEPARATE, NARROWER question than the pre-registered experiment:
"excluding extreme cold-start cases, are the methods practically
equivalent?" — by excluding test votes where EITHER model has fewer than 30
training votes at the window's checkpoint. The pre-registered
full-population INCONCLUSIVE verdict stands as the headline regardless of
what this shows; this is a decomposition, not a resolution or re-verdict.

Fits are identical to scripts/15 (the filter changes scoring only).

Output: results/tables/rq3_posthoc_coldstart_filter.csv (+ per-window
tables per unit)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fit import fit_gaplink  # noqa: E402
from lattice_link import LatticeLink, LogisticLink  # noqa: E402
from rq3_eval import (  # noqa: E402
    MPD_LOGLOSS,
    classify,
    evaluate_window,
    pooled_estimate,
    window_table,
)

FINAL_CUTOFF = 1723479651.547
MIN_TRAIN_VOTES = 30

battles = pd.read_parquet(ROOT / "data" / "processed" / "clean_battle_20240814.parquet")
RENAMES = {"chatgpt-4o-latest": "chatgpt-4o-latest-2024-08-08"}
battles = battles.assign(model_a=battles["model_a"].replace(RENAMES),
                         model_b=battles["model_b"].replace(RENAMES))
dedup = battles[battles["dedup_sampled"]]

month_ends = pd.date_range("2023-08-01", "2024-08-01", freq="MS", tz="UTC")
cutoffs = [t.timestamp() for t in month_ends] + [FINAL_CUTOFF]
labels = [str((t - pd.Timedelta(days=1)).date()) for t in month_ends] + ["2024-08-12"]

UNITS = {"u0.5855": 0.5855, "u0.1": 0.1, "u0.8": 0.8}
bt_link = LogisticLink()
lat_links = {name: LatticeLink(unit=u) for name, u in UNITS.items()}

results: dict[str, list] = {name: [] for name in UNITS}
excluded_counts = []
for k in range(len(cutoffs) - 1):
    train = dedup[dedup["tstamp"] <= cutoffs[k]]
    test = dedup[(dedup["tstamp"] > cutoffs[k]) & (dedup["tstamp"] <= cutoffs[k + 1])]
    votes = pd.concat([train["model_a"], train["model_b"]]).value_counts()
    ok_models = votes[votes >= MIN_TRAIN_VOTES].index
    test_f = test[test["model_a"].isin(ok_models) & test["model_b"].isin(ok_models)]
    excluded_counts.append({"window": labels[k], "n_test": len(test),
                            "n_after_filter": len(test_f),
                            "excluded_frac": 1 - len(test_f) / max(len(test), 1)})
    th_bt = fit_gaplink(train, bt_link, mode="half_tie", include_both_bad=True)
    for name, link in lat_links.items():
        th_lat = fit_gaplink(train, link, mode="half_tie", include_both_bad=True)
        results[name].append(evaluate_window(th_bt, bt_link, th_lat, link,
                                             train, test_f, cutoffs[k], labels[k]))
    print(f"[{labels[k]}] filtered test votes {len(test):,} -> {len(test_f):,}")

rows = []
for name in UNITS:
    wt = window_table(results[name])
    wt.to_csv(ROOT / "results" / "tables" / f"rq3_posthoc_filtered_window_{name}.csv", index=False)
    pooled = pooled_estimate(wt)
    verdict = classify(wt, pooled)
    n_bt = int((wt["mean_d_logloss"] < 0).sum())
    print(f"\n===== {name} (POST-HOC filtered, min {MIN_TRAIN_VOTES} training votes) =====")
    print(wt[["window", "n", "mean_d_logloss"]].round(6).to_string(index=False))
    print(f"pooled {pooled['pooled']/MPD_LOGLOSS:+.3f}x MPD, "
          f"CI ({pooled['ci_lo']/MPD_LOGLOSS:+.3f},{pooled['ci_hi']/MPD_LOGLOSS:+.3f})x MPD; "
          f"BT better in {n_bt}/13; verdict={verdict['verdict']} "
          f"{verdict.get('directional_note', '')}")
    rows.append({"unit": name, "pooled_x_mpd": pooled["pooled"] / MPD_LOGLOSS,
                 "ci_lo_x_mpd": pooled["ci_lo"] / MPD_LOGLOSS,
                 "ci_hi_x_mpd": pooled["ci_hi"] / MPD_LOGLOSS,
                 "n_bt_better": n_bt, "verdict": verdict["verdict"],
                 "note": verdict.get("directional_note", "")})

out = pd.DataFrame(rows)
out.to_csv(ROOT / "results" / "tables" / "rq3_posthoc_coldstart_filter.csv", index=False)
pd.DataFrame(excluded_counts).to_csv(
    ROOT / "results" / "tables" / "rq3_posthoc_filter_exclusions.csv", index=False)
print("\nDONE")

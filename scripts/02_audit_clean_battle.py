"""Data audit of the clean_battle_20240814 parquet, answering the Phase 1 questions:
pairwise/listwise, ties, timestamp granularity, model/vote counts, skew,
and field-growth structure (model entry dates) needed for RQ1.
"""

from pathlib import Path

import numpy as np
import pandas as pd

P = Path(__file__).resolve().parents[1] / "data" / "processed" / "clean_battle_20240814.parquet"
df = pd.read_parquet(P)

print("=" * 70)
print(f"rows: {len(df):,}")

# --- outcome labels / ties ---
print("\nwinner label counts:")
print(df["winner"].value_counts(dropna=False).to_string())
tie_mask = df["winner"].str.startswith("tie")
print(f"\ntie share (all tie labels): {tie_mask.mean():.4f}")

# --- anony flag (leaderboard uses anony-only votes) ---
print("\nanony flag:")
print(df["anony"].value_counts(dropna=False).to_string())

# --- timestamps ---
ts = pd.to_datetime(df["tstamp"], unit="s")
print(f"\ndate range: {ts.min()}  ->  {ts.max()}")
sub_second = (df["tstamp"] % 1 != 0).mean()
print(f"share of timestamps with sub-second precision: {sub_second:.3f}")

# --- models ---
models = pd.concat([df["model_a"], df["model_b"]]).astype(str)
vc = models.value_counts()
print(f"\ndistinct models: {vc.size}")
print(f"votes-per-model: median={vc.median():,.0f} mean={vc.mean():,.0f} "
      f"min={vc.min():,} max={vc.max():,}")
print(f"top 5:\n{vc.head(5).to_string()}")
print(f"bottom 5:\n{vc.tail(5).to_string()}")
gini_sorted = np.sort(vc.values)
n = len(gini_sorted)
gini = (2 * np.arange(1, n + 1) - n - 1) @ gini_sorted / (n * gini_sorted.sum())
print(f"Gini of votes-per-model: {gini:.3f}")

# --- field growth: model entry dates (first anony battle) ---
d = df[df["anony"]].copy()
d["date"] = pd.to_datetime(d["tstamp"], unit="s").dt.floor("D")
first_a = d.groupby("model_a")["date"].min()
first_b = d.groupby("model_b")["date"].min()
entry = pd.concat([first_a, first_b], axis=1).min(axis=1)
entry_by_month = entry.dt.to_period("M").value_counts().sort_index()
print("\nnew models entering the pool, by month:")
print(entry_by_month.to_string())

# --- pair coverage ---
pair = df[["model_a", "model_b"]].apply(lambda r: tuple(sorted(r)), axis=1)
pair_counts = pair.value_counts()
print(f"\ndistinct unordered pairs observed: {len(pair_counts):,} "
      f"(complete graph would be {vc.size*(vc.size-1)//2:,})")
print(f"battles per pair: median={pair_counts.median():.0f} max={pair_counts.max():,}")
print(f"pairs with >=20 battles: {(pair_counts>=20).sum():,}; "
      f">=100: {(pair_counts>=100).sum():,}; >=500: {(pair_counts>=500).sum():,}")

# --- judges ---
print(f"\ndistinct judges (anonymized users): {df['judge'].nunique():,}")
print(f"\nlanguages: {df['language'].nunique()}; top: "
      f"{df['language'].value_counts().head(3).to_dict()}")

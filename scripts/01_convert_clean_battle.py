"""Stream-convert the LMSYS clean_battle public JSON (2GB) to a compact parquet.

Keeps only battle-level metadata needed for ranking-model research (no
conversation text exists in this file anyway; we drop nested category tags).

Input : data/raw/clean_battle_20240814_public.json
Output: data/processed/clean_battle_20240814.parquet
"""

from pathlib import Path

import ijson
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "clean_battle_20240814_public.json"
OUT = Path(__file__).resolve().parents[1] / "data" / "processed" / "clean_battle_20240814.parquet"

KEEP = ["model_a", "model_b", "winner", "judge", "turn", "anony", "language", "tstamp"]

rows = []
with open(RAW, "rb") as f:
    for rec in ijson.items(f, "item"):
        row = {k: rec.get(k) for k in KEEP}
        # dedup_tag is used by LMSYS's own leaderboard pipeline to downweight
        # duplicated prompts; keep both flags for faithful BT replication.
        dt = rec.get("dedup_tag") or {}
        row["dedup_sampled"] = bool(dt.get("sampled", False))
        row["dedup_high_freq"] = bool(dt.get("high_freq", False))
        rows.append(row)

df = pd.DataFrame(rows)
df["tstamp"] = pd.to_numeric(df["tstamp"], errors="coerce")
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"rows={len(df):,}  cols={list(df.columns)}")
print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")

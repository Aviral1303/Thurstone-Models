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
        rows.append({k: rec.get(k) for k in KEEP})

df = pd.DataFrame(rows)
df["tstamp"] = pd.to_numeric(df["tstamp"], errors="coerce")
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT, index=False)
print(f"rows={len(df):,}  cols={list(df.columns)}")
print(f"wrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")

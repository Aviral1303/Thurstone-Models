"""Download + convert lmarena-ai/arena-human-preference-140k (2025 era) to a
compact metadata parquet — the second-dataset generalization set scoped in
the Phase 1 audit. Labeled: transport of the pre-registered designs to new
data; not itself pre-registered.

Keeps only ranking-relevant columns (drops conversation text ~1.6GB).
Winner labels mapped to the clean_battle vocabulary
('both_bad' -> 'tie (bothbad)') so all existing machinery runs unchanged.

Output: data/processed/arena2025_140k.parquet
"""

import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "data" / "raw" / "hf_cache"))

from datasets import load_dataset  # noqa: E402

ds = load_dataset("lmarena-ai/arena-human-preference-140k", split="train")
df = ds.select_columns(["model_a", "model_b", "winner", "timestamp"]).to_pandas()
df["tstamp"] = pd.to_datetime(df["timestamp"]).astype("int64") / 1e9
df = df.drop(columns=["timestamp"])
df["winner"] = df["winner"].replace({"both_bad": "tie (bothbad)"})
print("rows:", len(df))
print(df["winner"].value_counts().to_string())
ts = pd.to_datetime(df["tstamp"], unit="s")
print("range:", ts.min(), "->", ts.max())
print("models:", pd.concat([df.model_a, df.model_b]).nunique())
q = (df.winner == "tie").sum() / df.winner.isin(["model_a", "model_b", "tie"]).sum()
print(f"quality-tie share (non-bothbad denom): {q:.4f}")
out = ROOT / "data" / "processed" / "arena2025_140k.parquet"
df.to_parquet(out, index=False)
print("wrote", out)

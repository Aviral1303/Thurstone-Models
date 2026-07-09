"""Extract published BT ratings from elo_results_20240813.pkl.

The pickle embeds plotly Figure objects built with an old plotly; we stub out
all plotly classes during unpickling (we only need the pandas tables).

Output: results/tables/published_bt_20240813.csv + a small JSON of metadata.
"""

import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKL = ROOT / "data" / "raw" / "elo_results_20240813.pkl"
OUT_CSV = ROOT / "results" / "tables" / "published_bt_20240813.csv"
OUT_META = ROOT / "results" / "tables" / "published_bt_20240813_meta.json"


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("plotly") or module.startswith("_plotly"):
            return _Stub
        return super().find_class(module, name)


with open(PKL, "rb") as f:
    obj = SafeUnpickler(f).load()

full = obj["text"]["full"]
df = full["leaderboard_table_df"].copy()
df.index.name = "model"
df = df.sort_values("rating", ascending=False)
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV)

meta = {
    "rating_system": full["rating_system"],
    "last_updated_datetime": full["last_updated_datetime"],
    "last_updated_tstamp": float(full["last_updated_tstamp"]),
    "n_models": int(len(df)),
}
OUT_META.write_text(json.dumps(meta, indent=2))
print(json.dumps(meta, indent=2))
print(df.head(10).to_string())

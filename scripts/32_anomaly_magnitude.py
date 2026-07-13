"""Express the RQ2b additivity anomaly in interpretable units (punch-list):
median absolute additivity violation of triples, in ability units and
Elo-equivalent, for excess vs clean blocks, next to the sampling-expected
level. No fitting; logit-link inversion of raw win rates.
Output: results/tables/anomaly_magnitude.csv
"""
import sys
from itertools import combinations
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd

ELO = 400 / np.log(10)
BLOCK_DAYS, DAY = 60, 86400.0
b = pd.read_parquet(ROOT / "data/processed/clean_battle_20240814.parquet")
b = b[b.dedup_sampled]
first = pd.concat([b.groupby("model_a").tstamp.min(), b.groupby("model_b").tstamp.min()], axis=1).min(axis=1)
last = pd.concat([b.groupby("model_a").tstamp.max(), b.groupby("model_b").tstamp.max()], axis=1).max(axis=1)
t0 = float(b.tstamp.min())

def legs(df, min_dec=100):
    dec = df[df.winner.isin(("model_a", "model_b"))]
    a_win = dec.winner == "model_a"
    f = np.where(dec.model_a < dec.model_b, dec.model_a, dec.model_b)
    s = np.where(dec.model_a < dec.model_b, dec.model_b, dec.model_a)
    fw = np.where(dec.model_a < dec.model_b, a_win, ~a_win)
    g = pd.DataFrame({"f": f, "s": s, "fw": fw}).groupby(["f", "s"]).fw.agg(["sum", "size"])
    g = g[g["size"] >= min_dec]
    return {(x, y): (r["sum"] / r["size"], int(r["size"])) for (x, y), r in g.iterrows()}

rows = []
for k in range(6, -1, -1):
    b0, b1 = t0 + k * BLOCK_DAYS * DAY, t0 + (k + 1) * BLOCK_DAYS * DAY
    stable = set(first[first <= b0].index) & set(last[last >= b1].index)
    blk = b[(b.tstamp > b0) & (b.tstamp <= b1) & b.model_a.isin(stable) & b.model_b.isin(stable)]
    L = legs(blk)
    models = sorted({m for pr in L for m in pr})
    adj = {m: set() for m in models}
    for (x, y) in L:
        adj[x].add(y); adj[y].add(x)
    for a, c, d in combinations(models, 3):
        if c in adj[a] and d in adj[a] and d in adj[c]:
            def gv(pr):
                p, n = L[pr]
                p = min(max(p, 1e-6), 1 - 1e-6)
                return np.log(p / (1 - p)), p * (1 - p) * (1 / (p * (1 - p))) ** 2 / n
            gac, vac = gv((a, d)); gab, vab = gv((a, c)); gbc, vbc = gv((c, d))
            viol = gac - gab - gbc
            rows.append({"block_start": str(pd.Timestamp(b0, unit="s").date()),
                         "viol_ability": viol, "sd_expected": np.sqrt(vab + vbc + vac)})
df = pd.DataFrame(rows)
df["excess_block"] = df.block_start.isin(["2023-08-22", "2023-12-20"])
out = df.groupby("excess_block").apply(lambda d: pd.Series({
    "n_triples": len(d),
    "median_abs_viol_ability": d.viol_ability.abs().median(),
    "median_abs_viol_elo": d.viol_ability.abs().median() * ELO,
    "median_expected_sd_elo": d.sd_expected.median() * ELO,
}), include_groups=False)
print(out.round(3).to_string())
out.to_csv(ROOT / "results/tables/anomaly_magnitude.csv")

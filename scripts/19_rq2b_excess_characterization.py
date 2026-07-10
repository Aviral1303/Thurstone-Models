"""DESCRIPTIVE / EXPLORATORY characterization of the RQ2b additivity excess.

*** Not a pre-registered test. This characterizes an anomaly (shared mean
*** z^2 ~ 1.2-1.3 across all links, concentrated in two blocks) to inform
*** the RQ2a descope decision. Nothing here is a hypothesis test.

Questions (user review 2026-07-10):
  1. Which blocks show the excess, exact date ranges?
  2. Block covariates: entry activity, judge/user population, language mix,
     tie share, high-freq-prompt share, overlap with the (production)
     dedup-filter switchover epoch (May-Jun 2024).
  3. Concentration: specific recurring models/triples, or spread evenly?
  4. Half-block check: does the excess vanish when blocks are split in two
     (=> within-block gap drift is the mechanism)?

Helpers duplicated from scripts/17 (kept standalone so 17's recorded run
stays untouched).

Outputs: results/tables/rq2b_excess_{covariates,models,halfblock}.csv
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lattice_link import LatticeLink, LogisticLink  # noqa: E402

BLOCK_DAYS = 60
DAY = 86400.0


class LinkInverter:
    def __init__(self, link):
        if isinstance(link, LogisticLink):
            self._logit = True
        else:
            self._logit = False
            gs = link._gaps
            F = link.f_decisive(gs)
            keep = np.concatenate([[True], np.diff(F) > 1e-15])
            self._F, self._g = F[keep], gs[keep]
            self._dF = np.gradient(self._F, self._g)

    def gap_and_var(self, p, n):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        if self._logit:
            g = np.log(p / (1 - p))
            dF = p * (1 - p)
        else:
            g = np.interp(p, self._F, self._g)
            dF = np.interp(p, self._F, self._dF)
        return g, p * (1 - p) / (n * dF ** 2)


INV = {"logit": LinkInverter(LogisticLink()),
       "lat_u0.5855": LinkInverter(LatticeLink(unit=0.5855))}


def legs_from_battles(df, min_dec):
    dec = df[df["winner"].isin(("model_a", "model_b"))]
    a_win = dec["winner"] == "model_a"
    first = np.where(dec["model_a"] < dec["model_b"], dec["model_a"], dec["model_b"])
    second = np.where(dec["model_a"] < dec["model_b"], dec["model_b"], dec["model_a"])
    first_wins = np.where(dec["model_a"] < dec["model_b"], a_win, ~a_win)
    g = pd.DataFrame({"first": first, "second": second, "fw": first_wins}).groupby(
        ["first", "second"])["fw"].agg(["sum", "size"])
    g = g[g["size"] >= min_dec]
    return {(a, b): (row["sum"] / row["size"], int(row["size"]))
            for (a, b), row in g.iterrows()}


def triples_z(leg_stats):
    models = sorted({m for pr in leg_stats for m in pr})
    adj = {m: set() for m in models}
    for (a, b) in leg_stats:
        adj[a].add(b)
        adj[b].add(a)
    rows = []
    for a, b, c in combinations(models, 3):
        if b in adj[a] and c in adj[a] and c in adj[b]:
            rec = {"a": a, "b": b, "c": c}
            for name, inv in INV.items():
                g_ab, v_ab = inv.gap_and_var(*leg_stats[(a, b)])
                g_bc, v_bc = inv.gap_and_var(*leg_stats[(b, c)])
                g_ac, v_ac = inv.gap_and_var(*leg_stats[(a, c)])
                rec[f"z_{name}"] = float((g_ac - g_ab - g_bc) / np.sqrt(v_ab + v_bc + v_ac))
            rows.append(rec)
    return pd.DataFrame(rows)


battles = pd.read_parquet(ROOT / "data" / "processed" / "clean_battle_20240814.parquet")
RENAMES = {"chatgpt-4o-latest": "chatgpt-4o-latest-2024-08-08"}
battles = battles.assign(model_a=battles["model_a"].replace(RENAMES),
                         model_b=battles["model_b"].replace(RENAMES))
dedup = battles[battles["dedup_sampled"]]
first_seen = pd.concat([dedup.groupby("model_a")["tstamp"].min(),
                        dedup.groupby("model_b")["tstamp"].min()], axis=1).min(axis=1)
last_seen = pd.concat([dedup.groupby("model_a")["tstamp"].max(),
                       dedup.groupby("model_b")["tstamp"].max()], axis=1).max(axis=1)

t0 = float(dedup["tstamp"].min())
t_end = float(dedup["tstamp"].max())

cov_rows, model_rows, half_rows = [], [], []
k = 0
while t0 + (k + 1) * BLOCK_DAYS * DAY <= t_end:
    b_start = t0 + k * BLOCK_DAYS * DAY
    b_end = t0 + (k + 1) * BLOCK_DAYS * DAY
    label = f"block{k:02d}"
    dates = f"{pd.Timestamp(b_start, unit='s').date()}..{pd.Timestamp(b_end, unit='s').date()}"
    k += 1
    stable = set(first_seen[first_seen <= b_start].index) & set(
        last_seen[last_seen >= b_end].index)
    blk_all = dedup[(dedup["tstamp"] > b_start) & (dedup["tstamp"] <= b_end)]
    blk = blk_all[blk_all["model_a"].isin(stable) & blk_all["model_b"].isin(stable)]
    legs = legs_from_battles(blk, 100)
    tz = triples_z(legs)
    if len(tz) == 0:
        continue

    entries_in_block = first_seen[(first_seen > b_start) & (first_seen <= b_end)]
    new_models = set(entries_in_block.index) | set(
        first_seen[(first_seen > b_start - 14 * DAY) & (first_seen <= b_start)].index)
    entrant_votes = (blk_all["model_a"].isin(new_models) | blk_all["model_b"].isin(new_models))
    mzsq_logit = float((tz["z_logit"] ** 2).mean())
    cov_rows.append({
        "block": label, "dates": dates, "n_triples": len(tz),
        "mean_zsq_logit": mzsq_logit,
        "mean_zsq_lat": float((tz["z_lat_u0.5855"] ** 2).mean()),
        "share_zsq_gt4_logit": float((tz["z_logit"] ** 2 > 4).mean()),
        "block_votes": len(blk_all),
        "n_entries_in_block": len(entries_in_block),
        "entrant_vote_share": float(entrant_votes.mean()),
        "n_judges": int(blk_all["judge"].nunique()),
        "votes_per_judge": float(len(blk_all) / blk_all["judge"].nunique()),
        "english_share": float((blk_all["language"] == "English").mean()),
        "tie_share": float(blk_all["winner"].str.startswith("tie").mean()),
        "high_freq_share": float(blk_all["dedup_high_freq"].mean()),
        "overlaps_dedup_switch": (b_start <= pd.Timestamp("2024-06-02").timestamp()
                                  and b_end >= pd.Timestamp("2024-05-01").timestamp()),
    })

    # concentration: model representation among excess triples (z^2>4, logit)
    exc = tz[tz["z_logit"] ** 2 > 4]
    all_counts = pd.concat([tz["a"], tz["b"], tz["c"]]).value_counts()
    exc_counts = pd.concat([exc["a"], exc["b"], exc["c"]]).value_counts()
    for m in all_counts.index:
        model_rows.append({"block": label, "model": m,
                           "triples_total": int(all_counts[m]),
                           "triples_excess": int(exc_counts.get(m, 0)),
                           "share_of_model_triples_excess":
                               float(exc_counts.get(m, 0) / all_counts[m])})

    # half-block drift check (legs >= 50 decisive per half)
    mid = (b_start + b_end) / 2
    for half, (h0, h1) in (("H1", (b_start, mid)), ("H2", (mid, b_end))):
        hblk = blk[(blk["tstamp"] > h0) & (blk["tstamp"] <= h1)]
        hlegs = legs_from_battles(hblk, 50)
        htz = triples_z(hlegs)
        if len(htz):
            half_rows.append({"block": label, "half": half, "n_triples": len(htz),
                              "mean_zsq_logit": float((htz["z_logit"] ** 2).mean())})
    print(f"[{label} {dates}] triples={len(tz)} mean z^2 logit={mzsq_logit:.3f}")

cov = pd.DataFrame(cov_rows)
cov.to_csv(ROOT / "results" / "tables" / "rq2b_excess_covariates.csv", index=False)
print("\n=== block covariates ===")
print(cov.round(3).to_string(index=False))

models = pd.DataFrame(model_rows)
models.to_csv(ROOT / "results" / "tables" / "rq2b_excess_models.csv", index=False)
print("\n=== model concentration in excess triples (blocks with mean z^2 > 1.5) ===")
for blk_label in cov[cov["mean_zsq_logit"] > 1.5]["block"]:
    sub = models[(models["block"] == blk_label) & (models["triples_excess"] > 0)]
    print(f"\n{blk_label}:")
    print(sub.sort_values("triples_excess", ascending=False).head(8).to_string(index=False))

halves = pd.DataFrame(half_rows)
halves.to_csv(ROOT / "results" / "tables" / "rq2b_excess_halfblock.csv", index=False)
print("\n=== half-block check (within-block drift diagnostic) ===")
print(halves.round(3).to_string(index=False))
print("\nDONE")

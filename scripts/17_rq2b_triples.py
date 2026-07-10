"""RQ2b — triple-additivity link-shape test, run as pre-registered in
logs/RQ2_DESIGN.md section 5.

Any gap-link model F implies additivity of inverted head-to-head rates:
F^-1(p_AB) + F^-1(p_BC) = F^-1(p_AC). For each dense stable triple we
compute the standardized additivity residual under each candidate link:

    z = [F^-1(p_AC) - F^-1(p_AB) - F^-1(p_BC)] / sqrt(v_AB + v_BC + v_AC)
    v_XY = p(1-p) / (n_XY * F'(g)^2)     (delta method)

If the link's shape is right (and gaps are stable within the window),
z ~ N(0,1) and mean z^2 ~ 1. Excess mean z^2 measures shape misfit;
z is scale-invariant, so the links compete fairly despite different
natural scales.

Pre-registered selection: 60-day non-overlapping blocks; models present
throughout the block (first battle <= block start, last battle >= block
end, computed on the full dedup file); every leg >= 100 decisive votes
within the block. Links: logit, lattice u0.5855 (primary), u0.1, u0.8.

A SYNTHETIC calibration check runs first (labeled): logit-generated and
lattice-generated worlds with leg sizes like the real blocks must give
mean z^2 ~ 1 for the true link. Gates must pass before the real
computation is reported.

Caveat printed with results: triples sharing legs are dependent;
inference is per-block descriptive + block-level sign consistency, no
vote-level p-values.

Outputs: results/tables/rq2b_blocks.csv, rq2b_summary.csv
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lattice_link import LatticeLink, LogisticLink  # noqa: E402

SEED = 20260713
BLOCK_DAYS = 60
MIN_DECISIVE_PER_LEG = 100
rng = np.random.default_rng(SEED)


# ---------------- link inversion helpers ----------------

class LinkInverter:
    def __init__(self, link, name):
        self.name = name
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
        var = p * (1 - p) / (n * dF ** 2)
        return g, var


LINKS = {
    "logit": LinkInverter(LogisticLink(), "logit"),
    "lat_u0.5855": LinkInverter(LatticeLink(unit=0.5855), "lat_u0.5855"),
    "lat_u0.1": LinkInverter(LatticeLink(unit=0.1), "lat_u0.1"),
    "lat_u0.8": LinkInverter(LatticeLink(unit=0.8), "lat_u0.8"),
}


def triple_zsq(leg_stats, links=LINKS):
    """leg_stats: dict pair(tuple sorted) -> (p_first_beats_second, n_dec).
    Returns dict link -> list of z^2 per triple, and triple count."""
    models = sorted({m for pr in leg_stats for m in pr})
    adj = {m: set() for m in models}
    for (a, b) in leg_stats:
        adj[a].add(b)
        adj[b].add(a)
    out = {k: [] for k in links}
    n_triples = 0
    for a, b, c in combinations(models, 3):
        if b in adj[a] and c in adj[a] and c in adj[b]:
            n_triples += 1
            for name, inv in links.items():
                p_ab, n_ab = leg_stats[(a, b)]
                p_bc, n_bc = leg_stats[(b, c)]
                p_ac, n_ac = leg_stats[(a, c)]
                g_ab, v_ab = inv.gap_and_var(p_ab, n_ab)
                g_bc, v_bc = inv.gap_and_var(p_bc, n_bc)
                g_ac, v_ac = inv.gap_and_var(p_ac, n_ac)
                z = (g_ac - g_ab - g_bc) / np.sqrt(v_ab + v_bc + v_ac)
                out[name].append(float(z ** 2))
    return out, n_triples


def legs_from_battles(df):
    dec = df[df["winner"].isin(("model_a", "model_b"))]
    a_win = dec["winner"] == "model_a"
    first = np.where(dec["model_a"] < dec["model_b"], dec["model_a"], dec["model_b"])
    second = np.where(dec["model_a"] < dec["model_b"], dec["model_b"], dec["model_a"])
    first_wins = np.where(dec["model_a"] < dec["model_b"], a_win, ~a_win)
    g = pd.DataFrame({"first": first, "second": second, "fw": first_wins}).groupby(
        ["first", "second"])["fw"].agg(["sum", "size"])
    g = g[g["size"] >= MIN_DECISIVE_PER_LEG]
    return {(a, b): (row["sum"] / row["size"], int(row["size"]))
            for (a, b), row in g.iterrows()}


# ---------------- SYNTHETIC calibration check (labeled; gates) ----------------
print("=== SYNTHETIC calibration check (not a research result) ===")
gen_links = {"logit": LogisticLink(), "lat_u0.8": LatticeLink(unit=0.8)}
gates_ok = True
for gen_name, gen in gen_links.items():
    n_models = 25
    theta = rng.normal(0, 0.35, n_models)  # real-scale gaps (median |gap| ~0.3)
    names = [f"s{k:02d}" for k in range(n_models)]
    rows = []
    for i, j in combinations(range(n_models), 2):
        n_leg = int(rng.choice([120, 250, 600, 1500]))
        p = float(gen.f_decisive(theta[i] - theta[j]))
        wins = rng.binomial(n_leg, p)
        rows.append((names[i], names[j], wins, n_leg))
    leg_stats = {(a, b): (w / n, n) for a, b, w, n in rows}
    zsq, n_tri = triple_zsq(leg_stats)
    msg = ", ".join(f"{k}: {np.mean(v):.3f}" for k, v in zsq.items())
    print(f"[gen={gen_name}] {n_tri} triples; mean z^2 -> {msg}")
    true_key = "logit" if gen_name == "logit" else "lat_u0.8"
    m = np.mean(zsq[true_key])
    if not (0.8 <= m <= 1.3):
        gates_ok = False
        print(f"  GATE FAIL: true link mean z^2 = {m:.3f} outside [0.8, 1.3]")
assert gates_ok, "synthetic calibration gates failed — do not trust real run"
print("gates PASS\n")

# ---------------- real data ----------------
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
DAY = 86400.0
block_rows, summary_acc = [], {k: [] for k in LINKS}
k = 0
while t0 + (k + 1) * BLOCK_DAYS * DAY <= t_end:
    b_start = t0 + k * BLOCK_DAYS * DAY
    b_end = t0 + (k + 1) * BLOCK_DAYS * DAY
    label = f"block{k:02d}_{pd.Timestamp(b_start, unit='s').date()}"
    k += 1
    stable = set(first_seen[(first_seen <= b_start)].index) & set(
        last_seen[(last_seen >= b_end)].index)
    blk = dedup[(dedup["tstamp"] > b_start) & (dedup["tstamp"] <= b_end)
                & dedup["model_a"].isin(stable) & dedup["model_b"].isin(stable)]
    leg_stats = legs_from_battles(blk)
    zsq, n_tri = triple_zsq(leg_stats)
    if n_tri == 0:
        print(f"[{label}] no qualifying triples")
        continue
    rec = {"block": label, "n_stable_models": len(stable),
           "n_legs": len(leg_stats), "n_triples": n_tri}
    for name in LINKS:
        v = np.asarray(zsq[name])
        rec[f"mean_zsq_{name}"] = float(v.mean())
        rec[f"share_absz_gt2_{name}"] = float((v > 4).mean())
        summary_acc[name].append((v, n_tri))
    d = np.asarray(zsq["logit"]) - np.asarray(zsq["lat_u0.5855"])
    rec["mean_zsq_logit_minus_lat"] = float(d.mean())
    rec["share_triples_lat_better"] = float((d > 0).mean())
    block_rows.append(rec)
    print(f"[{label}] models={len(stable)} legs={len(leg_stats)} triples={n_tri} "
          f"mean z^2: logit={rec['mean_zsq_logit']:.3f} "
          f"lat0.5855={rec['mean_zsq_lat_u0.5855']:.3f} "
          f"lat0.1={rec['mean_zsq_lat_u0.1']:.3f} lat0.8={rec['mean_zsq_lat_u0.8']:.3f}")

blocks = pd.DataFrame(block_rows)
blocks.to_csv(ROOT / "results" / "tables" / "rq2b_blocks.csv", index=False)

print("\n=== pooled (triple-weighted; triples within a block share legs — "
      "descriptive, block-level consistency is the inference unit) ===")
summary = []
for name in LINKS:
    allv = np.concatenate([v for v, _ in summary_acc[name]])
    summary.append({"link": name, "n_triples_total": len(allv),
                    "mean_zsq": float(allv.mean()),
                    "median_zsq": float(np.median(allv)),
                    "share_absz_gt2": float((allv > 4).mean())})
    print(summary[-1])
n_blocks = len(blocks)
lat_better_blocks = int((blocks["mean_zsq_logit_minus_lat"] > 0).sum())
print(f"\nblock-level: lattice(u0.5855) lower mean z^2 than logit in "
      f"{lat_better_blocks}/{n_blocks} blocks")
pd.DataFrame(summary).to_csv(ROOT / "results" / "tables" / "rq2b_summary.csv", index=False)
print("DONE")

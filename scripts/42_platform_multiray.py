"""Platform-wide multiray rating: MultiRay embedding of ALL clone accounts.

Extends script 41 (bot-arena multiray) to the whole Manifold-clone population,
exactly as script 34 extended script 33: input is script 34's returns_all.csv
(every account's realized return per resolved YES/NO market, v0 API cache),
races are markets with >= 2 traders, and the multiray machinery is imported
from script 41.

HISTORY / DIAGNOSIS (2026-07-17 first run, revised 2026-07-20):

The first full run (kept archived as
results/platform_multiray_degenerate_20260717/ — documented NEGATIVE result)
saturated: all 494 conditions ended with |centered ability| 1e7-6.8e10
against a +-7.5 lattice, consensus skills 1e9-1e10, seed tau ~0.30. The
original hypothesis (big fields -> identical Laplace-floor prices -> under-
determined geometry) was HALF right; probing showed a field-size cap does
NOT fix it (caps 8-20 still saturate; it is a per-seed lottery). The root
cause is numerical: the package's fit_inner targets y = -err/slope with
slopes clamped at slope_floor = 1e-10, so any entrant in the FLAT TAIL of a
condition's price curve (exactly where floor-priced entrants in 35-48-item
fields sit) gets a ~1e10 Gauss-Newton step target, which then pollutes the
shared least-squares updates and cascades. Fixes, all outside the vendored
model:

1. TRUST REGION (m41.SLOPE_FLOOR = 0.05): slope_floor is an exposed
   constructor parameter; raising it bounds |step target| <= |err|/0.05
   while leaving responsive-region slopes (~0.1-0.3) untouched. Kills the
   saturation everywhere (uncapped MSE 1.9e-2 -> ~9e-4, abilities O(1)).
2. SLOPE-WEIGHTED IDENTIFIED SUMMARY (m41.identified_skill_slopewt): even
   post-fix, seeds sit on an MSE plateau (~0.8-1.1e-3) with DIFFERENT
   rankings (pairwise tau ~0.40; longer inner loops and dim=1 do not help):
   a flat-tail price pins its ability only to a half-line, so ~16% of
   (condition, entrant) cells carry optimizer-arbitrary point values, and
   the plain summary averages them in. Weighting each cell by
   n_races * |local price-curve slope| removes exactly the unpinned cells:
   pairwise seed tau 0.40 -> 0.58, disjoint 5-seed half-ensembles 0.78.
3. SEED-ENSEMBLE CONSENSUS: the board = mean slope-weighted summary over
   ENSEMBLE_SEEDS (10 fits). Certified by probe (2026-07-20): two DISJOINT
   10-seed ensembles agree at tau 0.821 (rated 0.811), vs 0.15-0.48 for
   single seeds; per-seed summaries are persisted (seed_summaries.csv).

CALIBRATION for the vs-script-34 comparison: the naive smoothed-win-rate
baseline reaches only tau ~0.41 vs the pairwise board — race-price
aggregation and 2.5M pairwise duels measure genuinely different things on
this platform, so tau ~0.4 is the data's ceiling for ANY race-price model,
not a defect of this one.

Design (unchanged from the original):
- Condition = (underlying cluster, fixed entrant set) with >= MIN_RACES_COND
  races (primary 3: 494 conditions / 71 traders / 1,890 races, ONE connected
  block; argv override 1 places all 163 traders across 3,897 conditions).
- Prices = Laplace-smoothed win frequencies (ALPHA = 0.5), script 33 tie
  rules, identical to script 41.
- Uncertainty (min>=3 run): cluster bootstrap (resample underlying clusters,
  rebuild conditions, refit a BOOT_SEEDS ensemble, re-rank), B = B_BOOT
  replicates in a process pool, draws pre-generated from a fixed seed.
  RANK CIs are the reportable uncertainty (skill quantiles recorded).
- Sensitivities (min=3 run only, 3-seed ensembles): dim=1, alpha 0.25 / 1.0,
  min_races=5. The min=1 all-traders run fits the primary ensemble only and
  reports half-ensemble tau + tau vs the min=3 board instead of a bootstrap.
- Agreement vs script 34's pairwise board (theta_lattice) over placed
  traders, overall and restricted to script 34's rated set (>= 20 races),
  plus the naive-baseline calibration rows.

Usage:
    .venv/bin/python scripts/42_platform_multiray.py        # min_races=3
    .venv/bin/python scripts/42_platform_multiray.py 1      # all traders

Outputs under results/platform_multiray[/_min1]/:
  conditions.csv, prices.csv, embedding.csv, rays.csv, abilities.csv,
  consensus.csv, agreement.csv, seed_summaries.csv, bootstrap_ranks.csv
"""

import importlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

# in pool WORKERS keep each process on one BLAS thread (parallelism comes
# from the pool); the parent process keeps its default threading. Spawned
# workers import this module as __mp_main__ before numpy.
if __name__ == "__mp_main__":
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# module 41 reads argv at import time; feed it a neutral argv, then restore
_argv = sys.argv[:]
sys.argv = ["41_bot_arena_multiray.py", "3"]
m41 = importlib.import_module("41_bot_arena_multiray")
sys.argv = _argv

# fix 1: trust region on the package's Gauss-Newton step (see docstring).
# Module-level so spawned pool workers (which re-import this module) get it.
m41.SLOPE_FLOOR = 0.05

RETURNS_ALL = ROOT / "results" / "platform_leaderboard" / "returns_all.csv"
BOARD_34 = ROOT / "results" / "platform_leaderboard" / "board.csv"

MIN_RACES_COND = int(sys.argv[1]) if len(sys.argv) > 1 else 3
# Optional field-size cap (argv[2]), kept from the diagnosis phase for
# sensitivity use. NOT needed for identification any more (the slope-floor
# trust region fixes the saturation the cap was aimed at).
MAX_FIELD = int(sys.argv[2]) if len(sys.argv) > 2 else None
_suffix = ("" if MIN_RACES_COND == 3 else f"_min{MIN_RACES_COND}") + \
          (f"_k{MAX_FIELD}" if MAX_FIELD else "")
OUT = ROOT / "results" / ("platform_multiray" + _suffix)
OUT.mkdir(parents=True, exist_ok=True)

ALPHA = 0.5
ALPHA_SENS = (0.25, 1.0)
DIM = 2
ENSEMBLE_SEEDS = (999, 7, 43, 101, 202, 303, 404, 505, 606, 707)
SENS_SEEDS = (999, 7, 43)   # 3-seed ensembles for sensitivities
BOOT_SEEDS = (999, 7, 43)   # 3-seed ensemble per bootstrap replicate
B_BOOT = 200
BOOT_SEED = 20260717
N_WORKERS = 14
RATED_34 = 20  # script 34's headline threshold, for the restricted tau

# ------------------------------------------------------------- pool workers

_W = {}  # worker globals, set once per process by the initializers


def capped_block_conditions(races, min_races, alpha, block, max_field):
    """block_conditions + whole-condition field-size cap (order-invariant:
    both filters drop entire conditions)."""
    c = m41.block_conditions(races, min_races, alpha, block)
    if max_field and not c.empty:
        c = c[c["field_size"] <= max_field]
    return c


def _fit_init(cond_frames):
    _W["cond_frames"] = cond_frames


def _fit_one(args):
    """Fit one (variant, seed) job; returns the slope-weighted summary plus
    fit diagnostics (and the raw fit params for the primary variant)."""
    tag, seed, dim = args
    bc = _W["cond_frames"][tag]
    t0 = time.time()
    fit = m41.fit_block(bc, dim, seed)
    mse, mx = m41.fit_mse(fit)
    sw = m41.identified_skill_slopewt(fit, bc)
    plain = m41.identified_skill(fit, bc)
    max_ctr = 0.0
    for spec in fit.conditions:
        a = np.array([fit.ability(spec.cond_id, b) for b in spec.item_ids])
        max_ctr = max(max_ctr, float(np.max(np.abs(a - a.mean()))))
    extras = None
    if tag == "primary":
        rayp = m41.rayproj_skill(fit)
        Z = {b: fit.Z[b].copy() for b in fit.item_ids}
        rays = [{"cond_id": spec.cond_id,
                 "v": fit.V[spec.cond_id].copy(),
                 "beta": fit.beta[spec.cond_id],
                 "item_ids": list(spec.item_ids),
                 "prices": np.asarray(spec.prices).tolist(),
                 "p_hat": fit.predict_condition(spec.cond_id).tolist(),
                 "abilities": [fit.ability(spec.cond_id, b)
                               for b in spec.item_ids]}
                for spec in fit.conditions]
        extras = {"rayproj": rayp.to_dict(), "Z": Z, "rays": rays}
    return (tag, seed, mse, mx, max_ctr, time.time() - t0,
            sw.to_dict(), plain.to_dict(), extras)


def _boot_init(races, block, min_races, alpha, level, headline, cons,
               max_field):
    _W.update(races=races, block=block, min_races=min_races, alpha=alpha,
              level=level, headline=headline, cons=cons, max_field=max_field)


def _boot_one(args):
    rep_id, draw = args
    races, block = _W["races"], _W["block"]
    parts = []
    for di, g in enumerate(draw):
        part = races[races[_W["level"]] == g].copy()
        part["race_key"] = part["race_key"].astype(str) + f"#{di}"
        parts.append(part)
    bb = capped_block_conditions(pd.concat(parts, ignore_index=True),
                                 _W["min_races"], _W["alpha"], block,
                                 _W["max_field"])
    if bb.empty or bb["cond_id"].nunique() < 1 or bb["bot"].nunique() < 2:
        return rep_id, None
    sws = []
    for s in BOOT_SEEDS:
        try:
            fb = m41.fit_block(bb, DIM, s)
        except Exception:
            continue
        sws.append(m41.identified_skill_slopewt(fb, bb))
    if not sws:
        return rep_id, None
    cb = pd.DataFrame({i: s for i, s in enumerate(sws)}).mean(axis=1)
    present = [b for b in _W["headline"] if b in cb.index]
    if len(present) < 2:
        return rep_id, None
    cb = cb + (_W["cons"][present].median() - cb[present].median())
    return rep_id, cb.reindex(_W["headline"]).to_dict()


# ------------------------------------------------------------------- main

def load_races() -> pd.DataFrame:
    r = pd.read_csv(RETURNS_ALL)
    for col in ("question", "cluster", "series"):
        r[col] = r[col].fillna("")
    nb = r.groupby("market_id")["bot"].nunique()
    races = r[r["market_id"].isin(nb[nb >= 2].index)].copy()
    races["race_key"] = races["market_id"]
    return races


def ens_mean(d: dict, seeds) -> pd.Series:
    return pd.DataFrame({s: d[s] for s in seeds}).mean(axis=1)


def main():
    t_start = time.time()
    races = load_races()
    print(f"platform races: {races['race_key'].nunique()} | "
          f"traders: {races['bot'].nunique()} | "
          f"clusters: {races['cluster'].nunique()}", flush=True)

    cond = m41.build_conditions(races, MIN_RACES_COND, ALPHA)
    if MAX_FIELD:
        cond = cond[cond["field_size"] <= MAX_FIELD]
    cond.to_csv(OUT / "conditions.csv", index=False)
    placed = sorted(cond["bot"].unique())
    unplaced = sorted(set(races["bot"]) - set(placed))
    print(f"conditions (>= {MIN_RACES_COND} races, fixed field"
          + (f", <= {MAX_FIELD} entrants" if MAX_FIELD else "") + "): "
          f"{cond['cond_id'].nunique()}, traders placed: {len(placed)}, "
          f"unplaced: {len(unplaced)}", flush=True)

    blocks = m41.bipartite_blocks(cond)
    print(f"bipartite block sizes: {[len(b) for b in blocks]}", flush=True)
    block = blocks[0]
    if len(blocks) > 1:
        outside = sorted(set(placed) - block)
        print(f"NOTE: fitting the giant block only; outside it: {outside}",
              flush=True)
    bc = cond[cond["bot"].isin(block)]
    bc = bc[bc.groupby("cond_id")["bot"].transform("size") >= 2]

    print(f"--- giant block: {len(block)} traders, "
          f"{bc['cond_id'].nunique()} conditions ---", flush=True)

    # ---- variant condition frames + fit jobs (one pool, all ensembles) ----
    cond_frames = {"primary": bc}
    jobs = [("primary", s, DIM) for s in ENSEMBLE_SEEDS]
    if MIN_RACES_COND >= 3:
        jobs += [("dim1", s, 1) for s in SENS_SEEDS]
        cond_frames["dim1"] = bc
        for a_s in ALPHA_SENS:
            ca = capped_block_conditions(races, MIN_RACES_COND, a_s, block,
                                         MAX_FIELD)
            cond_frames[f"alpha{a_s}"] = ca
            jobs += [(f"alpha{a_s}", s, DIM) for s in SENS_SEEDS]
        cm_all = m41.build_conditions(races, 5, ALPHA)
        if MAX_FIELD:
            cm_all = cm_all[cm_all["field_size"] <= MAX_FIELD]
        b5 = max(m41.bipartite_blocks(cm_all),
                 key=lambda s: len(s & block), default=set())
        if len(b5 & block) >= 2:
            cond_frames["min5"] = capped_block_conditions(
                races, 5, ALPHA, b5, MAX_FIELD)
            jobs += [("min5", s, DIM) for s in SENS_SEEDS]

    t0 = time.time()
    sw_all, plain_all, extras_by_seed = {}, {}, {}
    fit_meta = []
    print(f"fitting {len(jobs)} (variant, seed) jobs on {N_WORKERS} workers",
          flush=True)
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_fit_init,
                             initargs=(cond_frames,)) as ex:
        for (tag, seed, mse, mx, max_ctr, dt, sw, plain,
             extras) in ex.map(_fit_one, jobs, chunksize=1):
            sw_all.setdefault(tag, {})[seed] = pd.Series(sw)
            plain_all.setdefault(tag, {})[seed] = pd.Series(plain)
            fit_meta.append({"variant": tag, "seed": seed, "price_mse": mse,
                             "max_abs_err": mx, "max_ctr_ability": max_ctr})
            if extras is not None:
                extras_by_seed[seed] = extras
            if max_ctr > 100:
                print(f"  WARNING: {tag} seed {seed} saturated "
                      f"(max|ctr ability| {max_ctr:.2e})", flush=True)
    meta = pd.DataFrame(fit_meta)
    print(f"fits done [{(time.time()-t0)/60:.1f} min]; primary per-seed MSE "
          + ", ".join(f"{r.seed}: {r.price_mse:.1e}" for r in
                      meta[meta.variant == "primary"].itertuples())
          + f"; max|ctr ability| over all fits "
          f"{meta.max_ctr_ability.max():.2f}", flush=True)

    prim = sw_all["primary"]
    cons = ens_mean(prim, ENSEMBLE_SEEDS)
    cons_plain = ens_mean(plain_all["primary"], ENSEMBLE_SEEDS)
    cons_std = pd.DataFrame(prim).std(axis=1)
    best_seed = int(meta[meta.variant == "primary"]
                    .sort_values("price_mse").iloc[0]["seed"])
    best_mse = float(meta[(meta.variant == "primary")
                          & (meta.seed == best_seed)]["price_mse"].iloc[0])
    rayp = pd.Series(extras_by_seed[best_seed]["rayproj"])

    pd.DataFrame(prim).to_csv(OUT / "seed_summaries.csv")

    # ---- agreement table ----
    n_races_bot = (races[races["bot"].isin(block)]
                   .groupby("bot")["race_key"].nunique())
    rated = n_races_bot[n_races_bot >= RATED_34].index

    def tau_pair(a, b, restrict=None):
        if restrict is not None:
            a = a[a.index.isin(restrict)]
            b = b[b.index.isin(restrict)]
        return m41.kendall(a, b)

    ag_rows = []
    taus = [tau_pair(prim[a], prim[b])
            for a, b in combinations(ENSEMBLE_SEEDS, 2)]
    taus_r = [tau_pair(prim[a], prim[b], rated)
              for a, b in combinations(ENSEMBLE_SEEDS, 2)]
    ag_rows.append({"comparison": "mean pairwise seed tau (slope-wt summary)",
                    "kendall_tau": float(np.nanmean(taus))})
    ag_rows.append({"comparison": f"mean pairwise seed tau (rated >= {RATED_34})",
                    "kendall_tau": float(np.nanmean(taus_r))})
    h1 = ens_mean(prim, ENSEMBLE_SEEDS[:5])
    h2 = ens_mean(prim, ENSEMBLE_SEEDS[5:])
    ag_rows.append({"comparison": "half-ensemble (5 vs 5 disjoint seeds)",
                    "kendall_tau": tau_pair(h1, h2)})
    ag_rows.append({"comparison": f"half-ensemble (rated >= {RATED_34})",
                    "kendall_tau": tau_pair(h1, h2, rated)})
    ag_rows.append({"comparison": "consensus vs plain-summary ensemble "
                                  "(diagnostic)",
                    "kendall_tau": tau_pair(cons, cons_plain)})
    ag_rows.append({"comparison": "consensus vs best-seed rayproj (diagnostic)",
                    "kendall_tau": tau_pair(cons, rayp)})
    for tag, label in (("dim1", "dim=2 vs dim=1 (3-seed ens)"),
                       ("alpha0.25", f"alpha={ALPHA} vs 0.25 (3-seed ens)"),
                       ("alpha1.0", f"alpha={ALPHA} vs 1.0 (3-seed ens)"),
                       ("min5", f"min_races={MIN_RACES_COND} vs 5 (3-seed ens)")):
        if tag in sw_all:
            ag_rows.append({"comparison": label,
                            "kendall_tau": tau_pair(
                                cons, ens_mean(sw_all[tag], SENS_SEEDS))})

    naive = (bc.groupby("bot")
             .apply(lambda g: float(np.average(g["price"] * g["field_size"],
                                               weights=g["n_races"])),
                    include_groups=False))
    ag_rows.append({"comparison": "consensus vs naive win-rate baseline",
                    "kendall_tau": tau_pair(cons, naive)})

    if MIN_RACES_COND < 3:
        dense_dir = "platform_multiray" + (f"_k{MAX_FIELD}" if MAX_FIELD else "")
        dense_path = ROOT / "results" / dense_dir / "consensus.csv"
        if dense_path.exists():
            dense = pd.read_csv(dense_path).set_index("bot")["consensus_skill"]
            ag_rows.append({"comparison": "min=1 vs min=3 board (shared traders)",
                            "kendall_tau": tau_pair(cons, dense)})

    if BOARD_34.exists():
        b34 = pd.read_csv(BOARD_34).set_index("bot")
        th34 = b34["theta_lattice"].dropna()
        ag_rows.append({"comparison": "consensus vs script-34 theta (all placed)",
                        "kendall_tau": tau_pair(cons, th34)})
        ag_rows.append({"comparison": f"consensus vs script-34 theta "
                                      f"(rated >= {RATED_34} races)",
                        "kendall_tau": tau_pair(cons, th34, rated)})
        ag_rows.append({"comparison": "naive baseline vs script-34 theta "
                                      "(calibration: the data's ceiling)",
                        "kendall_tau": tau_pair(naive, th34)})

    # ---- bootstrap (min>=3 only; parallel over replicates) ----
    rank_df = skill_df = pd.DataFrame()
    level, lower_bound = "cluster", False
    if MIN_RACES_COND >= 3 and B_BOOT > 0:
        groups = sorted(races[races["bot"].isin(block)][level].unique())
        races_in_groups = races[races[level].isin(groups)]
        headline = sorted(cons.index)
        rng = np.random.default_rng(BOOT_SEED)
        draws = [(i, list(rng.choice(groups, size=len(groups), replace=True)))
                 for i in range(B_BOOT)]
        print(f"bootstrap: level={level} ({len(groups)} groups), B={B_BOOT}, "
              f"{len(BOOT_SEEDS)}-seed ensemble per replicate, "
              f"{N_WORKERS} workers", flush=True)
        t0 = time.time()
        results, n_fail, n_done = {}, 0, 0
        with ProcessPoolExecutor(
                max_workers=N_WORKERS, initializer=_boot_init,
                initargs=(races_in_groups, block, MIN_RACES_COND, ALPHA,
                          level, headline, cons, MAX_FIELD)) as ex:
            for rep_id, res in ex.map(_boot_one, draws, chunksize=1):
                n_done += 1
                if res is None:
                    n_fail += 1
                else:
                    results[rep_id] = res
                if n_done % 20 == 0:
                    print(f"  {n_done}/{B_BOOT} replicates "
                          f"[{(time.time()-t0)/60:.1f} min]", flush=True)
        skills = [pd.Series(results[i]) for i in sorted(results)]
        skill_df = pd.DataFrame(skills)
        rank_df = skill_df.rank(axis=1, ascending=False)
        rank_df.to_csv(OUT / "bootstrap_ranks.csv", index=False)
        print(f"bootstrap: {len(rank_df)} replicates, {n_fail} failed "
              f"[{(time.time()-t0)/60:.1f} min]", flush=True)
    else:
        print("bootstrap: SKIPPED (all-traders run; half-ensemble tau + tau "
              "vs the min=3 board are the stability evidence)", flush=True)

    # ---- outputs ----
    ex_best = extras_by_seed[best_seed]
    all_emb, all_rays, all_abil, all_cons = [], [], [], []
    for b in sorted(cons.index):
        z = ex_best["Z"][b]
        row = {"bot": b, "z1": z[0], "z2": z[1] if DIM > 1 else np.nan,
               "consensus_skill": cons[b],
               "consensus_seed_std": cons_std.get(b, np.nan),
               "plain_skill_diagnostic": cons_plain.get(b, np.nan),
               "rayproj_skill_diagnostic": rayp.get(b, np.nan),
               "best_seed": best_seed, "price_mse_best": best_mse,
               "n_conditions": int((bc["bot"] == b).sum()),
               "n_races": int(n_races_bot.get(b, 0))}
        if len(rank_df) and b in rank_df.columns and rank_df[b].notna().any():
            rk, sk = rank_df[b].dropna(), skill_df[b].dropna()
            row.update({"rank_median": float(rk.median()),
                        "rank_lo95": float(rk.quantile(0.025)),
                        "rank_hi95": float(rk.quantile(0.975)),
                        "skill_lo95": float(sk.quantile(0.025)),
                        "skill_hi95": float(sk.quantile(0.975)),
                        "boot_coverage": len(rk) / max(len(rank_df), 1),
                        "boot_level": level, "boot_lower_bound": lower_bound})
        all_cons.append(row)
        all_emb.append({"bot": b, "z1": z[0],
                        "z2": z[1] if DIM > 1 else np.nan})
    for spec in ex_best["rays"]:
        v = spec["v"]
        all_rays.append({"cond_id": spec["cond_id"], "v1": v[0],
                         "v2": v[1] if DIM > 1 else np.nan,
                         "beta": spec["beta"],
                         "n_races": int(bc[bc["cond_id"] == spec["cond_id"]]
                                        ["n_races"].iloc[0])})
        for k_, b in enumerate(spec["item_ids"]):
            all_abil.append({"cond_id": spec["cond_id"], "bot": b,
                             "ability_racetime": spec["abilities"][k_],
                             "skill": -spec["abilities"][k_],
                             "p_obs": spec["prices"][k_],
                             "p_hat": spec["p_hat"][k_]})

    pd.DataFrame(all_emb).to_csv(OUT / "embedding.csv", index=False)
    pd.DataFrame(all_rays).to_csv(OUT / "rays.csv", index=False)
    pd.DataFrame(all_abil).to_csv(OUT / "abilities.csv", index=False)
    cond[["cond_id", "cluster", "bot", "n_races", "field_size", "win_credit",
          "price"]].to_csv(OUT / "prices.csv", index=False)
    meta.to_csv(OUT / "fit_meta.csv", index=False)
    agreement = pd.DataFrame(ag_rows)
    agreement.to_csv(OUT / "agreement.csv", index=False)
    consensus = (pd.DataFrame(all_cons)
                 .sort_values("consensus_skill", ascending=False))
    consensus.to_csv(OUT / "consensus.csv", index=False)

    print("\n=== AGREEMENT ===")
    print(agreement.to_string(index=False))
    print("\n=== CONSENSUS BOARD (top 40; ensemble slope-weighted skill, "
          "higher better) ===")
    with pd.option_context("display.width", 250):
        print(consensus.head(40).to_string(index=False))
    print(f"\nunplaced traders (no fixed field with >= {MIN_RACES_COND} "
          f"races): {len(unplaced)}")
    print(f"total wall time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()

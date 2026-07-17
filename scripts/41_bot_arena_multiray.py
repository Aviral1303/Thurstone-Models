"""Bot arena, multidimensional: MultiRay embedding of quantbots (Cotton's repo).

Applies the vendored package's multidimensional rating system
(`thurstone.multiray.MultiRayGlobalCalibrator`) to the same clone-market race
data as script 33. Each bot gets a latent vector Z in R^dim; each CONDITION
gets a unit ray V and offset beta; the bot's effective 1D ability in that
condition is a = beta + V.Z, and the model prices the condition's race from
those abilities through the lattice machinery. Fitting alternates
least-squares updates of (beta, V) and Z against observed win probabilities.

Design (first multiray pass, 2026-07-16):

- Item = bot. Condition = (underlying cluster, fixed entrant set) observed in
  >= MIN_RACES_COND resolved multi-bot races (primary 3, sensitivity 2).
  Script 33's races mostly repeat a few fixed fields per underlying, which is
  exactly the package's semantics: a condition is a repeated race, and its
  observed prices are the entrants' win frequencies.
- Race winner = top realized return ratio (input: script 33's returns.csv).
  Tie rules mirror script 33: all-nonpositive race -> all entrants tie;
  otherwise leaders within EPS_TIE of the top log-return split the win
  credit. Prices = Laplace-smoothed win frequencies,
      p_i = (credit_i + ALPHA) / (n_races + ALPHA * k),
  ALPHA = 0.5 primary (0.25 / 1.0 sensitivity) — raw 0/1 frequencies are not
  invertible by the calibrator.
- Identifiable blocks: connected components of the bot-condition bipartite
  graph. Bots in different blocks share no condition, so their relative
  placement is pure gauge — each block is fitted and reported SEPARATELY
  (the pairwise analysis' league split resurfaces here structurally).
- Fit: dim=2 primary (dim=1 sensitivity), example-default base density
  (skew-normal, UniformLattice unit=0.1), seeds (999, 7, 43).
  CONVERGENCE (discovered on first run, 2026-07-16): the package's
  fit_with_rebuild is NOT monotone in the outer iterations on this data
  (seed 999 price MSE 2e-4 at outer=5 -> 1.8e-2 at outer=30), and beta rides
  a per-condition translation ridge (race prices depend only on relative
  abilities, so beta wanders freely; one seed reached |beta| ~ 1e13 without
  hurting the fit). Mitigation kept OUTSIDE the model: run up to MAX_OUTER
  outer iterations, snapshot parameters at every outer step, keep the
  best-MSE snapshot (early stop after PATIENCE non-improving steps); fit all
  seeds and take the best-MSE seed as primary. Model itself untouched.
- SIGN: the package is race-time (lower ability = better; verified from the
  lookup curve before this script was written). The flip to higher-is-better
  happens EXACTLY ONCE: skill = -ability at the reporting layer.
- Consensus ranking per block = the IDENTIFIED summary: race-count-weighted
  average of within-condition-CENTERED abilities, sign-flipped. Centering
  removes beta (pure gauge) and observed prices pin centered abilities, so
  this summary is optimizer-stable wherever the fit actually converges. The
  naive block-mean-ray projection -(V_bar . Z) is kept as a diagnostic
  column only: block 0 has more free parameters than price observations
  (dim=2: ~16 effective params vs 12 free prices), the embedding geometry is
  a solution MANIFOLD, and the ray projection came out seed-dependent
  (tau 0.0-0.33 across seeds on the first run).
- Uncertainty: bootstrap over underlying clusters within block if the block
  has >= 2 clusters, else over races (finer level -> CIs are lower bounds;
  labeled). B = B_BOOT refits; ranks on the consensus skill.
- The package's fit weighs every condition equally regardless of how many
  races back its prices (no likelihood weighting). Kept as-is — this run
  tries Peter's system as built; smoothing pulls thin conditions toward
  uniform, which partially de-weights them.

Outputs under results/bot_arena_multiray/:
  conditions.csv, prices.csv, embedding.csv, rays.csv, abilities.csv,
  consensus.csv, agreement.csv, bootstrap_ranks_block*.csv
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from thurstone import AbilityCalibrator, Density, UniformLattice  # noqa: E402
from thurstone.multiray import MultiRayGlobalCalibrator  # noqa: E402

RETURNS = ROOT / "results" / "bot_arena" / "returns.csv"
BOARD_33 = ROOT / "results" / "bot_arena" / "board.csv"

EPS_TIE = 0.01
# primary condition threshold; override via argv[1]. min_races=1 admits
# single-race winner-take-all conditions (heavily smoothed) — the only way
# every bot places; at 1 the whole 11-bot field is one connected block,
# bridged cotton<->metals by a SINGLE race (flagged in the output).
MIN_RACES_COND = int(sys.argv[1]) if len(sys.argv) > 1 else 3
MIN_SENS_LIST = (2,) if MIN_RACES_COND == 3 else tuple(
    m for m in (2, 3) if m != MIN_RACES_COND)

OUT = ROOT / "results" / ("bot_arena_multiray" if MIN_RACES_COND == 3
                          else f"bot_arena_multiray_min{MIN_RACES_COND}")
OUT.mkdir(parents=True, exist_ok=True)
ALPHA = 0.5
ALPHA_SENS = (0.25, 1.0)
DIM = 2
SEEDS = (999, 7, 43)
# L=150 checked against the example default L=500 before adoption:
# consensus skills identical to 1.7e-13, 4.4x faster (needed for bootstrap)
LATTICE_L = 150
LATTICE_UNIT = 0.1
MAX_OUTER, INNER, PATIENCE = 30, 10, 6
B_BOOT = 200
BOOT_SEED = 20260716


# ---------------------------------------------------------------- conditions

def race_win_credit(ratios: pd.Series, eps: float) -> dict:
    """Win credit per bot for one race; mirrors script 33's tie rules."""
    if (ratios <= 0).all():
        winners = list(ratios.index)
    else:
        top = ratios.max()
        winners = [b for b, r in ratios.items()
                   if r > 0 and abs(np.log(r / top)) <= eps]
    return {b: 1.0 / len(winners) for b in winners}


def build_conditions(races: pd.DataFrame, min_races: int, alpha: float) -> pd.DataFrame:
    """One row per (condition, bot): smoothed win-frequency price.

    Condition key = (cluster, sorted entrant tuple). Duplicate-safe: races may
    appear multiple times (bootstrap replicates duplicate whole groups).
    """
    rows = []
    entr = races.groupby("race_key")["bot"].apply(lambda s: tuple(sorted(s)))
    races = races.assign(entrants=races["race_key"].map(entr))
    for (cluster, entrants), g in races.groupby(["cluster", "entrants"]):
        n = g["race_key"].nunique()
        if n < min_races or len(entrants) < 2:
            continue
        credit = defaultdict(float)
        for _, race in g.groupby("race_key"):
            for b, c in race_win_credit(race.set_index("bot")["ratio"], EPS_TIE).items():
                credit[b] += c
        k = len(entrants)
        for b in entrants:
            rows.append({
                "cond_id": f"{cluster}|{'+'.join(entrants)}",
                "cluster": cluster, "bot": b, "n_races": n, "field_size": k,
                "win_credit": credit[b],
                "price": (credit[b] + alpha) / (n + alpha * k),
            })
    return pd.DataFrame(rows)


def block_conditions(races_sub: pd.DataFrame, min_races: int, alpha: float,
                     block: set) -> pd.DataFrame:
    """Conditions whose ENTIRE field lies inside the block.

    Never filter races by bot before building conditions: that truncates
    mixed fields (e.g. a 3-entrant race with one out-of-block bot) into fake
    fixed-field conditions with misattributed win credit. Mixed conditions
    are dropped whole instead.
    """
    c = build_conditions(races_sub, min_races, alpha)
    if c.empty:
        return c
    ok = c.groupby("cond_id")["bot"].transform(lambda s: s.isin(block).all())
    return c[ok]


def bipartite_blocks(cond: pd.DataFrame) -> list[set]:
    """Connected components of bots linked by shared conditions."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, g in cond.groupby("cond_id"):
        bots = list(g["bot"])
        for b in bots[1:]:
            ra, rb = find(bots[0]), find(b)
            if ra != rb:
                parent[ra] = rb
    comps = defaultdict(set)
    for x in parent:
        comps[find(x)].add(x)
    return sorted(comps.values(), key=len, reverse=True)


# ---------------------------------------------------------------- fitting

def fit_block(cond: pd.DataFrame, dim: int, seed: int) -> MultiRayGlobalCalibrator:
    """Fit with best-iterate selection (the package's loop is not monotone)."""
    bots = sorted(cond["bot"].unique())
    fit = MultiRayGlobalCalibrator(item_ids=bots, dim=dim, random_state=seed)
    for cid, g in cond.groupby("cond_id"):
        grid = UniformLattice(L=LATTICE_L, unit=LATTICE_UNIT)
        base = Density.skew_normal(grid, loc=0.0, scale=1.0, a=0.0)
        fit.add_condition(cond_id=cid, calibrator=AbilityCalibrator(base),
                          item_ids=list(g["bot"]), prices=g["price"].to_numpy())
    best_mse, best_params, since_best = np.inf, None, 0
    for _ in range(MAX_OUTER):
        fit.fit_with_rebuild(num_outer_iters=1, num_inner_iters=INNER)
        mse, _ = fit_mse(fit)
        if mse < best_mse - 1e-12:
            best_mse = mse
            best_params = ({b: z.copy() for b, z in fit.Z.items()},
                           {c: v.copy() for c, v in fit.V.items()},
                           dict(fit.beta))
            since_best = 0
        else:
            since_best += 1
            if since_best >= PATIENCE:
                break
    fit.Z, fit.V, fit.beta = best_params
    fit.rebuild_all_curves()
    return fit


def fit_block_best_seed(cond: pd.DataFrame, dim: int,
                        seeds=SEEDS) -> tuple[dict, int]:
    """Fit every seed, return all fits plus the best-MSE seed."""
    fits = {s: fit_block(cond, dim, s) for s in seeds}
    best = min(seeds, key=lambda s: fit_mse(fits[s])[0])
    return fits, best


def identified_skill(fit: MultiRayGlobalCalibrator, cond: pd.DataFrame) -> pd.Series:
    """Race-weighted mean of within-condition-centered abilities, flipped.

    Centering removes beta (translation gauge); prices pin centered
    abilities. THE sign flip to higher-is-better happens here and in the
    per-condition skill column only.
    """
    n = cond.drop_duplicates("cond_id").set_index("cond_id")["n_races"]
    num, den = defaultdict(float), defaultdict(float)
    for spec in fit.conditions:
        a = np.array([fit.ability(spec.cond_id, b) for b in spec.item_ids])
        a = a - a.mean()
        w = float(n[spec.cond_id])
        for k, b in enumerate(spec.item_ids):
            num[b] += w * a[k]
            den[b] += w
    return pd.Series({b: -num[b] / den[b] for b in num}).sort_index()


def rayproj_skill(fit: MultiRayGlobalCalibrator) -> pd.Series:
    """DIAGNOSTIC ONLY: -(mean ray . Z); seed-dependent when the geometry
    is underdetermined (see module docstring)."""
    vbar = np.mean([fit.V[c.cond_id] for c in fit.conditions], axis=0)
    nrm = np.linalg.norm(vbar)
    if nrm > 0:
        vbar = vbar / nrm
    return pd.Series({b: -float(np.dot(vbar, fit.Z[b])) for b in fit.item_ids})


def fit_mse(fit: MultiRayGlobalCalibrator) -> tuple[float, float]:
    errs = []
    for spec in fit.conditions:
        errs.append(fit.predict_condition(spec.cond_id) - spec.prices)
    e = np.concatenate(errs)
    return float(np.mean(e ** 2)), float(np.max(np.abs(e)))


def kendall(a: pd.Series, b: pd.Series) -> float:
    from scipy.stats import kendalltau
    common = a.index.intersection(b.index)
    if len(common) < 2:
        return np.nan
    return float(kendalltau(a[common], b[common]).statistic)


# ---------------------------------------------------------------- main

def main():
    races = pd.read_csv(RETURNS)
    races["race_key"] = races["market_id"]
    cond = build_conditions(races, MIN_RACES_COND, ALPHA)
    cond.to_csv(OUT / "conditions.csv", index=False)
    placed = sorted(cond["bot"].unique())
    unplaced = sorted(set(races["bot"]) - set(placed))
    n_cond = cond["cond_id"].nunique()
    print(f"conditions (>= {MIN_RACES_COND} races, fixed field): {n_cond}, "
          f"bots placed: {len(placed)}, unplaced: {unplaced}")

    blocks = bipartite_blocks(cond)
    print(f"bipartite blocks: {[sorted(b) for b in blocks]}")
    print("(cross-block geometry is unidentified — blocks fitted separately)\n")

    if MIN_RACES_COND < 3:
        # which thin conditions glue the dense (min=3) leagues together?
        dense = bipartite_blocks(build_conditions(races, 3, ALPHA))
        side = {b: f"league{i}" for i, s in enumerate(dense) for b in s}
        bridge_rows = []
        for cid, g in cond.groupby("cond_id"):
            sides = {side.get(b, f"unplaced:{b}") for b in g["bot"]}
            if len(sides) > 1:
                bridge_rows.append({"cond_id": cid,
                                    "n_races": int(g["n_races"].iloc[0]),
                                    "spans": " + ".join(sorted(sides))})
        bridges = pd.DataFrame(bridge_rows)
        bridges.to_csv(OUT / "bridge_conditions.csv", index=False)
        print("bridge conditions (join dense min=3 leagues / place thin bots;"
              " the joint gauge hangs on these):")
        print(bridges.to_string(index=False), "\n")

    board33 = pd.read_csv(BOARD_33).set_index("bot") if BOARD_33.exists() else None

    all_emb, all_rays, all_abil, all_cons, ag_rows = [], [], [], [], []
    rng = np.random.default_rng(BOOT_SEED)
    for bi, block in enumerate(blocks):
        bc = cond[cond["bot"].isin(block)]
        bc = bc[bc.groupby("cond_id")["bot"].transform("size") >= 2]
        print(f"--- block {bi}: {sorted(block)} "
              f"({bc['cond_id'].nunique()} conditions, "
              f"clusters {sorted(bc['cluster'].unique())}) ---")

        fits, best_seed = fit_block_best_seed(bc, DIM)
        fit = fits[best_seed]
        mse, mx = fit_mse(fit)
        seed_mses = {s: fit_mse(f)[0] for s, f in fits.items()}
        print(f"fit (dim={DIM}): best seed {best_seed}, price MSE {mse:.5f}, "
              f"max abs err {mx:.4f}; per-seed MSE "
              + ", ".join(f"{s}: {m:.1e}" for s, m in seed_mses.items()))

        cons = identified_skill(fit, bc)
        rayp = rayproj_skill(fit)
        # cross-seed agreement on the identified summary AND the ray projection
        for s in SEEDS:
            if s == best_seed:
                continue
            ag_rows.append({"block": bi, "comparison": f"seed {best_seed} vs {s}",
                            "kendall_tau": kendall(cons, identified_skill(fits[s], bc))})
            ag_rows.append({"block": bi,
                            "comparison": f"rayproj seed {best_seed} vs {s} (diagnostic)",
                            "kendall_tau": kendall(rayp, rayproj_skill(fits[s]))})
        # dim=1 sensitivity
        f1s, b1 = fit_block_best_seed(bc, 1)
        ag_rows.append({"block": bi, "comparison": "dim=2 vs dim=1",
                        "kendall_tau": kendall(cons, identified_skill(f1s[b1], bc))})
        # smoothing sensitivity
        for a_s in ALPHA_SENS:
            ca = block_conditions(races, MIN_RACES_COND, a_s, block)
            if ca.empty or ca["cond_id"].nunique() < 1:
                continue
            fas, ba = fit_block_best_seed(ca, DIM)
            ag_rows.append({"block": bi, "comparison": f"alpha={ALPHA} vs {a_s}",
                            "kendall_tau": kendall(cons, identified_skill(fas[ba], ca))})
        # min-races sensitivity: rebuild blocks at each other threshold, fit
        # the component that overlaps this block most, compare on shared bots
        for m_s in MIN_SENS_LIST:
            cm_all = build_conditions(races, m_s, ALPHA)
            blocks2 = bipartite_blocks(cm_all)
            b2 = max(blocks2, key=lambda s: len(s & block), default=set())
            if len(b2 & block) < 2:
                continue
            cm = block_conditions(races, m_s, ALPHA, b2)
            if cm.empty or cm["cond_id"].nunique() == bc["cond_id"].nunique():
                continue
            fms, bm = fit_block_best_seed(cm, DIM)
            ag_rows.append({"block": bi,
                            "comparison": f"min_races={MIN_RACES_COND} vs {m_s}"
                                          f" ({len(b2 & block)} shared bots)",
                            "kendall_tau": kendall(cons, identified_skill(fms[bm], cm))})
        # vs script 33 pairwise board — WITHIN script-33 leagues only
        # (theta gauges are per-league; cross-league theta is not comparable)
        if board33 is not None:
            for lg, gl in board33.groupby("league"):
                th = gl.loc[gl.index.isin(block), "theta_lattice"].dropna()
                if len(th) >= 3:
                    ag_rows.append({"block": bi,
                                    "comparison": f"consensus vs script-33 theta (league {lg})",
                                    "kendall_tau": kendall(cons, th)})

        # ---- bootstrap on consensus ranks ----
        level = "cluster" if bc["cluster"].nunique() >= 2 else "race_key"
        groups = sorted(races[races["bot"].isin(block)][level].unique())
        lower_bound = level != "cluster"
        print(f"bootstrap level: {level} ({len(groups)} groups)"
              + ("  [finer than underlying -> CIs are lower bounds]" if lower_bound else ""))
        block_races = races[races["bot"].isin(block)]
        # resample FULL race rows per group; block purity re-applied per replicate
        races_in_groups = races[races[level].isin(groups)]
        by_group = {g: races_in_groups[races_in_groups[level] == g] for g in groups}
        ranks, skills, n_fail = [], [], 0
        headline = sorted(cons.index)
        for it in range(B_BOOT):
            draw = rng.choice(groups, size=len(groups), replace=True)
            parts = []
            for di, g in enumerate(draw):
                part = by_group[g].copy()
                part["race_key"] = part["race_key"].astype(str) + f"#{di}"
                parts.append(part)
            bb = block_conditions(pd.concat(parts, ignore_index=True),
                                  MIN_RACES_COND, ALPHA, block)
            if bb.empty or bb["cond_id"].nunique() < 1 or bb["bot"].nunique() < 2:
                n_fail += 1
                continue
            try:
                fb = fit_block(bb, DIM, best_seed)
            except Exception:
                n_fail += 1
                continue
            cb = identified_skill(fb, bb)
            present = [b for b in headline if b in cb.index]
            if len(present) < 2:
                n_fail += 1
                continue
            cb = cb + (cons[present].median() - cb[present].median())
            ranks.append(cb[present].rank(ascending=False).reindex(headline))
            skills.append(cb.reindex(headline))
        rank_df = pd.DataFrame(ranks)
        skill_df = pd.DataFrame(skills)
        rank_df.to_csv(OUT / f"bootstrap_ranks_block{bi}.csv", index=False)
        print(f"bootstrap: {len(rank_df)} replicates, {n_fail} failed")

        # ---- collect outputs ----
        for b in fit.item_ids:
            z = fit.Z[b]
            row = {"block": bi, "bot": b, "z1": z[0], "z2": z[1] if DIM > 1 else np.nan,
                   "consensus_skill": cons[b],
                   "rayproj_skill_diagnostic": rayp[b],
                   "best_seed": best_seed, "price_mse": mse,
                   "n_conditions": int((bc["bot"] == b).sum()),
                   "n_races": int(block_races[block_races["bot"] == b]
                                  ["race_key"].nunique())}
            if b in rank_df.columns and rank_df[b].notna().any():
                rk, sk = rank_df[b].dropna(), skill_df[b].dropna()
                row.update({"rank_median": float(rk.median()),
                            "rank_lo95": float(rk.quantile(0.025)),
                            "rank_hi95": float(rk.quantile(0.975)),
                            "skill_lo95": float(sk.quantile(0.025)),
                            "skill_hi95": float(sk.quantile(0.975)),
                            "boot_coverage": len(rk) / max(len(rank_df), 1),
                            "boot_level": level, "boot_lower_bound": lower_bound})
            all_cons.append(row)
            all_emb.append({"block": bi, "bot": b, "z1": z[0],
                            "z2": z[1] if DIM > 1 else np.nan})
        for spec in fit.conditions:
            v = fit.V[spec.cond_id]
            all_rays.append({"block": bi, "cond_id": spec.cond_id,
                             "v1": v[0], "v2": v[1] if DIM > 1 else np.nan,
                             "beta": fit.beta[spec.cond_id],
                             "n_races": int(bc[bc["cond_id"] == spec.cond_id]
                                            ["n_races"].iloc[0])})
            p_hat = fit.predict_condition(spec.cond_id)
            for k_, b in enumerate(spec.item_ids):
                all_abil.append({"block": bi, "cond_id": spec.cond_id, "bot": b,
                                 "ability_racetime": fit.ability(spec.cond_id, b),
                                 "skill": -fit.ability(spec.cond_id, b),
                                 "p_obs": spec.prices[k_], "p_hat": p_hat[k_]})
        print()

    pd.DataFrame(all_emb).to_csv(OUT / "embedding.csv", index=False)
    pd.DataFrame(all_rays).to_csv(OUT / "rays.csv", index=False)
    abil = pd.DataFrame(all_abil)
    abil.to_csv(OUT / "abilities.csv", index=False)
    cond[["cond_id", "cluster", "bot", "n_races", "field_size", "win_credit",
          "price"]].to_csv(OUT / "prices.csv", index=False)
    agreement = pd.DataFrame(ag_rows)
    agreement.to_csv(OUT / "agreement.csv", index=False)
    consensus = (pd.DataFrame(all_cons)
                 .sort_values(["block", "consensus_skill"], ascending=[True, False]))
    consensus.to_csv(OUT / "consensus.csv", index=False)

    print("=== AGREEMENT ===")
    print(agreement.to_string(index=False))
    print("\n=== CONSENSUS BOARD (skill = -ability, higher better; "
          "per-block gauge; cross-block gaps NOT identified) ===")
    with pd.option_context("display.width", 220):
        print(consensus.to_string(index=False))
    print("\nunplaced bots (no fixed-field condition with >= "
          f"{MIN_RACES_COND} races): {unplaced}")
    print("\n=== FITTED vs OBSERVED PRICES ===")
    with pd.option_context("display.width", 220):
        print(abil.to_string(index=False))


if __name__ == "__main__":
    main()

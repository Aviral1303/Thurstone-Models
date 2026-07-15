"""Bot arena: head-to-head ranking of quantbots on the Manifold clone.

Bots = horses, markets = races. Design approved by user 2026-07-14:

- Race = a resolved YES/NO market on the clone with >= 2 distinct bots holding
  ENTRY positions (CANCEL markets excluded: refunds make every entrant flat;
  unresolved excluded: mark-to-market mostly never realizes given ~93% cancel
  rate and is contaminated by the bots' own CPMM price impact).
- Performance per (bot, race) = realized log-return on invested mana:
      r = log((resolution payout + exit proceeds) / mana invested)
  Size-invariant on purpose: raw PnL would rank bankrolls, not bettors.
- Races decompose into within-market pairs; tie when |delta r| <= EPS_TIE
  (primary 0.01, sensitivity 0.005 / 0.02). Both-zero-payout pairs are ties.
- Fit with the paper machinery (src/fit.py, mode="half_tie": identical tie
  treatment both links, per standing rule): BT (logistic) baseline + lattice
  Thurstone at units 0.1 / 0.5855 / 0.8002. Primary board = lattice 0.5855.
  NOTE: this is pairwise decomposition of multi-entrant races; it does NOT
  test Cotton's multi-entrant coherence (standing rule).
- Anchor: max-coverage bot pinned to 0 for every fit (display gauge only;
  bootstrap uses gauge-invariant ranks + median-aligned theta).
- Uncertainty: cluster bootstrap over UNDERLYINGS (strike ladders on the same
  underlying resolve together and are not independent races), B=2000.
- Headline board restricted to bots with >= MIN_RACES races; thinner bots
  reported unrated.
- Companion skill metrics (execution-order-independent caveat: the daily
  cycle runs bots in a fixed priority order, so realized returns include
  queue position): mana-weighted directional hit rate + Brier of
  llm_estimate where present, over ALL resolved YES/NO markets each bot
  traded (not just multi-bot ones).

Outputs under results/bot_arena/:
  returns.csv, battles.csv, board.csv, bootstrap_ranks.csv,
  agreement.csv, companion.csv
"""

import re
import sqlite3
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchoring import anchor  # noqa: E402
from fit import fit_gaplink  # noqa: E402
from lattice_link import LatticeLink, LogisticLink  # noqa: E402

DB_PATH = Path("/Users/mikhail/Bots/data/quantbots.sqlite")
OUT = ROOT / "results" / "bot_arena"
OUT.mkdir(parents=True, exist_ok=True)

EPS_TIE = 0.01
EPS_SENS = (0.005, 0.02)
UNITS = (0.1, 0.5855, 0.8002)
PRIMARY_UNIT = 0.5855
MIN_RACES = 5
B_BOOT = 2000
SEED = 20260714


# ---------------------------------------------------------------- extraction

def load_ledger():
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    trades = pd.read_sql_query(
        """SELECT t.bot_id, b.name AS bot, t.market_id, t.trade_type,
                  t.direction, t.amount, t.shares, t.llm_estimate,
                  t.date_executed
           FROM trade t JOIN bot b USING(bot_id)""", con)
    markets = pd.read_sql_query(
        """SELECT market_id, question, resolution, is_resolved
           FROM market_cache""", con)
    con.close()
    return trades, markets


def underlying_cluster(question: str) -> str:
    """Map a market question to its underlying for the cluster bootstrap.

    Conservative: ALL cotton series (price, open interest, CFTC positioning)
    share one cluster — they resolve off the same underlying/report family.
    Unmatched questions get their own singleton cluster.
    """
    q = (question or "").lower()
    keys = [
        ("cotton", "cotton"), ("gold", "gold"), ("silver", "silver"),
        ("platinum", "platinum"), ("palladium", "palladium"),
        ("copper", "copper"), ("wti", "wti"), ("brent", "brent"),
        ("rbob", "rbob"), ("gasoline", "rbob"), ("cerium", "cerium"),
        ("cocoa", "cocoa"), ("coffee", "coffee"), ("sugar", "sugar"),
        ("wheat", "wheat"), ("corn", "corn"), ("soybean", "soybean"),
        ("natural gas", "natgas"), ("nickel", "nickel"), ("zinc", "zinc"),
        ("aluminium", "aluminium"), ("aluminum", "aluminium"),
        ("lithium", "lithium"), ("uranium", "uranium"), ("crude oil", "oil"),
    ]
    for kw, cluster in keys:
        if kw in q:
            return cluster
    return f"other:{q[:40]}"


def per_market_returns(trades: pd.DataFrame, markets: pd.DataFrame) -> pd.DataFrame:
    """Realized return ratio per (bot, market) on resolved YES/NO markets."""
    res = markets[markets["resolution"].isin(["YES", "NO"])].set_index("market_id")
    t = trades[trades["market_id"].isin(res.index)].copy()
    t["resolution"] = t["market_id"].map(res["resolution"])

    invested = (t[t["trade_type"] == "ENTRY"]
                .groupby(["bot", "market_id"])["amount"].sum())
    proceeds = (t[t["trade_type"].isin(["EXIT", "PARTIAL_EXIT"])]
                .groupby(["bot", "market_id"])["amount"].sum())
    rc = t[t["trade_type"] == "RESOLUTION_CLOSE"].copy()
    rc["payout"] = np.where(rc["direction"] == rc["resolution"], rc["shares"], 0.0)
    payout = rc.groupby(["bot", "market_id"])["payout"].sum()

    df = invested.rename("invested").to_frame()
    df["proceeds"] = proceeds.reindex(df.index).fillna(0.0)
    df["payout"] = payout.reindex(df.index).fillna(0.0)
    df = df[df["invested"] > 0].reset_index()
    df["ratio"] = (df["proceeds"] + df["payout"]) / df["invested"]
    df["resolution"] = df["market_id"].map(res["resolution"])
    df["question"] = df["market_id"].map(res["question"])
    df["cluster"] = df["question"].map(underlying_cluster)
    # series = question with numbers stripped: collapses thresholds AND dates
    # of one ladder family; finer than underlying, coarser than market.
    df["series"] = df["question"].map(
        lambda q: re.sub(r"\s+", " ", re.sub(r"\d+(\.\d+)?", "", (q or "").lower())).strip())
    return df


def build_battles(returns: pd.DataFrame, eps: float) -> pd.DataFrame:
    """Decompose each multi-bot race into within-market pairwise outcomes."""
    rows = []
    for mid, g in returns.groupby("market_id"):
        if len(g) < 2:
            continue
        recs = g.set_index("bot")[["ratio", "cluster", "series"]]
        for a, b in combinations(sorted(recs.index), 2):
            ra, rb = recs.loc[a, "ratio"], recs.loc[b, "ratio"]
            if ra <= 0 and rb <= 0:
                winner = "tie"
            elif ra <= 0:
                winner = "model_b"
            elif rb <= 0:
                winner = "model_a"
            elif abs(np.log(ra / rb)) <= eps:
                winner = "tie"
            else:
                winner = "model_a" if ra > rb else "model_b"
            rows.append({"model_a": a, "model_b": b, "winner": winner,
                         "market_id": mid, "cluster": recs.loc[a, "cluster"],
                         "series": recs.loc[a, "series"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- analysis

def win_digraph_sccs(battles: pd.DataFrame) -> list[set]:
    """Identifiable blocks = strongly connected components of the win digraph.

    Ford (1957): the BT/gap-link MLE gap between two groups is finite iff win
    weight flows in BOTH directions across every partition. In half_tie mode a
    decisive win adds weight a->b; a tie adds weight both ways. Cross-SCC
    theta gaps are ridge artifacts, not estimates — rank only within SCCs.
    """
    edges = defaultdict(set)
    nodes = set(battles["model_a"]) | set(battles["model_b"])
    for _, r in battles.iterrows():
        if r["winner"] == "tie":
            edges[r["model_a"]].add(r["model_b"])
            edges[r["model_b"]].add(r["model_a"])
        elif r["winner"] == "model_a":
            edges[r["model_a"]].add(r["model_b"])
        else:
            edges[r["model_b"]].add(r["model_a"])

    # Kosaraju on <= dozens of nodes
    order, seen = [], set()

    def dfs(u, adj, acc):
        stack = [(u, iter(adj[u]))]
        seen.add(u)
        while stack:
            node, it = stack[-1]
            for v in it:
                if v not in seen:
                    seen.add(v)
                    stack.append((v, iter(adj[v])))
                    break
            else:
                stack.pop()
                acc.append(node)

    for u in nodes:
        if u not in seen:
            dfs(u, edges, order)
    redges = defaultdict(set)
    for u, vs in edges.items():
        for v in vs:
            redges[v].add(u)
    seen = set()
    sccs = []
    for u in reversed(order):
        if u not in seen:
            comp = []
            dfs(u, redges, comp)
            sccs.append(set(comp))
    return sorted(sccs, key=len, reverse=True)


def connected_components(battles: pd.DataFrame) -> list[set]:
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, r in battles.iterrows():
        ra, rb = find(r["model_a"]), find(r["model_b"])
        if ra != rb:
            parent[ra] = rb
    comps = defaultdict(set)
    for x in parent:
        comps[find(x)].add(x)
    return sorted(comps.values(), key=len, reverse=True)


def fit_all(battles: pd.DataFrame, links: dict, anchor_bot: str) -> pd.DataFrame:
    out = {}
    for name, link in links.items():
        theta = fit_gaplink(battles, link, mode="half_tie")
        out[name] = anchor(theta, model=anchor_bot, value=0.0)
    return pd.DataFrame(out)


def cluster_bootstrap(battles: pd.DataFrame, link, headline: list[str],
                      theta_full: pd.Series, b_boot: int, seed: int,
                      group_col: str = "cluster"):
    """Bootstrap over resolution clusters (group_col). Returns (rank_df, theta_df)."""
    rng = np.random.default_rng(seed)
    clusters = sorted(battles[group_col].unique())
    by_cluster = {c: battles[battles[group_col] == c] for c in clusters}
    ranks, thetas, n_fail = [], [], 0
    for _ in range(b_boot):
        draw = rng.choice(clusters, size=len(clusters), replace=True)
        bb = pd.concat([by_cluster[c] for c in draw], ignore_index=True)
        present_pair = set(bb["model_a"]) | set(bb["model_b"])
        if len(present_pair) < 2:
            n_fail += 1
            continue
        try:
            th = fit_gaplink(bb, link, mode="half_tie")
        except (RuntimeError, ValueError):
            n_fail += 1
            continue
        head_present = [m for m in headline if m in th.index]
        if len(head_present) >= 2:
            # median-align to the full fit over shared headline bots (gauge)
            th = th + (theta_full[head_present].median() - th[head_present].median())
            order = th[head_present].rank(ascending=False)
            ranks.append(order.reindex(headline))
        thetas.append(th.reindex(headline))
    return pd.DataFrame(ranks), pd.DataFrame(thetas), n_fail


def kendall(a: pd.Series, b: pd.Series) -> float:
    from scipy.stats import kendalltau
    common = a.index.intersection(b.index)
    return float(kendalltau(a[common], b[common]).statistic)


# ---------------------------------------------------------------- companion

def companion_metrics(trades: pd.DataFrame, markets: pd.DataFrame,
                      returns_all: pd.DataFrame) -> pd.DataFrame:
    """Execution-independent skill metrics over ALL resolved YES/NO markets."""
    res = markets[markets["resolution"].isin(["YES", "NO"])].set_index("market_id")
    ent = trades[(trades["trade_type"] == "ENTRY")
                 & trades["market_id"].isin(res.index)].copy()
    ent["resolution"] = ent["market_id"].map(res["resolution"])
    ent["hit"] = (ent["direction"] == ent["resolution"]).astype(float)
    ent["outcome"] = (ent["resolution"] == "YES").astype(float)

    rows = []
    for bot, g in ent.groupby("bot"):
        w = g["amount"]
        est = g.dropna(subset=["llm_estimate"])
        agg = returns_all[returns_all["bot"] == bot]
        invested = agg["invested"].sum()
        realized = (agg["proceeds"] + agg["payout"] - agg["invested"]).sum()
        rows.append({
            "bot": bot,
            "n_resolved_markets": agg.shape[0],
            "invested_mana": invested,
            "realized_pnl": realized,
            "return_per_mana": realized / invested if invested else np.nan,
            "hit_rate_mana_weighted": float((g["hit"] * w).sum() / w.sum()),
            "brier_llm_estimate": (float(((est["llm_estimate"] - est["outcome"]) ** 2).mean())
                                    if len(est) else np.nan),
            "estimate_coverage": len(est) / len(g),
        })
    return pd.DataFrame(rows).sort_values("return_per_mana", ascending=False)


# ---------------------------------------------------------------- main

def main():
    trades, markets = load_ledger()
    returns_all = per_market_returns(trades, markets)

    # races = multi-bot resolved markets
    nb = returns_all.groupby("market_id")["bot"].nunique()
    race_ids = nb[nb >= 2].index
    races = returns_all[returns_all["market_id"].isin(race_ids)].copy()
    races.to_csv(OUT / "returns.csv", index=False)

    n_races = races.groupby("bot")["market_id"].nunique().sort_values(ascending=False)
    print(f"races: {len(race_ids)} markets, {len(n_races)} bots, "
          f"clusters: {races['cluster'].nunique()}")
    print(n_races.to_string())

    battles = build_battles(races, EPS_TIE)
    battles.to_csv(OUT / "battles.csv", index=False)
    n_pairs = len(battles)
    tie_share = float((battles["winner"] == "tie").mean())
    print(f"\npairwise comparisons: {n_pairs}, tie share {tie_share:.3f}")

    comps = connected_components(battles)
    print(f"connected components: {[len(c) for c in comps]}")
    if len(comps) > 1:
        keep = comps[0]
        dropped = sorted(set().union(*comps[1:]))
        print(f"restricting to giant component; dropped: {dropped}")
        battles = battles[battles["model_a"].isin(keep) & battles["model_b"].isin(keep)]
        races = races[races["bot"].isin(keep)]

    # identifiable blocks: cross-SCC gaps are ridge artifacts (Ford 1957)
    sccs = win_digraph_sccs(battles)
    leagues = [s for s in sccs if len(s) >= 2]
    singletons = sorted(set().union(*[s for s in sccs if len(s) < 2])) if len(sccs) > len(leagues) else []
    print(f"\nwin-digraph SCCs (identifiable leagues): {[sorted(s) for s in sccs]}")
    league_of = {}
    for li, s in enumerate(leagues):
        for bot in s:
            league_of[bot] = li

    def cross_league(r):
        la, lb = league_of.get(r["model_a"]), league_of.get(r["model_b"])
        return la != lb or la is None
    bridge = battles[battles.apply(cross_league, axis=1)]
    if len(bridge):
        print("cross-league bridge comparisons (direction only, gap NOT identified):")
        for _, r in bridge.iterrows():
            print(f"  {r['model_a']} vs {r['model_b']} -> {r['winner']} [{r['cluster']}]")
    bridge.to_csv(OUT / "bridge_comparisons.csv", index=False)

    links = {"bt": LogisticLink()}
    for u in UNITS:
        links[f"lattice_u{u}"] = LatticeLink(unit=u)
    primary_col = f"lattice_u{PRIMARY_UNIT}"

    all_boards, ag_rows = [], []
    for li, league in enumerate(leagues):
        lb = battles[battles["model_a"].isin(league) & battles["model_b"].isin(league)]
        lraces = races[races["bot"].isin(league)]
        anchor_bot = n_races[n_races.index.isin(league)].index[0]
        theta = fit_all(lb, links, anchor_bot)
        headline = sorted(b for b in league
                          if n_races.get(b, 0) >= MIN_RACES and b in theta.index)
        print(f"\n--- league {li}: {sorted(league)} (anchor {anchor_bot}) ---")

        # agreement across links + tie-eps sensitivity (headline bots only)
        if len(headline) >= 3:
            for col in theta.columns:
                if col != primary_col:
                    ag_rows.append({"league": li,
                                    "comparison": f"{primary_col} vs {col}",
                                    "kendall_tau": kendall(theta.loc[headline, primary_col],
                                                           theta.loc[headline, col])})
            for eps in EPS_SENS:
                b_eps = build_battles(lraces, eps)
                b_eps = b_eps[b_eps["model_a"].isin(league) & b_eps["model_b"].isin(league)]
                th = anchor(fit_gaplink(b_eps, links[primary_col], mode="half_tie"),
                            model=anchor_bot, value=0.0)
                ag_rows.append({"league": li,
                                "comparison": f"{primary_col} eps={EPS_TIE} vs eps={eps}",
                                "kendall_tau": kendall(theta.loc[headline, primary_col],
                                                       th.reindex(headline)),
                                "tie_share": float((b_eps["winner"] == "tie").mean())})

        # bootstrap (primary lattice + BT). Resample at the coarsest level
        # that still has >= 2 groups: underlying > series > market. Finer
        # levels understate correlation -> CIs are lower bounds; label them.
        boot_level = next((c for c in ("cluster", "series", "market_id")
                           if lb[c].nunique() >= 2), None)
        if boot_level is None:
            print("bootstrap: SKIPPED (single resolution group — no CI possible)")
            rank_ci = {primary_col: pd.DataFrame()}
            theta_ci = {primary_col: pd.DataFrame()}
        else:
            print(f"bootstrap resampling level: {boot_level} "
                  f"({lb[boot_level].nunique()} groups)"
                  + ("  [finer than underlying -> CIs are lower bounds]"
                     if boot_level != "cluster" else ""))
            rank_ci, theta_ci = {}, {}
            for col in (primary_col, "bt"):
                rk, th, n_fail = cluster_bootstrap(
                    lb, links[col], headline, theta[col], B_BOOT, SEED,
                    group_col=boot_level)
                rank_ci[col] = rk
                theta_ci[col] = th
                print(f"bootstrap [{col}]: {len(rk)} rank replicates, {n_fail} failed")
        rank_primary = rank_ci[primary_col]
        rank_primary.to_csv(OUT / f"bootstrap_ranks_league{li}.csv", index=False)

        wins = defaultdict(int); losses = defaultdict(int); ties = defaultdict(int)
        for _, r in lb.iterrows():
            if r["winner"] == "tie":
                ties[r["model_a"]] += 1; ties[r["model_b"]] += 1
            elif r["winner"] == "model_a":
                wins[r["model_a"]] += 1; losses[r["model_b"]] += 1
            else:
                wins[r["model_b"]] += 1; losses[r["model_a"]] += 1

        for bot in theta.index:
            rated = bot in headline
            row = {
                "league": li,
                "bot": bot,
                "rated": rated,
                "n_races": int(n_races.get(bot, 0)),
                "wins": wins[bot], "losses": losses[bot], "ties": ties[bot],
                "theta_lattice": theta.loc[bot, primary_col],
                "theta_bt": theta.loc[bot, "bt"],
            }
            if rated and bot in rank_primary.columns:
                rk = rank_primary[bot].dropna()
                th = theta_ci[primary_col][bot].dropna()
                row.update({
                    "rank_median": float(rk.median()),
                    "rank_lo95": float(rk.quantile(0.025)),
                    "rank_hi95": float(rk.quantile(0.975)),
                    "theta_lo95": float(th.quantile(0.025)),
                    "theta_hi95": float(th.quantile(0.975)),
                    "boot_coverage": len(rk) / max(len(rank_primary), 1),
                    "boot_level": boot_level,
                })
            all_boards.append(row)

    for bot in singletons:
        all_boards.append({"league": None, "bot": bot, "rated": False,
                           "n_races": int(n_races.get(bot, 0))})

    agreement = pd.DataFrame(ag_rows)
    agreement.to_csv(OUT / "agreement.csv", index=False)
    print("\nagreement:")
    print(agreement.to_string(index=False))

    board = (pd.DataFrame(all_boards)
             .sort_values(["league", "rated", "theta_lattice"],
                          ascending=[True, False, False], na_position="last"))
    board.to_csv(OUT / "board.csv", index=False)

    comp = companion_metrics(trades, markets, returns_all)
    comp.to_csv(OUT / "companion.csv", index=False)

    print("\n=== BOARD (primary: lattice u=%s; per-league anchor=0; "
          "cross-league gaps NOT identified) ===" % PRIMARY_UNIT)
    with pd.option_context("display.width", 200):
        print(board.to_string(index=False))
    print("\n=== COMPANION (all resolved YES/NO markets, execution-independent-ish) ===")
    with pd.option_context("display.width", 200):
        print(comp.to_string(index=False))


if __name__ == "__main__":
    main()

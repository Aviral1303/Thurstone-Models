"""Platform-wide bot leaderboard for the Manifold clone (extends script 33).

Script 33 ranked only OUR quantbots because the local ledger records only our
trades. This script pulls the WHOLE platform from the clone's v0 API (read-only
GETs, throttled under the 500 req/min limit), then runs the same pipeline over
every account: races = resolved YES/NO binary markets with >= 2 traders,
performance = realized return per mana, pairwise duels, SCC leagues,
half_tie fits (BT + lattice), cluster bootstrap over underlyings.

Usage (credentials come from Doppler in ~/Bots):
    cd ~/Bots && doppler run -- <thurstone-venv-python> scripts/34_platform_leaderboard.py fetch
    <thurstone-venv-python> scripts/34_platform_leaderboard.py analyze

fetch is incremental-ish: it rewrites each cache file whole (simple + atomic),
markets/users are small; bets is the big one and supports resume via --resume
(continues from the oldest cached bet id).

Accounting per (user, market) from raw bets (Manifold semantics):
    gross  = sum of positive bet amounts (mana out: buys)
    proceeds = -(sum of negative amounts)  (mana in: sells + redemptions)
    shares[outcome] summed signed -> payout = shares[resolution] if > 0
    ratio  = (payout + proceeds) / gross
Redemption bet pairs (isRedemption) are included: they return mana and burn
matched YES/NO shares, which is exactly how the platform accounts them.
Users with gross <= 0 in a market (pure redemption artifacts) are dropped.
"""

import gzip
import json
import os
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

RAW = ROOT / "data" / "raw" / "clone_api"
RAW.mkdir(parents=True, exist_ok=True)
OUT = ROOT / "results" / "platform_leaderboard"
OUT.mkdir(parents=True, exist_ok=True)

BASE = "https://manifold.mikhailtal.dev/api/v0"
THROTTLE_S = 0.13  # ~460 req/min worst case, under the 500/min limit

EPS_TIE = 0.01
EPS_SENS = (0.005, 0.02)
UNITS = (0.1, 0.5855, 0.8002)
PRIMARY_UNIT = 0.5855
MIN_RACES = 20      # headline threshold (platform scale; sensitivity at 5)
B_BOOT = 1000
SEED = 20260714


# ---------------------------------------------------------------- fetch

def _get(path: str):
    req = urllib.request.Request(BASE + path)
    req.add_header("Authorization", "Key " + os.environ["MANIFOLD_CLONE_API_KEY"])
    req.add_header("CF-Access-Client-Id", os.environ["CF_ACCESS_CLIENT_ID"])
    req.add_header("CF-Access-Client-Secret", os.environ["CF_ACCESS_CLIENT_SECRET"])
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 — retry any transient failure
            if attempt == 4:
                raise
            wait = 2 ** attempt
            print(f"  retry {attempt + 1} after {type(e).__name__}: {e} (sleep {wait}s)")
            time.sleep(wait)


def _fetch_paged(endpoint: str, id_key: str, limit: int, out_path: Path,
                 resume_before: str | None = None):
    n, before = 0, resume_before
    mode = "at" if resume_before else "wt"
    with gzip.open(out_path, mode) as f:
        while True:
            q = f"{endpoint}?limit={limit}" + (f"&before={before}" if before else "")
            page = _get(q)
            if not page:
                break
            for row in page:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
            n += len(page)
            before = page[-1][id_key]
            if n % (limit * 20) == 0:
                print(f"  {endpoint}: {n} rows...", flush=True)
            if len(page) < limit:
                break
            time.sleep(THROTTLE_S)
    print(f"{endpoint}: {n} rows fetched -> {out_path.name}")


def fetch(resume: bool = False):
    _fetch_paged("/users", "id", 500, RAW / "users.jsonl.gz")
    _fetch_paged("/markets", "id", 1000, RAW / "markets.jsonl.gz")
    bets_path = RAW / "bets.jsonl.gz"
    resume_before = None
    if resume and bets_path.exists():
        last = None
        with gzip.open(bets_path, "rt") as f:
            for line in f:
                last = line
        if last:
            resume_before = json.loads(last)["id"]
            print(f"resuming bets before id {resume_before}")
    _fetch_paged("/bets", "id", 1000, bets_path, resume_before)


# ---------------------------------------------------------------- analyze

def _load_jsonl(path: Path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def build_returns():
    import pandas as pd

    users = {u["id"]: u for u in _load_jsonl(RAW / "users.jsonl.gz")}
    markets = {}
    for m in _load_jsonl(RAW / "markets.jsonl.gz"):
        if (m.get("outcomeType") == "BINARY" and m.get("isResolved")
                and m.get("resolution") in ("YES", "NO")):
            markets[m["id"]] = m

    acc = defaultdict(lambda: {"gross": 0.0, "proceeds": 0.0,
                               "YES": 0.0, "NO": 0.0})
    n_bets = 0
    for b in _load_jsonl(RAW / "bets.jsonl.gz"):
        cid = b.get("contractId")
        if cid not in markets:
            continue
        a = acc[(b["userId"], cid)]
        amt = float(b.get("amount") or 0.0)
        if amt >= 0:
            a["gross"] += amt
        else:
            a["proceeds"] += -amt
        oc = b.get("outcome")
        if oc in ("YES", "NO"):
            a[oc] += float(b.get("shares") or 0.0)
        n_bets += 1
    print(f"bets on resolved YES/NO binary markets: {n_bets}")

    rows = []
    for (uid, cid), a in acc.items():
        if a["gross"] <= 1e-9:
            continue
        m = markets[cid]
        payout = max(a[m["resolution"]], 0.0)
        u = users.get(uid, {})
        rows.append({
            "bot": u.get("username", uid),
            "user_id": uid,
            "market_id": cid,
            "invested": a["gross"],
            "proceeds": a["proceeds"],
            "payout": payout,
            "ratio": (payout + a["proceeds"]) / a["gross"],
            "resolution": m["resolution"],
            "question": m.get("question", ""),
        })
    return pd.DataFrame(rows)


def _aggregate(battles):
    """Collapse duplicate votes into weighted rows (identical likelihood).

    2.5M platform battles hold only ~19k distinct (a, b, winner) votes; the
    per-vote MLE is O(rows), so fitting the weighted frame is ~100x faster.
    """
    return (battles.groupby(["model_a", "model_b", "winner"], as_index=False)
            .size().rename(columns={"size": "weight"}))


def cluster_bootstrap_weighted(battles, link, headline, theta_full,
                               b_boot, seed, group_col="cluster"):
    """Cluster bootstrap via weight multiplication (script 33 semantics).

    Drawing cluster c with multiplicity m and concatenating equals multiplying
    its vote weights by m — same likelihood, no million-row concat per
    replicate. Same seed => same cluster draws as mod33.cluster_bootstrap.
    """
    import numpy as np
    import pandas as pd
    from fit import fit_gaplink

    rng = np.random.default_rng(seed)
    clusters = sorted(battles[group_col].unique())
    agg = (battles.groupby([group_col, "model_a", "model_b", "winner"],
                           as_index=False)
           .size().rename(columns={"size": "weight"}))
    cl_codes = pd.Categorical(agg[group_col], categories=clusters).codes
    base_w = agg["weight"].to_numpy(float)
    ranks, thetas, n_fail = [], [], 0
    for _ in range(b_boot):
        draw = rng.choice(clusters, size=len(clusters), replace=True)
        mult = np.bincount(pd.Categorical(draw, categories=clusters).codes,
                           minlength=len(clusters))
        m = mult[cl_codes]
        keep = m > 0
        bb = agg.loc[keep, ["model_a", "model_b", "winner"]].copy()
        bb["weight"] = base_w[keep] * m[keep]
        bb = bb.groupby(["model_a", "model_b", "winner"],
                        as_index=False)["weight"].sum()
        if len(set(bb["model_a"]) | set(bb["model_b"])) < 2:
            n_fail += 1
            continue
        try:
            th = fit_gaplink(bb, link, mode="half_tie")
        except (RuntimeError, ValueError):
            n_fail += 1
            continue
        head_present = [x for x in headline if x in th.index]
        if len(head_present) >= 2:
            th = th + (theta_full[head_present].median() - th[head_present].median())
            ranks.append(th[head_present].rank(ascending=False).reindex(headline))
        thetas.append(th.reindex(headline))
    return pd.DataFrame(ranks), pd.DataFrame(thetas), n_fail


def analyze(cached: bool = False):
    import numpy as np
    import pandas as pd

    mod33 = __import__("importlib").import_module("33_bot_arena_ranking")
    from anchoring import anchor
    from fit import fit_gaplink
    from lattice_link import LatticeLink, LogisticLink

    if cached and (OUT / "returns_all.csv").exists():
        returns_all = pd.read_csv(OUT / "returns_all.csv")
        for col in ("question", "cluster", "series"):
            returns_all[col] = returns_all[col].fillna("")
        print(f"cached returns: {len(returns_all)} rows")
    else:
        returns_all = build_returns()
        returns_all["cluster"] = returns_all["question"].map(mod33.underlying_cluster)
        import re
        returns_all["series"] = returns_all["question"].map(
            lambda q: re.sub(r"\s+", " ", re.sub(r"\d+(\.\d+)?", "", (q or "").lower())).strip())
        returns_all.to_csv(OUT / "returns_all.csv", index=False)

    nb = returns_all.groupby("market_id")["bot"].nunique()
    race_ids = nb[nb >= 2].index
    races = returns_all[returns_all["market_id"].isin(race_ids)].copy()
    n_races = races.groupby("bot")["market_id"].nunique().sort_values(ascending=False)
    print(f"races: {len(race_ids)} | traders: {len(n_races)} | "
          f"clusters: {races['cluster'].nunique()}")
    print("race-count distribution (traders per market):")
    print(nb[nb >= 2].value_counts().sort_index().to_string())

    battles = mod33.build_battles(races, EPS_TIE)
    print(f"pairwise duels: {len(battles)} | tie share "
          f"{(battles['winner'] == 'tie').mean():.3f}")
    battles.to_csv(OUT / "battles.csv", index=False)

    sccs = mod33.win_digraph_sccs(battles)
    sizes = [len(s) for s in sccs]
    print(f"win-digraph SCCs: {len(sccs)} (sizes: {sorted(sizes, reverse=True)[:10]}...)")
    giant = sccs[0]
    in_giant = races["bot"].isin(giant)
    print(f"giant SCC: {len(giant)} traders covering "
          f"{races[in_giant]['market_id'].nunique()} races")
    (OUT / "sccs.json").write_text(json.dumps(
        [sorted(s) for s in sccs], indent=1))

    lb = battles[battles["model_a"].isin(giant) & battles["model_b"].isin(giant)]
    lraces = races[in_giant]
    anchor_bot = n_races[n_races.index.isin(giant)].index[0]
    links = {"bt": LogisticLink()}
    for u in UNITS:
        links[f"lattice_u{u}"] = LatticeLink(unit=u)
    primary_col = f"lattice_u{PRIMARY_UNIT}"
    lb_agg = _aggregate(lb)
    print(f"aggregation: {len(lb)} votes -> {len(lb_agg)} weighted rows")
    theta = mod33.fit_all(lb_agg, links, anchor_bot)
    headline = sorted(b for b in giant
                      if n_races.get(b, 0) >= MIN_RACES and b in theta.index)
    print(f"headline traders (>= {MIN_RACES} races, in giant SCC): {len(headline)}")

    ag_rows = []
    for col in theta.columns:
        if col != primary_col:
            ag_rows.append({"comparison": f"{primary_col} vs {col}",
                            "kendall_tau": mod33.kendall(
                                theta.loc[headline, primary_col],
                                theta.loc[headline, col])})
    for eps in EPS_SENS:
        b_eps = mod33.build_battles(lraces, eps)
        b_eps = b_eps[b_eps["model_a"].isin(giant) & b_eps["model_b"].isin(giant)]
        th = anchor(fit_gaplink(_aggregate(b_eps), links[primary_col], mode="half_tie"),
                    model=anchor_bot, value=0.0)
        ag_rows.append({"comparison": f"eps={EPS_TIE} vs eps={eps}",
                        "kendall_tau": mod33.kendall(
                            theta.loc[headline, primary_col], th.reindex(headline)),
                        "tie_share": float((b_eps["winner"] == "tie").mean())})
    agreement = pd.DataFrame(ag_rows)
    agreement.to_csv(OUT / "agreement.csv", index=False)
    print(agreement.to_string(index=False))

    boot_level = next((c for c in ("cluster", "series", "market_id")
                       if lb[c].nunique() >= 2), None)
    print(f"bootstrap level: {boot_level} ({lb[boot_level].nunique()} groups), B={B_BOOT}")
    rk, th_b, n_fail = cluster_bootstrap_weighted(
        lb, links[primary_col], headline, theta[primary_col], B_BOOT, SEED,
        group_col=boot_level)
    print(f"bootstrap: {len(rk)} replicates, {n_fail} failed")
    rk.to_csv(OUT / "bootstrap_ranks.csv", index=False)

    wins = defaultdict(int); losses = defaultdict(int); ties = defaultdict(int)
    for _, r in lb_agg.iterrows():
        w = int(r["weight"])
        if r["winner"] == "tie":
            ties[r["model_a"]] += w; ties[r["model_b"]] += w
        elif r["winner"] == "model_a":
            wins[r["model_a"]] += w; losses[r["model_b"]] += w
        else:
            wins[r["model_b"]] += w; losses[r["model_a"]] += w

    ra = returns_all.assign(
        realized=returns_all["proceeds"] + returns_all["payout"] - returns_all["invested"])
    pnl = ra.groupby("bot").agg(
        n_markets=("market_id", "nunique"),
        invested=("invested", "sum"),
        realized=("realized", "sum"))
    pnl["ret_per_mana"] = pnl["realized"] / pnl["invested"]

    rows = []
    for bot in theta.index:
        rated = bot in headline
        row = {
            "bot": bot, "rated": rated, "n_races": int(n_races.get(bot, 0)),
            "wins": wins[bot], "losses": losses[bot], "ties": ties[bot],
            "theta_lattice": theta.loc[bot, primary_col],
            "theta_bt": theta.loc[bot, "bt"],
        }
        if bot in pnl.index:
            row.update({k: float(pnl.loc[bot, k]) for k in
                        ("n_markets", "invested", "realized", "ret_per_mana")})
        if rated and bot in rk.columns:
            rkb = rk[bot].dropna()
            thb = th_b[bot].dropna()
            row.update({
                "rank_median": float(rkb.median()),
                "rank_lo95": float(rkb.quantile(0.025)),
                "rank_hi95": float(rkb.quantile(0.975)),
                "theta_lo95": float(thb.quantile(0.025)),
                "theta_hi95": float(thb.quantile(0.975)),
            })
        rows.append(row)
    board = (pd.DataFrame(rows)
             .sort_values(["rated", "theta_lattice"], ascending=[False, False]))
    board.to_csv(OUT / "board.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 400):
        print(board.head(60).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "analyze"
    if cmd == "fetch":
        fetch(resume="--resume" in sys.argv)
    elif cmd == "analyze":
        analyze(cached="--cached" in sys.argv)
    else:
        raise SystemExit(f"unknown command: {cmd}")

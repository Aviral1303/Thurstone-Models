# Bot Arena: Thurstone head-to-head ranking of quantbots on the Manifold clone

**Date:** 2026-07-14 · **Script:** `scripts/33_bot_arena_ranking.py` ·
**Outputs:** `results/bot_arena/` · **Data:** `~/Bots/data/quantbots.sqlite`
(read-only), ledger through 2026-07-14.

Side project from the main paper track (user request 2026-07-14): rank the
trading bots on the private Manifold clone by head-to-head performance on
shared markets, using the lattice-Thurstone machinery. Bots = horses,
markets = races.

## Design (approved 2026-07-14)

- **Race** = resolved YES/NO market with ≥ 2 bots holding ENTRY positions.
  CANCEL markets excluded (refunds → every entrant flat, zero information);
  unresolved markets excluded (mark-to-market mostly never realizes under
  the clone's ~93% cancel rate and is contaminated by the bots' own CPMM
  price impact).
- **Performance** per (bot, race) = realized log-return on invested mana,
  r = log((resolution payout + exit proceeds) / mana invested).
  Size-invariant by construction. Payout computation verified by hand against
  ledger-credited amounts on one winning and one losing case (exact match).
  NB: `price_after = 1.0` on RESOLUTION_CLOSE rows does NOT mean the bet won —
  never use that field for outcome attribution.
- **Comparisons**: races decomposed into within-market pairs; tie when
  |Δr| ≤ 0.01 (sensitivity 0.005 / 0.02); both-wiped-out pairs are ties.
  This is pairwise decomposition — it does NOT test Cotton's multi-entrant
  coherence (standing rule).
- **Fits**: `src/fit.py` mode="half_tie" (identical tie treatment both
  links). BT (logistic) + lattice at units 0.1 / 0.5855 / 0.8002; primary
  board = lattice u = 0.5855. Per-league anchor bot pinned to 0.
- **Uncertainty**: cluster bootstrap B = 2000 over resolution clusters —
  underlying commodity where possible (league 0: 8 clusters), series level
  (question minus numerals) where a league lives inside one commodity
  (league 1: 6 cotton series; those CIs are lower bounds on uncertainty).
- **Headline** restricted to bots with ≥ 5 races.

## Data

97 races (of 446 multi-bot markets; the rest unresolved/cancelled), 11 bots,
214 pairwise comparisons, tie share 0.350. All races are commodity strike
ladders on ~10 underlyings (LBMA gold/silver/platinum/palladium, LME copper,
WTI/Brent/RBOB, cerium, and four cotton series: futures price, open
interest, managed-money net long, producer/merchant net short).

## Headline structural finding: two leagues, not one board

The comparison graph is *connected* but the win digraph is not *strongly*
connected. Ford's (1957) condition fails: only 2 cross-group comparisons
exist and both were won by the same side, so the between-group ability gap
is unbounded in the MLE (the joint fit put it at −6 to −11, pure ridge
artifact — first-run numbers, do not cite). Ranking is only identified
within strongly connected components of the win digraph:

- **League 0 (metals/energy)**: commodity_spot_1, pair_trading_1,
  diffusion_mc_1, news_drift_1, ladder_arb_1, ensemble_1.
- **League 1 (cotton)**: cotton_fundamental_1, commodity_1, cftc_softs_1
  (+ nass_cotton_1, unrated at 2 races).
- mercury_ensemble_1: singleton (1 race, 1 loss) — unrankable.
- Bridge evidence (direction only): ladder_arb_1 beat mercury_ensemble_1
  (cerium) and beat cotton_fundamental_1 (cotton). Insufficient to place
  the leagues on one scale.

## Boards (lattice u=0.5855, anchor = league leader = 0; θ ~ log-odds of
winning the return duel at gap θ via the decisive link)

**League 0 — metals/energy** (bootstrap over 8 underlying clusters):

| rank | bot | races | W-L-T | θ | θ 95% CI | rank 95% CI |
|---|---|---|---|---|---|---|
| 1 | commodity_spot_1 | 47 | 55-9-19 | 0.00 | (−0.33, +0.67) | 1–3 |
| 2 | ladder_arb_1 | 7 | 4-0-6 | −0.15 | (−0.78, +5.95) | 1–4 |
| 3 | ensemble_1 | 6 | 4-0-6 | −0.33 | (−0.93, +0.55) | 1–5 |
| 4 | pair_trading_1 | 47 | 21-40-18 | −0.87 | (−1.33, −0.40) | 2–6 |
| 4 | diffusion_mc_1 | 39 | 22-35-21 | −0.94 | (−1.47, −0.18) | 2–6 |
| 6 | news_drift_1 | 26 | 5-27-22 | −1.26 | (−1.77, −0.78) | 4–6 |

Reliable separations: {commodity_spot_1} > {pair_trading_1, diffusion_mc_1}
> is not resolved between those two; news_drift_1 last among the
well-sampled four (its rank CI excludes 1–3). ladder_arb_1 and ensemble_1
are undefeated but thin (6–7 races) — top-heavy point estimates with rank
CIs spanning nearly the whole league.

**League 1 — cotton** (bootstrap over 6 series; CIs are lower bounds):

| rank | bot | races | W-L-T | θ | θ 95% CI | rank 95% CI |
|---|---|---|---|---|---|---|
| 1 | cotton_fundamental_1 | 32 | 18-4-21 | 0.00 | (−0.16, +0.43) | 1–3 |
| 2 | commodity_1 | 10 | 4-2-14 | −0.16 | (−0.16, −0.04) | 1–2 |
| 3 | cftc_softs_1 | 31 | 3-18-22 | −0.61 | (−1.05, −0.15) | 1–3 |
| — | nass_cotton_1 | 2 | 1-2-1 | −0.72 | unrated | — |

cotton_fundamental_1 > cftc_softs_1 is solid (18-4 head-to-head record
dominates the league); commodity_1 sits between with wide overlap.

## Robustness

Kendall τ = 1.000 for every check in both leagues: BT vs lattice, all three
lattice units, and both tie-band sensitivities (ε = 0.005 / 0.02, tie share
0.20–0.58). Exactly as the paper predicts for an estimation-limited regime
(97 races): the link family does not matter here; the ordering is carried
by the data, not the model.

## Companion table (execution-independent-ish skill, all resolved YES/NO
markets, `results/bot_arena/companion.csv`)

- commodity_spot_1 is the real-money star among well-sampled bots: +1,344
  mana realized on 3,398 invested (+0.40/mana), 92% mana-weighted hit rate,
  Brier 0.079.
- ladder_arb_1 profits differently: 19% hit rate but +0.25/mana — a
  longshot/arb profile (many small losses, occasional large payouts).
- enso_1 has the best calibration in the fleet (Brier 0.004 on 8 markets).
- The whole cotton league loses money in absolute terms
  (cotton_fundamental −0.17/mana, cftc_softs −0.29/mana): league 1 ranks
  the least-bad cotton bettor. Its Briers (0.49–0.60) are worse than coin
  flips — the cotton bots are systematically miscalibrated on their own
  ladders.
- ensemble_1 is 4-0-6 in league 0 races but −0.64/mana overall (Brier
  0.72): it wins its rare shared races while bleeding on solo markets —
  a clean illustration of why the head-to-head board and the PnL table
  answer different questions.
- Small-n curiosities (not rankable): surface_arb_1 +23×/mana on 2 markets;
  llm_ag_coverage −100% on 8.

## Caveats (report with the board, always)

1. **Fixed execution order**: the daily cycle runs bots in a deterministic
   priority order (verified: 100%/0% same-day precedence patterns), so
   earlier bots systematically get better prices on shared signals. The
   board measures "mana banked per mana staked as deployed", which includes
   queue position, not pure forecasting skill. The companion Brier/hit-rate
   columns are the execution-light view.
2. **Selection**: comparisons are conditional on bots choosing the same
   markets; strategies self-select into different market types (hence the
   league split).
3. Races share resolution events within ladders — handled by cluster
   bootstrap, but league 1's series-level clusters still share the cotton
   complex, so its CIs are optimistic.
4. 349 multi-bot markets are still open or cancelled; the board will move
   as they resolve (next refresh: just re-run the script).

---

# Platform-wide leaderboard (script 34, 2026-07-15)

**Script:** `scripts/34_platform_leaderboard.py` · **Outputs:**
`results/platform_leaderboard/` · **Data:** clone v0 API snapshot cached at
`data/raw/clone_api/` (users 280, markets 88,938, bets 6,096,312; consistent
snapshot as of 2026-07-14 ~16:03; fetch verified complete — zero bets exist
before the oldest cached one, platform genesis 2026-02-18 12:44).

Extends script 33 from our quantbots to EVERY account on the clone, same
pipeline: races = resolved YES/NO binary markets with >= 2 traders,
performance = realized return per mana ((payout + sale proceeds) / mana
bought, from raw bets incl. redemptions), pairwise duels, SCC leagues,
half_tie fits (BT + lattice u 0.1/0.5855/0.8002, primary 0.5855), cluster
bootstrap B=1000 over underlying clusters.

## Data

5,795 races · 163 traders · 2,482,549 pairwise duels · tie share 0.116 ·
1,083 underlying clusters. Markets routinely have 30–45 traders (vs 2–4 in
our private arena), so the comparison graph is dense.

## Structure: ONE board (unlike the private arena)

Win digraph SCCs: 4, sizes [159, 2, 1, 1]. The giant SCC covers 159/163
traders and 5,794/5,795 races — platform-wide, a single joint board IS
identified (Ford's condition holds inside it). Headline = 130 traders with
>= 20 races.

## Robustness

Kendall tau on the headline board: 0.995 (lattice vs BT), 0.995–0.997
across lattice units, 0.981–0.985 across tie eps 0.005/0.02 (tie share
0.09–0.16). No longer the exact tau = 1.000 of the 97-race arena — with
2.5M duels the link starts to matter at the margin, exactly the
data-limited -> estimation-limited transition the paper describes — but
agreement remains near-total.

## Headline board (lattice u=0.5855, anchor = MultiLensBot [max races] = 0)

Top of the board (rank / races / W-L-T / theta / rank 95% CI):

| # | trader | races | W-L-T | θ | rank CI |
|---|---|---|---|---|---|
| 1 | KelvinWaveTrader | 57 | 240-77-88 | +0.38 | 1–5 |
| 2 | BiosecurityBot | 56 | 162-72-24 | +0.26 | 1–18 |
| 3 | FastmarketsSpotBot | 30 | 627-374-71 | +0.09 | 1–52 |
| 4 | ChrisManual | 24 | 465-286-62 | +0.02 | 1–38 |
| 5 | MultiLensBot | 4,937 | 79,024-53,325-10,421 | 0.00 | 3–10 |
| 7 | SupplyChainGPT | 4,111 | 69,912-49,620-9,080 | −0.03 | 4–12 |
| 8 | TemporalBot | 4,480 | 71,929-52,387-12,659 | −0.06 | 5–15 |

KelvinWaveTrader is the only trader whose rank CI stays in the top 5.
Ranks 2–4 are thin (24–57 races, wide CIs). **MultiLensBot is the best
high-volume trader** (4,937 races, rank CI 3–10) — among accounts with
1,000+ races the top tier is MultiLensBot > SupplyChainGPT > TemporalBot >
BearCaseBot/CalibrationBot/SkepticalClaude, all separated from the
mid-field by non-overlapping rank CIs.

## Where OUR bots land (sobering)

Platform usernames mapped via ~/Bots/config/bots.yaml account comments:

| platform rank / 130 | account | local bot | races | rank CI |
|---|---|---|---|---|
| 61 | Cottonfundamental1 | cotton_fundamental_1 | 32 | 54–81 |
| 65 | DiffusionMcBot | diffusion_mc_1 | 51 | 15–97 |
| 69 | CommoditySpotBot | commodity_spot_1 | 78 | 16–102 |
| 80 | PairTradingBot | pair_trading_1 | 54 | 62–116 |
| 98 | AviralPoddar (shared acct: enso, commodity_1, surface_arb, llm, mean_rev) | — | 230 | 45–123 |
| 110 | Cftcsofts1 | cftc_softs_1 | 31 | 95–117 |
| 125 | Bot007 | news_drift_1 | 42 | 94–130 |
| 127 | LadderArbBot | ladder_arb_1 | 98 | 108–130 |
| 130 | MikhailTal (personal account) | — | 108 | 122–130 |

Unrated (thin): Nasscotton1 (2 races), MercuryEnsembleBot (4),
EnsembleBot (15, 66-224-36 — its undefeated 4-0-6 private-arena record
does not survive platform-wide competition).

**Takeaways.** (1) Our private-arena champion commodity_spot_1 is mid-table
platform-wide (#69, wide CI): script 33 ranked relative skill among OUR
bots; the platform hosts systematically stronger accounts. (2) The
private-arena ordering is broadly preserved within our fleet (commodity_spot
/ diffusion_mc / pair_trading above cftc_softs / news_drift / ladder_arb),
consistent across both boards. (3) The personal account MikhailTal is
dead last of all 130 rated traders (−0.84/mana on 108 races) — the bots
beat the human. (4) theta ranks per-duel return-rate wins, not PnL:
AviralPoddar has +0.69/mana overall yet sits #98, because rare huge wins
don't win many duels.

## Method note: exact weighted-vote speedup (2026-07-15)

A naive rerun was infeasible: one lattice fit on 2.48M duels took 633 s ->
B=1000 bootstrap ~ a week. Fix: `src/fit.py` now accepts an optional
`weight` column (k identical votes collapse to one row with weight k —
identical likelihood, verified max|dtheta| ~ 1e-4 vs raw on real data;
all 20 repo tests pass). Script 34 aggregates duels (2,482,500 -> 19,027
weighted rows) and runs the cluster bootstrap by weight multiplication
instead of concatenation (verified: identical rank frames vs
mod33.cluster_bootstrap at the same seed, dtheta ~ 2e-6). Full analyze now
~25 min incl. duel building; bootstrap 1000/1000 replicates, 0 failures.

## Caveats

1. Same execution-order caveat as script 33 within our own fleet; unknown
   scheduling for other accounts.
2. Realized-return accounting from raw bets (buys/sells/redemptions);
   unresolved and CANCEL markets excluded as before.
3. league-0/league-1 of script 33 are subsumed: the dense platform graph
   bridges what our private ledger could not.
4. Snapshot is 2026-07-14 16:03. To refresh, rerun a FULL `fetch` then
   `analyze` — `--resume` only backfills OLDER bets after an interrupted
   pull (pagination walks newest -> oldest); it cannot pick up new bets.

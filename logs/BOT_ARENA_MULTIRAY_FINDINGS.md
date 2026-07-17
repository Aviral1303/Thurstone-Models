# Bot arena, multidimensional: MultiRay rating of the quantbots

**Date:** 2026-07-16 · **Script:** `scripts/41_bot_arena_multiray.py` (renumbered from 35 on 2026-07-17; a parallel script 35, extension ceilings, landed on main) ·
**Outputs:** `results/bot_arena_multiray/` · **Input:** script 33's
`results/bot_arena/returns.csv` (97 resolved multi-bot YES/NO races, 11 bots)

## What was run

First application of the vendored package's multidimensional rating system
(`thurstone.multiray.MultiRayGlobalCalibrator`) to the clone-market bot races.
Each bot gets a latent vector Z in R^2; each condition gets a unit ray V and
offset beta; the bot's effective ability in a condition is beta + V·Z, priced
through the lattice race machinery and fitted to observed win frequencies by
alternating least squares.

Data mapping (this run's design, documented in the script docstring):

- **Condition** = (underlying, fixed entrant set) observed in >= 3 races.
  The clone data repeats a few fixed fields per underlying, so conditions are
  literal repeated races: 10 conditions over 7 bots survive the threshold.
- **Prices** = Laplace-smoothed (alpha = 0.5) win frequencies; race winner =
  top realized log-return with script 33's tie rules.
- Bots sharing no condition form separate bipartite blocks, fitted
  independently: **block 0** metals/energy (commodity_spot_1, diffusion_mc_1,
  news_drift_1, pair_trading_1; 8 conditions over gold/palladium/platinum/
  silver) and **block 1** cotton (cftc_softs_1, commodity_1,
  cotton_fundamental_1; 2 conditions). Script 33's league split resurfaces
  structurally: cross-block placement is pure gauge.
- Unplaced (no fixed field with >= 3 races): ensemble_1, ladder_arb_1,
  mercury_ensemble_1, nass_cotton_1.
- Uncertainty: 200 bootstrap refits; block 0 resampled over underlying
  clusters, block 1 over races (single underlying -> CIs are lower bounds).

## Headline boards (consensus = race-weighted, within-condition-centered
ability, sign-flipped to higher-is-better; per-block gauge)

Block 0 — metals/energy:

| rank (median [95% CI]) | bot | consensus skill |
|---|---|---|
| 1 [1, 1] | commodity_spot_1 | +0.469 |
| 2 [2, 4] | diffusion_mc_1 | −0.118 |
| 3 [2, 4] | news_drift_1 | −0.264 |
| 3 [2, 4] | pair_trading_1 | −0.321 |

Block 1 — cotton (CIs are lower bounds):

| rank (median [95% CI]) | bot | consensus skill |
|---|---|---|
| 1 [1, 2] | cotton_fundamental_1 | +0.395 |
| 2 [1, 2] | commodity_1 | +0.202 |
| 3 [3, 3] | cftc_softs_1 | −0.465 |

Agreement with the script 33 pairwise board: **block 1 tau = 1.000** (same
order); **block 0 tau = 0.333** — same #1 (commodity_spot_1, bootstrap rank
CI [1, 1] in both), tail order permuted, but the tail is unresolved in both
methods (all three tail bots share overlapping [2, 4] rank CIs here).
Sensitivities: smoothing alpha 0.25/0.5/1.0 tau = 1.000 both blocks; dim=1
vs dim=2 tau = 0.667 / 1.000; min_races 3 vs 2 tau = 0.667 / 1.000.

## Findings about the system itself (as important as the board)

1. **Beta rides a translation ridge.** Race state prices depend only on
   relative abilities within a condition, so the per-condition offset beta is
   unidentified; on the first run one seed drifted to |beta| ~ 1e13 with no
   fit penalty. Any consumer of raw abilities must center within condition
   first.
2. **The alternating optimizer is not monotone and is a seed lottery.** With
   the package's fit loop as shipped, price MSE on block 0 went 2.0e-4 ->
   1.8e-2 between outer iteration 5 and 30 for one seed, while another seed
   improved. Even with best-iterate snapshotting and early stopping (added in
   the harness, model untouched), seeds reached MSE {7.9e-5, 3.6e-3, 2.2e-1}
   on block 0 — one good optimum in three starts. Multi-seed + best-MSE
   selection is mandatory.
3. **The embedding geometry is not identified at this data size.** Block 0
   has ~12 free price observations against ~16+ effective parameters at
   dim=2, so (Z, V) sit on a solution manifold. The naive block-mean-ray
   projection −(V̄·Z) was seed-dependent (tau −0.33..0.33 across seeds) and is
   reported as a diagnostic column only. The consensus above uses only
   within-condition-centered abilities, which the observed prices pin down;
   it is seed-stable wherever the fit converges (block 1: tau = 1.000 across
   all seeds; block 0: tau = 0.667 vs the second-best seed, whose residual
   misfit is large enough to flip the near-tied tail).
4. **Winner-take-all prices discard information the pairwise view keeps.** A
   race's price vector only records who finished first; second-vs-third in a
   3+ bot race never enters the likelihood, and 2-race fields (wti, rbob,
   brent, copper) fall below the condition threshold entirely. Script 33's
   pairwise decomposition uses all of that. This, plus the threshold, is the
   likely source of the block 0 tail permutation.

## Bottom line

On this dataset the multidimensional system reproduces the pairwise
Thurstone/BT conclusions where the data are dense (cotton: identical order;
metals: same clear #1, tail unresolved in both) and adds nothing resolvable
beyond them: with 4 bots per block and 8-10 conditions there is not enough
data to identify a genuine second skill dimension, and the extra freedom
shows up as optimizer pathology rather than insight. The per-condition
ability decomposition (see `abilities.csv`) is the one genuinely new view:
e.g. commodity_spot_1's edge is much larger in 2-bot palladium/silver fields
(win prob ~0.65-0.83) than in the 4-bot silver field (~0.21, below
diffusion_mc_1 and pair_trading_1 there).

Gauge/sign notes for reuse: package is race-time (lower ability = better);
the flip to higher-is-better happens exactly once, in
`identified_skill` / the `skill` column. Lattice L=150 (unit 0.1) verified
identical to the example's L=500 to 1.7e-13 and 4.4x faster.

Refresh: re-run script 33 first (it writes `returns.csv`), then script 41 (bot-arena multiray).

---

## Addendum 2026-07-16: all-bots board (min_races = 1)

Second run per user request ("do it for all of the bots"): `python
scripts/41_bot_arena_multiray.py 1` → `results/bot_arena_multiray_min1/`.
Admitting single-race fields gives 26 conditions and makes the whole 11-bot
field ONE connected block — the cotton↔metals gauge flows through single-race
bridges (ladder_arb_1 vs cotton_fundamental_1, one cotton ladder; ladder_arb_1
vs mercury_ensemble_1, one cerium race; thin copper/gold/rbob/platinum fields
place ensemble_1 and ladder_arb_1 vs league 0). `bridge_conditions.csv` lists
them.

Board (consensus skill; median rank [95% bootstrap rank CI], 200 replicates
over 10 underlying clusters): commodity_spot_1 +0.500, 1 [1–6];
ladder_arb_1 +0.366, 2.5 [1–8]; cotton_fundamental_1 +0.356, 3 [2–7];
ensemble_1 +0.291, 3 [1–6]; commodity_1 +0.203, 5 [2–5]; nass_cotton_1
−0.108, 6 [4–7]; diffusion_mc_1 −0.128, 6 [3–9]; news_drift_1 −0.224, 7
[2–9]; pair_trading_1 −0.341, 8 [3–10]; cftc_softs_1 −0.442, 10 [3–10];
mercury_ensemble_1 −0.483, 10 [6–11].

Stability: identified summary seed τ 0.85–1.00 (much better than min=3
metals — 26 conditions constrain the fit harder); α τ 0.93–0.96; τ = 1.000
vs the dense min=3 board on the 4 shared bots; vs script 33 within league 0
τ = 0.733 (6 bots), league 1 τ = 0.667 — all disagreements inside
overlapping rank CIs. Best seed 7, price MSE 6.2e-4.

⚠ Bootstrap SKILL quantiles are unusable in this run: a minority of
replicate fits diverge (identified skills reach ±1e9 — centered abilities
explode when a replicate's resampled prices are degenerate), inflating
skill_lo95/hi95 to ±1e9 while rank quantiles stay sane. Report rank CIs
only. Cotton-side bots have boot coverage ≈ 0.64 (cotton is a single
resampling cluster, absent in ~36% of replicates); mercury 0.63.

Webpage updated: `results/bot_arena/bot_arena_board.html` gained section
"06 · The multidimensional view" (all-bots rank-CI chart + table + bridge
and identification callouts), later sections renumbered, footer covers
script 41; republished to the "Bot Arena — Quantbots Head-to-Head"
artifact.

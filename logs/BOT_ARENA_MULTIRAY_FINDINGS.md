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

---

## Platform-wide multiray (script 42), 2026-07-20: diagnosis, fix, and board

**Script:** `scripts/42_platform_multiray.py` · **Outputs:**
`results/platform_multiray/` (min=3) and `results/platform_multiray_min1/`
(all traders) · **Input:** script 34's `returns_all.csv` (5,795 platform
races, 163 traders).

### The 2026-07-17 degenerate run and its real root cause

The first full run (archived at
`results/platform_multiray_degenerate_20260717/`; NEVER cite as a board)
saturated: every condition ended with |centered ability| 1e7–6.8e10 against
a ±7.5 lattice, consensus skills 1e9–1e10, seed τ ≈ 0.30. The hypothesis
recorded at the time — huge fields put ~90% of entrants on one Laplace-floor
price, leaving the geometry underdetermined, fixable by a field-size cap —
was tested and is WRONG as stated: caps 8/12/15/20 still saturate (probe
2026-07-20; it is a per-seed lottery — cap 15 seed 999 stayed healthy while
seed 7 blew up at the same data).

The actual root cause is numerical, in the package's `fit_inner`: the
Gauss–Newton step target is y = −err/slope with the slope clamped at
`slope_floor = 1e-10`. A floor-priced entrant in a 35–48-item field sits in
the FLAT TAIL of its condition's price curve (slope ≈ 0), so its clamped
step target is ~err×1e10; one such item pollutes the shared least-squares
updates of (β, V) for every condition containing it and the whole fit
cascades. Best-iterate snapshotting cannot save a fit that explodes inside
the first inner iteration.

### The fix (three parts, all outside the vendored model)

1. **Trust region** — `slope_floor` is an exposed constructor parameter;
   we set 0.05 (plumbed as `m41.SLOPE_FLOOR`, default None keeps the old
   behavior for the bot-arena scripts). Bounds |step target| ≤ |err|/0.05
   while leaving responsive-region slopes (~0.1–0.3) untouched. Saturation
   gone everywhere: uncapped MSE 1.9e-2 → ~8e-4, abilities O(1)
   (max |centered ability| over all 22 production fits: 1.74).
2. **Slope-weighted identified summary** — post-fix, seeds still land on an
   MSE plateau (0.8–1.2e-3) carrying DIFFERENT rankings (pairwise τ ≈ 0.40;
   neither longer inner loops nor dim=1 helps; restricting to well-placed
   traders makes it WORSE, so it is not small-sample noise). Cause: ~16% of
   (condition, entrant) cells are flat-tail, where the observed price pins
   the ability only to a half-line — the fitted point value is
   optimizer-arbitrary, and the plain summary averages it in.
   `identified_skill_slopewt` weights each cell by n_races × |local price
   slope|, excluding exactly the unpinned cells: pairwise seed τ 0.40 →
   0.58.
3. **Seed-ensemble consensus** — the board is the mean slope-weighted
   summary over 10 seeds. Certified before the production run: two DISJOINT
   10-seed ensembles agree at τ = 0.821 (rated ≥20 races: 0.811), vs
   0.15–0.48 for single fits. Per-seed summaries persisted
   (`seed_summaries.csv`).

### Calibration of the external check

The naive smoothed-win-rate baseline reaches only τ ≈ 0.413 against script
34's pairwise board — race-price aggregation (winner-take-all, fixed
fields) and 2.5M pairwise duels measure genuinely different things on this
platform. τ ≈ 0.41 is therefore the ceiling for ANY race-price model here,
not a defect. The multiray consensus hits exactly that ceiling: τ = 0.412
(all placed), 0.385 (rated).

### Board (min_races = 3): 494 conditions, 71 traders, ONE block

Fit: 10-seed ensemble, per-seed MSE 7.6e-4–1.2e-3, best seed 404.
Bootstrap: 200/200 cluster replicates (1,076 clusters), 0 failures, each
replicate a 3-seed ensemble. In-run stability: half-ensemble (5v5) τ 0.744;
α=0.25/1.0 τ 0.858/0.821; dim=1 τ 0.657; min_races=5 τ 0.426 (thinning).

Top of the board (rank = bootstrap median [95% CI]):
KelvinWaveTrader 2 [1–4] — the same #1 as script 34's independent pairwise
board; BiosecurityBot 1 [1–12.7]; PrivateFundingBot 3 [1–70] (1 condition,
19 races — the wide CI is honest); SurfaceBrowser 4 [1–69];
WhiteGoldTrader 5 [3–69]. Best high-volume traders: MomentumBot 11 [6–25]
(398 conditions / 4,453 races), BearCaseBot 14 [8–26], MultiLensBot 15
[9–28] (4,937 races; CI 3–10 on script 34 — consistent mid-top placement).

Our fleet (platform usernames per `~/Bots/config/bots.yaml`): AviralPoddar
(shared-key account) #26, LadderArbBot #63 point but bootstrap median 10
[3–66] (2 conditions only — point rank unstable, CI is the story),
PairTradingBot #66 [31.7–69.5], CommoditySpotBot #67 [59.8–70] of 71 —
consistent with script 34's mid-table-to-bottom fleet placement.
92 traders have no fixed field with ≥3 races and are unplaced here (the
min=1 run places them).

### Bottom line

The multidimensional system CAN rate the whole platform once its optimizer
is trust-regioned and the summary is restricted to price-pinned cells, and
its board agrees with the pairwise leaderboard exactly as much as the data
allows any race-price model to (τ at the naive ceiling). What it adds over
script 34 is the per-condition ability decomposition and the embedding; what
it cannot do at this sparsity is beat the pairwise board's precision (rank
CIs here are much wider than script 34's).

### Addendum: all-traders board (min_races = 1), same day

`python scripts/42_platform_multiray.py 1` → `results/platform_multiray_min1/`.
Admitting single-race fields places ALL 163 traders in ONE block over 3,897
conditions. Fit: 10-seed ensemble, MSE 5.8–9.4e-4, no saturation (max
|centered ability| 1.31), 20 min wall. No bootstrap by design; stability
evidence: half-ensemble τ 0.723 (rated 0.654); consensus vs the min=3 board
τ 0.463 on shared traders (single-race winner-take-all conditions genuinely
shift thin traders); vs script 34 τ 0.177/0.230 (all/rated) against a naive
ceiling of 0.138 here — the consensus extracts MORE pairwise-consistent
signal than the raw win-rate at this sparsity.

Read the min=1 board jointly with n_races and consensus_seed_std: the top is
dominated by thin accounts whose seed_std flags them (TermStructureBot #1 at
3 races, seed_std 0.23 — noise; OliviaDAngelo #2 at 2 races). Credible
high-data placements: BiosecurityBot #3 (56 races), KelvinWaveTrader #4
(57), CottonTextBot #8 (69). Fleet: EnsembleBot #6 (15 races),
LadderArbBot #12 (98), AviralPoddar #14 (230), MikhailTal #77 of 163
(108 races — dead last on script 34's per-mana pairwise board but midfield
on race wins: loses money per mana while still occasionally topping a
field), DiffusionMcBot #134, PairTradingBot #138, CommoditySpotBot #143,
Bot007 #150, Cottonfundamental1 #153, Cftcsofts1 #156, Nasscotton1 #159.

### Addendum 2026-07-20: dimensionality test — the 2nd dimension is not real

User question: can the two dimensions be classified? Post-hoc geometry
(best seed) suggested axis 1 ~ general skill (|rho| 0.46 with the board) and
axis 2 ~ a faint softs-vs-precious contrast (rho +-0.24-0.27), but rays are
near-isotropic (eigen split 55/45; mean |cos| 0.68 vs 0.64 for random) and
the orientation is seed-unstable, so a pre-registered gate was run first.

**Held-out dimensionality test** (scratchpad step1_dim_test.py; 5 random
half-splits of races within each >=4-race condition, 211 scored conditions /
484 test races per split; dims 1/2/3 fitted on train with the production
pipeline, 3-seed ensemble predictions; race-weighted MSE vs raw held-out win
frequencies): carry-forward of the train frequencies wins on ALL 5 splits
(12.95e-3) over dim3 (13.59) < dim2 (13.65) < dim1 (13.78) < uniform
(13.85). Higher dim improves prediction monotonically (dim2 < dim1 on 5/5
splits, dim3 < dim2 on 5/5) but only ever TOWARD the carry-forward ceiling,
never past it — the signature of per-condition memorization (each dim adds
ray parameters), not shared skill structure. The latent model at every
dimension predicts held-out races WORSE than the raw training frequencies:
cross-trader pooling adds no out-of-sample value at this sparsity.

**Verdict: no evidence for a genuine 2nd dimension.** The axis labels above
are unverified visual suggestions; the dimensions are unidentified surplus
capacity. Identification would need experimentally generated bridging races
(same entrant sets repeated across contrasting market families) — the fleet
can produce these; organic platform data does not contain them.

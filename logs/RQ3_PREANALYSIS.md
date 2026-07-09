# RQ3 Pre-Analysis Commitment — Held-out calibration, BT vs lattice-Thurstone

Status: PRE-COMMITTED 2026-07-11, before any real-data calibration number
has been computed. Thresholds, criteria, and interpretation rules below are
fixed; deviations at analysis time must be flagged loudly in the research
log. Synthetic validation of the full pipeline (section 6) PASSED before
this document was finalized. No real-data fitting has occurred for RQ3.

RQ1's null creates no expectation here in either direction: a link can
produce identical refit stability while differing in held-out calibration.
(That said, section 6's discoveries DO bound what is theoretically
achievable — stated there, derived from synthetic/analytic work only.)

**Pre-committed expectation (recorded before any real fitting, mirroring
RQ1_FINDINGS.md's scope paragraph):** grounded in the real gap distribution
and real estimation noise (§6.1–6.2), the a-priori expectation is that
full-population RQ3 will show practical equivalence; if a genuine effect
exists anywhere in this dataset, the analysis indicates it would most
plausibly appear in the recent-entrant stratum of the early (2023)
windows — and would only reach practical significance under a generating
process more skewed than a typical lattice link. This expectation is
recorded before real fitting so the eventual stratified framing cannot be
read as chosen post hoc.

## 1. Split design

**Rolling-origin, reusing RQ1's checkpoint grid.** Train on all dedup
battles with tstamp ≤ T_k; test on battles in (T_k, T_{k+1}]; T_k = the 14
RQ1 checkpoints (month-ends 2023-07-31 → 2024-07-31 + 2024-08-12 cutoff)
→ **13 test windows**. Reasoning: (i) matches the deployed refit cadence and
RQ1's structure (results directly comparable); (ii) test windows are
time-disjoint — unlike RQ1's δ=3 overlap there is no test-set overlap —
though training sets are nested, so window results are still not fully
independent (see §3); (iii) a single large split would give one number, no
temporal structure, and no recent-entrant dynamics.

**Fits**: identical code path to RQ1 (per-vote MLE, half-tie mode,
include_both_bad=True — the fastchat-faithful treatment, regression-tested),
identical train data for both methods, anchor irrelevant (predictions depend
only on gaps).

**Outcome space & tie treatment (structural decision)**: RQ3 scores
**conditional-on-decisive** two-outcome probabilities p = F_decisive(θ_A −
θ_B), on **decisive test votes only**. Ties (both kinds) are excluded from
scoring and their counts reported per window. Reasoning: BT-logistic has no
native tie probability — a trinomial contest would require Davidson-BT,
which is RQ4's fitted-vs-fitted design; forcing it here would entangle the
two RQs. RQ3's claim is therefore explicitly narrowed to: *calibration of
decisive-outcome probabilities*. Tie-prediction calibration is RQ4.

**Unscoreable votes**: test votes involving a model with zero training
votes are dropped and counted per window (no ability estimate exists; a
deployed system would face the same cold-start gap — the counts themselves
are reported as an operational fact).

**Recent-entrant definition (fixed now)**: a test vote is in the "recent"
stratum iff at least one of its two models has its first training-data
battle within **28 days** before T_k. Complement = "established". 28 days
aligns with the checkpoint grid; a vote-count-based definition was rejected
because it confounds recency with popularity. This definition is final —
it will not be revised after seeing which definition produces effects.

## 2. Unit / scale convention

The Elo-style rescaling used for RQ1's magnitude metrics is irrelevant here
— raw predicted probabilities are scored, and they are invariant to affine
display transforms. **But the lattice unit is NOT irrelevant for RQ3**,
unlike for RQ1's rank metrics: the unit changes the decisive link's shape
(measured slopes at gap 0: 0.290 / 0.324 / 0.337 for units 0.1 / 0.5855 /
0.8), hence the predicted probabilities. Different units are different
probability models.

Pre-committed evaluation set, same discipline as RQ1:
- **Primary: unit = 0.5855** (profiled on first-3-months data only, which
  predates every test window — look-ahead-free AND empirically grounded).
- Sensitivity: unit = 0.1 (arbitrary default) and unit = 0.8 (full-sample
  fit; carries look-ahead; retained as the band's upper bound).
- BT has no structural counterpart parameter; logistic fixed.

**Pre-committed interpretation caveat (from section 6's discovery, decided
before real data):** steeper links produce more extreme predictions and are
punished harder by held-out log-loss when ability estimates are noisy
(plugin-MLE overconfidence). The three units differ in steepness, so if the
real-data unit ordering comes out flattest-best (0.1 > 0.5855 > 0.8), that
pattern indicates plugin-noise dominance, NOT evidence about the true
tie-band width — and any lattice-vs-BT direction must be interpreted with
the same confound in mind (BT's effective link is the flattest of all four
models after MLE). To support attribution we additionally report, per
method × window, a reliability (calibration-slope) diagnostic on test
predictions — computed on the same test windows, hence diagnostic, not
confirmatory.

## 3. Effect-size threshold and statistical comparison

**Minimum practical difference (MPD), derived**: we anchor practical
significance to a **10-Elo-point systematic error** — the granularity at
which leaderboard consumers actually read ratings, and the scale RQ1 used
for magnitude context. A uniform 10-pt error in every matchup gap is
ε = 10·ln(10)/400 = 0.05756 nats. Its per-vote expected excess log-loss at
a toss-up (worst case, hence conservative) is
½·p(1−p)·ε² = ½·0.25·0.05756² = 4.14e-4 nats.
**MPD_logloss = 4e-4 nats/vote** (rounded). Differences below MPD are
declared *practically equivalent regardless of statistical significance* —
at ~1M scoreable test votes, deltas far below MPD will be "significant"; we
refuse to promote them. Brier analog: Δp = 0.25·ε = 1.44pp → **MPD_brier =
2e-4** (secondary metric, same rules).

**Statistical comparison**: per-vote paired delta d = ll_BT − ll_lattice
(positive = lattice better), aggregated to per-window means (the honest
units). Pooled estimate = vote-weighted mean of window means; CI =
**window-cluster bootstrap** (resample the 13 windows with replacement,
B=10,000, percentile CI). **Effective N = 13 windows, not ~1M votes** —
stated wherever a CI appears. Residual caveat (flagged now, mirroring RQ1's
δ=3 honesty): training sets are nested across windows, so even the 13
window means are not fully independent; the cluster bootstrap treats them
as exchangeable, which is the best available approximation and is labeled
as such. Sign consistency threshold: **≥10 of 13 windows** (one-sided
binomial p = 0.046 under a fair coin).

## 4. Pattern-based verdict criteria (fixed before outcomes exist)

Implemented verbatim in `src/rq3_eval.py::classify` (unit-tested on all
branches). Let Δ = pooled delta, CI its cluster-bootstrap interval, W⁺/W⁻
the windows where lattice/BT is better.

- **(a) Genuine equivalence**: |Δ| < MPD AND |median window delta| < MPD
  AND CI ⊂ (−MPD, +MPD). If additionally CI excludes 0, the verdict remains
  equivalence but is reported with an explicit **"sub-practical directional
  lean"** note — real but below deployment relevance (both synthetic
  worlds landed here, with the correct directions).
- **(b1) Genuine positive — lattice**: Δ ≥ MPD AND W⁺ ≥ 10/13 AND CI low
  > 0. Symmetrically **(b1′) positive — BT**: Δ ≤ −MPD AND W⁻ ≥ 10/13 AND
  CI high < 0. (Both directions pre-committed; the classifier is
  sign-symmetric.)
- **(b2) Mechanism grade, only on top of (b1)**: the recent-entrant-stratum
  pooled delta ≥ the full-population delta. Only (b1)+(b2) may be described
  as consistent with the field-change/extrapolation hypothesis. (b1)
  without (b2) is reported as "overall calibration edge, mechanism
  unattributed" — and per §2's caveat, link-steepness/plugin-noise must be
  discussed as a candidate explanation before generative correctness.
- **(c) Inconclusive**: CI covers ±MPD boundaries in a way that neither
  confirms equivalence nor a positive (CI straddles a band edge), OR effect
  size and sign consistency disagree (e.g., pooled ≥ MPD driven by a few
  giant windows without 10/13 consistency — labeled heterogeneous). If (c),
  we report it as inconclusive, state what data would resolve it, and do
  NOT force a call.
- **Robustness requirement**: the verdict must agree at the primary unit
  and at least one sensitivity unit to be quoted without qualification; if
  the three units land in different categories, the headline is the primary
  unit's verdict with the disagreement stated in the same sentence.

A priori expectation from section 6 (recorded so post-hoc rationalization
is impossible): **(a) is the strongly expected outcome; (b1)-lattice at
≥MPD is close to theoretically unachievable within the model family at this
data scale.** If (b1)-lattice nevertheless occurs, it is surprising and
demands the §4.1 confound procedure before any generative claim.

### 4.1 Pre-specified steepness-confound analysis (fixed before any real
number could trigger it)

If either positive verdict (b1 or b1′) fires on real data, the following
procedure runs BEFORE the verdict is written up, and its outcome wording is
fixed here:

1. **Uncertainty-integrated re-prediction.** For every fitted method,
   replace the plugin prediction p = F(ĝ) with the posterior-predictive-
   style p̃ = E_ε[F(ĝ + ε)], ε ~ N(0, SE_A² + SE_B²), where SE are the
   **bootstrap-calibrated** per-model standard errors from the SAME
   TRAINING fit that produced ĝ: Fisher SEs divided by the measured
   Fisher-to-bootstrap ratio 0.80 (validated cross-check vs the published
   bootstrap SDs, Spearman 0.983; machinery
   `src/rq3_eval.py::fisher_se_calibrated`). Raw uncalibrated Fisher SEs
   are run as a labeled sensitivity variant — calibration settled here,
   before any real result could trigger this procedure. The integral is
   evaluated numerically on the link's gap grid. This shrinks
   overconfident predictions more for steeper links and noisier models —
   exactly the mechanism World P2 exposed. Both methods get the identical
   correction; the entire pipeline (window table, pooled cluster bootstrap,
   classifier) is re-run on the corrected predictions.
2. **Fixed interpretation rule.**
   - If the positive effect **drops below MPD (or its CI no longer excludes
     the band)** under correction: the verdict is reported as
     **"steepness/plugin-noise-driven; practically equivalent after
     uncertainty correction"** — the plugin result is still shown, but the
     corrected result governs the headline, and no generative-link claim is
     made.
   - If the positive effect **survives correction** (still ≥ MPD, CI
     excluding the band on the same side, ≥10/13 consistency): reported as
     **"survives uncertainty correction — genuine link-shape effect"**, and
     only then may (b2) mechanism language be attached (if its criterion
     also holds).
   - Partial shrinkage that leaves the verdict at the band edge is reported
     as inconclusive-after-correction; no call is forced.
3. The corrected-prediction comparison is symmetric and will also be run if
   the verdict is equivalence-with-directional-lean, as supporting detail
   (labeled diagnostic, not confirmatory — it reuses the test windows).

### 4.2 Reporting rule: significance vs effect size (fixed now)

A statistically significant sign-consistency result (e.g., ≥10/13 windows,
or a pooled CI excluding zero) **below the MPD effect-size threshold** will
be reported as **"practically equivalent despite a detectable directional
lean"** — never as "BT wins" or "lattice wins". Significance and effect
size are always reported together, and **effect size is the governing
word**, exactly the discipline applied to RQ1's rank-metric null. This rule
is written before any real p-value exists.

## 5. Stratification reporting

- Recent-entrant stratum reported ALONGSIDE full population, always both,
  for every table — RQ1's headline+subgroup pattern. Established-stratum
  complement included for contrast.
- **Minimum stratum size for CI-bearing claims: ≥500 scoreable decisive
  votes in that stratum-window**; below that, the stratum-window is
  descriptive only. Pooled-across-windows stratum inference (cluster
  bootstrap over windows with nonzero stratum N) is the primary stratum
  read; per-window stratum numbers are supporting detail. (The synthetic
  worlds produced tiny recent strata — real Arena oversamples new models,
  so real strata will be far larger; if they are not, the b2 grade may be
  unpowerable and will be reported as such rather than forced.)

## 6. Synthetic validation — PASSED (scripts/10, 11; tables in results/)

All worlds run through the real pipeline code and real thresholds.
Structure mirrors the experiment: 14 monthly checkpoints, staggered model
entry, cumulative training, next-month testing. Deltas reported as
(true expected value | realized pooled | verdict).

| world | truth | lattice fit | true Δ (×MPD) | realized Δ (×MPD) | verdict | gate |
|---|---|---|---|---|---|---|
| P | lattice u0.8 | matched u0.8 | +0.13 | +0.27, CI incl. 0 | equivalence | no positive call ✓ |
| P′ | lattice u0.8 | u0.5855 | +0.52 | +0.62, CI > 0 | equivalence + lattice lean | direction detected, not promoted ✓ |
| P2 | lattice u1.2 skew6 | matched | **−1.41** | −1.39, CI < −MPD | bt_positive | ≥MPD effect, correct sign ✓ |
| N | logistic | u0.1 | −0.30 | −0.33, CI < 0 | equivalence + bt lean | no false lattice call ✓ |

Three pre-analysis discoveries that now constrain interpretation, all made
WITHOUT real outcome data (§6.1 grounds the first two in real inputs):

1. **In-family effect ceiling** (scripts/11, analytic): the largest
   population-level calibration advantage any plausible lattice link
   (units 0.1–1.2, skew 0–8) can have over a best-fit logistic — initially
   computed over illustrative Gaussian gap spreads as ≈ **1.56×MPD** —
   collapses to **≤ 0.23×MPD** on the real gap distribution (§6.1). The
   hypothesis space cannot produce practically-significant pooled
   calibration effects here.
2. **Plugin-noise overconfidence** (World P2): in a lattice-truth world
   with high ability-estimation noise, the CORRECTLY-SPECIFIED steep
   lattice loses to misspecified BT held-out (−1.41×MPD): plugin
   predictions from steeper links are overconfident under parameter noise.
   §6.1 shows this reversal does NOT operate at real pooled noise levels
   (measured SEs are ~5-10× smaller than the synthetic world's) — but the
   mechanism remains live for the recent-entrant stratum, whose per-model
   noise resembles the synthetic regime.
3. **Statistical sensitivity validated**: the pipeline correctly detected
   direction with CI excluding zero at true effects of 0.3×MPD (N world)
   and 0.5×MPD (P′), correctly declined to promote them past MPD,
   correctly fired the positive branch on a ≥MPD effect with the right
   sign (P2), and made no false calls. The lattice_positive branch —
   unreachable end-to-end at realistic scales — is the exact sign-mirror
   of P2's branch and is unit-tested directly
   (tests/test_rq3_classifier.py, all four verdicts).

## 6.1 Real-input grounding (added at review, 2026-07-11 — scripts/12)

The user correctly flagged that §6's ceiling and reversal rested on
illustrative inputs. Both were recomputed from quantities already measured
in this project (no held-out real calibration number was computed; all fits
used are training-style fits, and all expectations are under hypothetical
lattice-truth links):

**(a) Ceiling over the EMPIRICAL gap distribution.** The illustrative sweep
used Gaussian gap spreads σ_g ∈ {2.1, 3, 4}; the real vote-weighted gap
distribution (from the completed full-population fits over all 1.67M dedup
battles) is far narrower: median |gap| 0.31, p95 0.94, max 2.23 — real
matchups concentrate near toss-ups, where all links agree. Recomputed
ceiling (results/tables/rq3_ceiling_empirical.csv): **plausible symmetric
units 0.005–0.08×MPD; family maximum 0.23×MPD** (u0.8/skew4). Conclusion
direction unchanged, magnitude strengthened ~7×: no in-family world can
produce a pooled effect anywhere near MPD on real matchup gaps.

**(b) Reversal at REAL noise levels.** Per-model Fisher SEs computed from
real training fits at three regimes (early 2023-08, mid 2023-12, full
2024-08), both methods; machinery validated against the published
bootstrap SDs (Spearman 0.983, median ratio 0.80 — Fisher mildly below
bootstrap, as expected). Real SEs: median 1.6–4.5 Elo-equivalent points.
Monte-Carlo expected held-out delta under lattice-truth with real
noise (results/tables/rq3_reversal_real_inputs.csv):

| regime | truth u0.5855 | truth u1.2/skew6 (stress bound) |
|---|---|---|
| early 2023-08 | +0.27×MPD | +1.14×MPD |
| mid 2023-12 | +0.14×MPD | +0.44×MPD |
| full 2024-08 | +0.05×MPD | +0.12×MPD |

These rows are **era-level**: the MC used every model's own SE but averaged
the delta over pooled era votes (see §6.2 for the recent-entrant-stratum
version, added at review).

**Truth-scenario labels**: u0.5855 is the realistic scenario (the unit
actually profiled from early real data). **u1.2/skew6 is a stress-test
upper bound, not a realistic scenario**: skew_a=6 implies performance-noise
skewness ≈0.89, near the skew-normal family's maximum (≈0.995), and it was
selected as the family-maximum of the analytic sweep — nothing in the
observed data motivates extreme skew. The single cells that clear MPD under
this truth deserve correspondingly little weight.

**The P2 reversal does not hold at real pooled noise** (real SEs are ~5-10×
smaller than the synthetic world's): under lattice-truth the true model
keeps a small positive advantage in every regime — indeed slightly amplified
by noise, because the steeper link extracts more information per vote
(smaller SEs) and BT's best-fit scale <1 amplifies its own noise.
**Conclusion change, stated honestly:** the earlier claim "the mechanism
'lattice is the truer link' cannot produce a positive at this data scale
because plugin noise reverses it" was an artifact of illustrative noise;
the corrected statement is: *the achievable pooled advantage under
lattice-truth at real gaps and real noise is +0.05 to +0.27×MPD (realistic
truth) — still far below MPD, so the a-priori expected verdict remains
equivalence, but by the effect-ceiling route, not the reversal route.* The
overconfidence mechanism itself remains real and quantified — it bites at
high per-model noise, i.e., the recent-entrant stratum, where §4.1's
correction procedure applies with full force.

## 6.2 Recent-entrant-stratum noise and achievable effect (added at review)

The §6.1 table is era-level; the review asked for the analysis on the
stratum's OWN noise. Recomputed (scripts/13) with bootstrap-calibrated SEs,
restricting to votes involving ≥1 model that entered within 28 days of the
checkpoint (stratum composition taken from the last 28 training days —
strictly pre-test; no test-window data used):

| regime | recent models | stratum votes (28d) | cohort med SE (Elo) vs era | truth u0.5855 | truth u1.2/skew6 (stress) |
|---|---|---|---|---|---|
| early 2023-08 | 3 | 1,439 | 17.7 vs 5.6 | **+1.58×MPD** | +2.38×MPD |
| mid 2023-12 | 5 | 22,643 | 4.9 vs 4.4 | +0.24×MPD | +0.34×MPD |
| full 2024-08 | 13 | 116,311 | 3.1 vs 2.7 | +0.11×MPD | +0.17×MPD |

Reading, pre-committed: (i) recent-entrant cohorts were genuinely noisy
only in the EARLY era (Arena's oversampling of entrants matured by 2024 —
cohort SEs converge to era levels); (ii) under lattice-truth the stratum
effect clears MPD only in early-2023 windows — where the stratum is also
small (~1.4k votes/28d → per-window stratum N near the §5 minimum-N rule,
so per-window power will be marginal and the pooled-stratum read carries
the weight); (iii) even at 17.7-Elo cohort noise, calibrated real noise
never reproduces the P2 reversal — under lattice-truth the direction stays
lattice-positive, amplified by noise. Net: the only place a practically
significant real effect is even available under in-family truth is the
early-window recent-entrant stratum, and only toward the stress bound.

## Note for the paper (flagged at review; no action now)

The plugin-overconfidence mechanism — steeper links extracting more
information per vote yet producing systematically overconfident
point-estimate predictions under ability-estimation noise, with a
measurable crossover as noise grows (World P2: reversal at synthetic noise;
§6.1: no reversal at real pooled noise; open at recent-entrant noise) — is
a general, citable methodological finding independent of RQ3's eventual
real-data outcome. It applies to any ranking / reward-model deployment
using a steeper-than-necessary link without uncertainty correction, not
just Thurstone-vs-BT. Candidate for its own discussion-section
contribution.

Machinery note: the steep-skewed P2 link required a finer curve grid
(g_step 0.005) — coarse-grid interpolation kinks stalled L-BFGS above the
gradient-acceptance threshold; fit_gaplink also gained a restart-from-stall
remedy. Real-data links (units ≤ 0.8, no skew) never hit this, but the
default remedy stays in place.

## Execution plan after approval (unchanged machinery, no new decisions)

13 windows × 4 fits (BT + 3 units), reusing the RQ1 fit code path; score,
stratify, classify, report — with the verdicts and wording rules above
fixed in advance. Estimated compute: under an hour. Outputs:
`results/tables/rq3_window_table*.csv`, `rq3_pooled.csv`, figures, and a
findings doc following the RQ1_FINDINGS.md pattern.

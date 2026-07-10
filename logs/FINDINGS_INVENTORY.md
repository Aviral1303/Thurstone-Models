# Findings Inventory — factual compilation across all RQs

Compiled 2026-07-10 at synthesis kickoff. This is a COMPILATION with exact
numbers and source references, not a narrative. Sign conventions and MPDs
are stated per block; every number traces to a findings doc, the research
log, or a results table in the repo (commit 63fcc3d).

Dataset: LMSYS clean_battle_20240814 (1,799,991 anony battles, 129 models,
2023-04-24→2024-08-14; dedup_sampled subset 1,670,250 used for all fits).
Pipeline validated against 10 published BT-era leaderboard snapshots at
MAE 0.18–1.01 Elo pts (rq1_validation_published.csv); dedup switchover
empirically dated May–Jun 2024. All experiments pre-registered
(RQ1_SPEC.md, RQ3_PREANALYSIS.md, RQ4_PREANALYSIS.md, RQ2_DESIGN.md §5)
with synthetic gates passed before real data; post-hoc analyses labeled.

---

## RQ1 — Refit stability under population growth (RQ1_FINDINGS.md)

- **Headline (pre-registered): NULL.** BT vs lattice refit stability
  indistinguishable with data/ties/anchor held identical. δ=1mo,
  incumbents ≥1000 votes, 13 windows: τ_b lattice-better 5 / worse 3 /
  exactly tied 5; median paired Δτ_b = +0.00000. Mean τ_b 0.98434
  (lattice u0.1) vs 0.98358 (BT). Reproduced at units 0.1/0.5855/0.8,
  δ∈{1,3}, both incumbent sets. No incumbent moved >5 ranks in any
  window under either method. (rq1_metrics.csv)
- **Secondary: stability is governed by sample size/pool maturity, not
  link family.** τ_b window range 0.968→0.997 (improving with pool
  maturity) vs between-method differences ≤0.007 — ~2 orders of
  magnitude apart.
- Magnitude metrics (slope-matched convention θ·slope(0)/0.25,
  labeled convention-dependent): mean|Δθ| ≈1.73–1.77 Elo-equiv pts, all
  methods within 0.04 of each other.
- **Tie-propensity drift** (context for all unit variants): quality-tie
  share 13.1% (first 3 months) → 20.45% (full period, non-bothbad
  denominator); profile-fitted unit 0.5855 (early) vs 0.8002 (full).
- Post-hoc (labeled, scripts/09): entrant-intensity slice — nothing; all
  |ρ|≤0.37, descriptive p≥0.22, n=13.
- Method facts fixed here, used everywhere: anchor gpt-4-0613=0 both
  methods every fit (src/anchoring.py); slope matching because the
  logistic's 0.25 toss-up slope is unattainable by any unit (lattice
  slope 0.284 probit-limit → 0.337 at u0.8).

## RQ2a — Pool-composition DiD (RQ2_DESIGN.md; SCOPE_REFRAME.md descope)

- **DESCOPED, not run.** Reasons recorded in SCOPE_REFRAME.md: (1)
  diminishing marginal value after four convergent equivalence/ceiling
  results; (2) sharper: the one unexplained signal (RQ2b excess) is
  ANTI-correlated with entry activity (excess blocks: 7/16 entries,
  0.62 entrant vote share; clean blocks: 21–22, 0.89–0.91), so an
  entry-proximity DiD is structurally the wrong instrument for it.
  Design retained on record as designed-but-not-run.

## RQ2b — Link-shape via triple additivity (log entries 2026-07-10;
RQ2B_EXCESS_FINDING.md)

- **Headline (pre-registered): links indistinguishable.** 786 dense
  stable triples, 5×60-day blocks. Pooled mean z²: logit 1.202, lattice
  u0.5855 1.231, u0.1 1.252, u0.8 1.279; lattice lower in 1/5 blocks.
  Synthetic calibration passed (true-link z²=1.02–1.03) AND showed the
  statistic has no discrimination power between these links at real gap
  scales (spread ~0.01–0.04 in both generating directions).
  (rq2b_blocks.csv, rq2b_summary.csv)
- **Convergence fact: second independently-derived effect ceiling.**
  Structurally different statistic (within-window additivity residuals;
  no held-out scoring, no plugin, no time splits) agreeing with RQ3's
  analytic KL ceiling that link shape is not discriminable at real gaps.
- **Standalone finding (RQ2B_EXCESS_FINDING.md): temporally-localized
  excess dispersion under EVERY tested link.** Mean z² 1.57
  (2023-08-22→10-21; localized to its second half ≈ late Sep–Oct, half
  z² 2.04 vs 0.91) and 1.65 (2023-12-20→2024-02-18; both halves 1.74/
  1.54); 13–14% of triples |z|>2 vs 4.6% nominal. Ruled out: dedup
  transition (overlaps only the cleanest block), entry activity
  (anti-correlated), language/judge/tie-share/dup-prompt covariates,
  single-model artifact, within-window drift. Open question; candidates:
  judge-population composition shifts, genuine context/set effects.
  (rq2b_excess_covariates.csv, _models.csv, _halfblock.csv)

## RQ3 — Held-out decisive calibration (RQ3_FINDINGS.md)

Sign: Δ = ll(BT) − ll(lattice), positive = lattice better;
MPD = 4e-4 nats/vote (10-Elo-error anchor).

- **Headline (pre-registered): INCONCLUSIVE at all three units.**
  Primary u0.5855: pooled +0.25×MPD, CI (−0.17, +1.22); 11–12/13 windows
  sub-practical BT lean vs ONE window (2023-11-30) at +5.8×MPD.
  (rq3_pooled_verdicts.csv)
- **Best-case-location result (CI-bearing, citable):** in the one
  location theory predicted a lattice-truth effect could exist
  (high-noise-bin recent stratum; predicted +1.58×MPD under
  lattice-truth), measured **−0.50×MPD, CI (−1.10, −0.06)** — evidence
  against lattice-truth in its own best-case location.
  (rq3_bin_pooled.csv)
- Recent stratum overall (IV-pooled): −0.12×MPD, CI (−0.18, −0.06) →
  "practically equivalent despite a detectable directional lean (BT)."
- **Generalizable findings:** reliability slope ≈1.455 for ALL four
  models (shared underconfidence on next-month votes;
  rq3_reliability.csv); 25–75% of decisive test votes unscoreable
  (cold-start coverage). Joint point: coverage and nonstationary drift
  bind next-month predictability; link choice does not.
- **Cold-start shrinkage case study (n=1):** pplx-7b-online entered
  2023-11-30 with 2 training votes; BT extrapolated θ̂=+0.432, lattice
  +0.004; BT paid ≈68 nats that window (~80% of the window's delta).
  Estimation-side face of steep-link × parameter-uncertainty.
- Post-hoc decomposition (labeled, scripts/16): excluding <30-training-
  vote models, outlier collapses +5.8→+0.04×MPD; equivalence with
  sub-practical BT lean at all units (−0.141/−0.106/−0.217×MPD at
  u0.1/u0.5855/u0.8 — NOT monotone in steepness; steepest-unit-largest
  is suggestive only, correction logged 2026-07-10).
- Pre-analysis facts reused downstream: in-family effect ceiling
  ≤0.23×MPD on the empirical gap distribution (median |gap| 0.31);
  bootstrap-calibrated SEs = Fisher/0.80 (validated ρ=0.983 vs published
  bootstrap); plugin-noise reversal (World P2, −1.41×MPD with a
  correctly-specified steep link) does NOT transfer to real pooled noise.

## RQ4 — Tie mechanisms: lattice dead-heat vs Davidson (RQ4_FINDINGS.md)

Sign: Δ = ll(Davidson) − ll(lattice), positive = lattice better;
MPD = 3e-4 nats/vote (1pp tie-probability error at 20% tie level).

- **Headline (pre-registered): INCONCLUSIVE.** Main variant pooled
  +0.838×MPD, CI (+0.017, +2.542); 10/13 windows lattice-better; driven
  by the 2023-11-30 window (+9.55×MPD). (rq4_pooled_verdict.csv)
- **Outlier driver (corrected 2026-07-10 after an initial wrong
  attribution):** NOT the pplx event (−2.8% of window delta); the tie
  channel on near-peer same-family pairs (+407 nats on 7,403 tie votes
  vs −301 on decisive; top pairs gpt-4-0314↔gpt-4-1106-preview,
  gpt-4-0314↔gpt-4-0613, claude-1↔claude-2.1).
  (rq4_outlier_decomposition.csv)
- **Tie-band answer (the 5a question): DIVERGENT shapes the data cannot
  discriminate.** Same at-zero mass every window (P(tie|0) within ~0.01);
  half-max half-widths ≈2.85–2.93 (Davidson) vs ≈1.67–1.68 (lattice)
  ability units — stable ~1.7× structural difference; empirical decay
  lies between; bin RMS 0.0225 vs 0.0237; |ĝ|≳1.2 carries <10% of votes.
  (rq4_tie_band_table.csv, rq4_tie_curve_bins.csv)
- **Drift trajectories:** ν̂ 0.376→0.559, û 0.594→0.803, near-monotone,
  tracking tie-share drift. **Three-way cross-validation** (methods-
  validity): windowed û endpoints reproduce the two independent earlier
  estimates (0.5855 first-3-months; 0.8002 full-sample) to ~0.01.
  (rq4_param_trajectories.csv)
- **Entanglement asymmetry (diagnostic, qualified):** forcing both-bad-
  scale tie mass (~35%) pins the lattice unit at the pre-specified—but
  never stress-tested—grid ceiling (1.4) in 11/13 windows; capacity is
  not the issue (40% at-zero mass at u1.4); on identical held-out
  decisive votes the lattice loses +4.75×MPD (ceiling-window mean) vs
  +2.50×MPD for the Davidson control → lattice-specific excess ≈+2.2×MPD.
  **Tie mass and decisive discrimination are entangled in the lattice's
  single-unit parameterization; provably orthogonal in Davidson's** (its
  decisive link is exactly logistic for any ν — proof in PAPER_NOTES.md).
  (rq4_bothbad_entanglement.csv)
- Post-hoc filter (labeled): still inconclusive (+0.861×MPD) — the
  filter cannot resolve RQ4 because the outlier isn't cold-start votes.
- Pre-analysis fact: tie-mechanism ceiling ≤0.31×MPD in both truth
  directions on empirical gaps (rq4_tie_ceiling.csv).

## Cross-cutting mechanism: steep-link × parameter-uncertainty (two faces)

- **Prediction-side overconfidence:** steeper links produce overconfident
  plugin predictions under ability noise. Evidence: synthetic World P2
  (correctly-specified steep lattice loses −1.41×MPD at high noise);
  does NOT operate at real pooled noise (+0.05..+0.27×MPD retained under
  lattice-truth; scripts/12–13); suggestive qualified dose-response in
  RQ3's filtered leans (largest lean at steepest unit, shallow pair
  inverted). General caution for ranking/reward-model deployments
  (PAPER_NOTES.md).
- **Estimation-side shrinkage:** steeper links implicitly shrink
  near-zero-data models toward the field. Evidence: exactly ONE
  empirical instance (RQ3's pplx window, n=1) + structural argument.
  RQ4 is NOT a second instance (corrected).
- **Corrected cross-RQ statement:** both calibration comparisons (RQ3,
  RQ4) were driven inconclusive by single-window episodes in the same
  calendar month through DIFFERENT channels (cold-start/decisive vs
  tie-shape/near-peer) — at real effect ceilings (≤0.23–0.31×MPD),
  verdicts are hostage to individual episodes, not systematic link
  differences.

## Scope constraints binding all writing (SCOPE_REFRAME.md)

- Never claim to test Cotton's multi-entrant field coherence (pairwise
  data never exercises winner-of-many; 2-entrant lattice = gap link).
- Positioning: empirical audit of where a flexible-base-density
  Thurstonian gap-link does and doesn't measurably differ from BT in a
  real large-scale deployment. Answers: RQ1 stability — doesn't; RQ2b
  link shape — indistinguishable; RQ3 calibration — inconclusive-by-
  episode, equivalent-with-BT-lean away from cold-start; RQ4 ties —
  inconclusive-by-episode, divergent shapes indiscriminable. Nulls
  reported first-class.
- RQ2a descope stated with reasoning, excess named as open question.

## Numbers most likely to be quoted (quick reference)

| quantity | value | source |
|---|---|---|
| BT replication vs published board | MAE 0.18 Elo pts, ρ=0.99997 | scripts/04, Phase 2 |
| validation across eras | MAE 0.18–1.01 (10 BT-era snapshots) | rq1_validation_published.csv |
| RQ1 median paired Δτ_b | +0.00000 (5/3/5 split) | rq1_metrics.csv |
| tie-share drift | 13.1% → 20.45% | logs 2026-07-10 |
| fitted tie params drift | ν̂ 0.376→0.559; û 0.594→0.803 | rq4_param_trajectories.csv |
| RQ3 effect ceiling (empirical gaps) | ≤0.23×MPD | rq3_ceiling_empirical.csv |
| RQ4 tie-mechanism ceiling | ≤0.31×MPD | rq4_tie_ceiling.csv |
| RQ3 best-case-location | −0.50×MPD, CI (−1.10, −0.06) | rq3_bin_pooled.csv |
| reliability slope, all methods | ≈1.455 | rq3_reliability.csv |
| unscoreable test votes | 25–75%/window | rq3 window tables |
| RQ2b excess | z² 1.57/1.65, 13–14% \|z\|>2 | rq2b_excess_covariates.csv |
| tie-band half-widths | 2.85–2.93 vs 1.67–1.68 | rq4_tie_band_table.csv |
| entanglement excess | ≈+2.2×MPD decisive LL | rq4_bothbad_entanglement.csv |
| MPDs | RQ3 4e-4; RQ4 3e-4 nats/vote | pre-analysis docs |

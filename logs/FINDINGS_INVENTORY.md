# Findings Inventory — factual compilation across all RQs

Compiled 2026-07-10; REVISED 2026-07-13 after the interpolation-artifact
discovery (RESEARCH_LOG 2026-07-13): all fit-dependent tables regenerated
at HEAD (consistent Hermite-spline interpolation; synthetic gates
re-certified 05/06/10 all PASS). BT-only quantities (validation track)
were never affected. This is a COMPILATION with exact numbers and source
references, not a narrative; every number traces to a findings doc, the
research log, or a results table at current HEAD.

Dataset: LMSYS clean_battle_20240814 (1,799,991 anony battles, 129 models,
2023-04-24→2024-08-14; dedup_sampled subset 1,670,250 used for all fits).
Pipeline validated against 10 published BT-era leaderboard snapshots at
MAE 0.18–1.01 Elo pts on 9/10 (one outlier: 2024-04-03 at MAE 2.26,
flagged uninvestigated in the RQ1 log) (rq1_validation_published.csv);
dedup switchover empirically dated May–Jun 2024. All experiments pre-registered
(RQ1_SPEC.md, RQ3_PREANALYSIS.md, RQ4_PREANALYSIS.md, RQ2_DESIGN.md §5)
with synthetic gates passed before real data; post-hoc analyses labeled.

---

## RQ1 — Refit stability under population growth (RQ1_FINDINGS.md)

- **Headline (pre-registered): NULL** (HEAD re-run 2026-07-13). δ=1mo,
  incumbents ≥1000 votes, 13 windows: median paired Δτ_b = +0.00026;
  lattice-better 7 / worse 4 / tied 2; mean τ_b 0.98469 (lattice u0.1)
  vs 0.98358 (BT); max |window diff| 0.00713. Reproduced at all units,
  δ∈{1,3}, both incumbent sets. No incumbent moved >5 ranks in any
  window under either method. (rq1_metrics.csv)
- **Secondary: stability is governed by sample size/pool maturity, not
  link family.** τ_b window range 0.968→0.997 (improving with pool
  maturity) vs between-method differences ≤0.007 — ~2 orders of
  magnitude apart.
- Magnitude metrics (slope-matched convention θ·slope(0)/0.25,
  labeled convention-dependent): mean|Δθ| ≈1.70–1.77 Elo-equiv pts, all
  methods within 0.07 of each other (HEAD).
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

- **Headline (pre-registered classifier at HEAD): EQUIVALENCE with a
  sub-practical BT-leaning directional lean, at all three units.**
  Pooled: u0.5855 −0.121×MPD CI (−0.191, −0.074); u0.1 −0.148
  (−0.201, −0.116); u0.8 −0.226 (−0.321, −0.164). BT better 12–13/13.
  **ALL 13 windows inside the ±MPD band** (max |window| 0.63×MPD).
  Leans NOT monotone in link steepness (u0.1 exceeds u0.5855).
  (rq3_pooled_verdicts.csv, rq3_window_table_*.csv)
- **SUPERSEDED (artifact):** the previously stored INCONCLUSIVE verdict
  and its +5.8×MPD 2023-11-30 episode were artifacts of pre-spline
  interpolation (RESEARCH_LOG 2026-07-13; attribution reproduced
  exactly: old code +5.792, HEAD +0.017). The cold-start shrinkage case
  study is RETRACTED as a link property: at HEAD the lattice fits
  pplx-7b at +0.287 (vs BT +0.378) — no meaningful shrinkage.
- **Best-case-location (HEAD):** where lattice-truth predicted
  +1.58×MPD (high-noise-bin recent stratum), measured −0.275×MPD,
  CI (−1.169, +0.040): the prediction is decisively excluded; the
  interval now includes zero. (rq3_bin_pooled.csv)
- Recent stratum overall (IV-pooled, u0.5855): −0.110×MPD,
  CI (−0.172, −0.052) → sub-practical BT lean.
- **Generalizable findings (HEAD):** reliability slope 1.4555 (BT) /
  1.4581–1.4604 (lattice) — shared underconfidence on next-month votes
  (rq3_reliability.csv); 24.8–75.8% of decisive test votes unscoreable
  (rq3_unscoreable_by_window.csv). Coverage and drift bind; link does not.
- Post-hoc filter (<30 training votes, scripts/16, HEAD): essentially
  unchanged from unfiltered (−0.143/−0.121/−0.223) — with the artifact
  gone there is no cold-start episode for the filter to remove.
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
  half-max half-widths ≈2.84–2.93 (Davidson) vs ≈1.67–1.68 (lattice)
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
  (correctly-specified steep lattice loses −1.41×MPD at high noise;
  re-certified at HEAD); does NOT operate at real pooled noise
  (+0.05..+0.27×MPD retained under lattice-truth; scripts/12–13, HEAD);
  directionally visible in RQ3's sub-practical leans (largest at the
  steepest unit; NOT monotone — u0.1 exceeds u0.5855).
- **Estimation-side shrinkage: RETRACTED as a link property**
  (2026-07-13). The single apparent instance (pplx window) was the
  interpolation artifact. The real, controllable version is EXPLICIT
  regularization: ridge-BT (scripts/24, labeled post-hoc) shrinks
  θ̂(pplx) +0.378→0.000 over λ∈[1,3] and improves that window by
  ~5.5×MPD vs unregularized fits, with collateral shift on ≥1000-vote
  models ≤0.015 (λ=3). Full 13-window grid: ridge-BT vs lattice pooled
  −0.461×MPD CI (−1.359, −0.109) at λ=1 (inconclusive; driven by the
  ex-episode window where ridge protects and the corrected lattice does
  not). (ridge_sweep_episode.csv, ridge_windows.csv)
- **Posterior-predictive check (scripts/25, labeled post-hoc):**
  uncertainty-integrated prediction (§4.1 machinery, calibrated SEs,
  Gauss–Hermite) changes essentially nothing: pooled −0.121→−0.117×MPD;
  reliability slope 1.455→1.459 both methods. The binding
  miscalibration is shared drift-underconfidence, not plugin
  overconfidence. (pp_windows.csv, pp_reliability.csv)
- **Corrected cross-RQ statement (v2, 2026-07-13):** at HEAD, RQ3 has NO
  episode (all windows in-band) and returns equivalence; RQ4's
  tie-channel episode stands (+9.55×MPD window; post-spline result).
  The earlier "two episodes, same month, different channels" framing is
  superseded: one of the two episodes was numerics.

## Generalization: 2025 Arena release (scripts/27–28; pre-registered
designs transported verbatim; labeled post-hoc generalization)

- Data: lmarena-ai/arena-human-preference-140k — 135,634 battles, 53
  models, 2025-04-17→07-24 (14 weeks), quality-tie share 17.96%;
  12 weekly rolling windows mirror the monthly grid.
- Full-population consistency: Spearman 0.999839, max rank move 1.
- RQ3-style: **equivalence + sub-practical BT lean at all three units**
  (−0.084 / −0.175 / −0.176×MPD at u0.1 / u0.6892 / u0.6926; BT better
  11/13 legs) — replicates the HEAD-RQ3 finding on an independent era
  and pool. (arena2025_rq3_windows.csv, arena2025_verdicts.csv)
- RQ4-style (per-window profiled ν̂/û): lattice-leaning INCONCLUSIVE,
  pooled +0.779×MPD CI (+0.382, +1.093) — the tie-channel lattice lean
  replicates in direction, still straddling the practical threshold.
  (arena2025_rq4_windows.csv)
- Tie-parameter drift: û 0.6892 (first 3 weeks) vs 0.6926 (full) —
  near-absent over 14 weeks; the 16-month drift is a long-horizon
  phenomenon. (arena2025_rq4_trajectories.csv)

## Machinery re-certification at HEAD (2026-07-13)

- Synthetic gates all PASS at HEAD: scripts/05 (A/B/C/E incl. choix and
  fastchat cross-checks), scripts/06 (unit recovery 0.5134 vs 0.5),
  scripts/10 (P/P'/P2/N four gates). scripts/07 constants reproduce
  (full-sample unit 0.8002; tie share 0.2045; consistency ρ=0.99999,
  max rank move 1). Grounding reproduces (SE calibration ratio 0.803,
  ρ=0.983; best-case prediction +1.58×MPD).
- Ablations (scripts/26, labeled): bootstrap-B sweep, lattice
  resolution (L, g_step), profile-grid density — see ablations.csv.

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
| validation across eras | MAE 0.18–1.01 on 9/10 BT-era snapshots (outlier 2.26 at 2024-04-03) | rq1_validation_published.csv |
| RQ1 median paired Δτ_b (HEAD) | +0.00026 (7/4/2 split; means 0.98469 vs 0.98358) | rq1_metrics.csv |
| RQ3 verdict (HEAD) | equivalence + BT lean: −0.121×MPD CI (−0.191,−0.074); all 13 windows in-band | rq3_pooled_verdicts.csv |
| RQ3 best-case-location (HEAD) | −0.275×MPD, CI (−1.169, +0.040); prediction +1.58 excluded | rq3_bin_pooled.csv |
| tie-share drift | 13.1% → 20.45% (2023–24); 17.96% flat in 2025 | logs 2026-07-10/13 |
| fitted tie params drift | ν̂ 0.376→0.559; û 0.594→0.803 (2023–24); û 0.689→0.693 (2025) | rq4/arena2025 trajectories |
| RQ3 effect ceiling (empirical gaps) | ≤0.23×MPD | rq3_ceiling_empirical.csv |
| RQ4 tie-mechanism ceiling | ≤0.31×MPD | rq4_tie_ceiling.csv |
| reliability slope, all methods | 1.4555–1.4604 (plugin); 1.459–1.461 (posterior-predictive) | rq3_reliability.csv, pp_reliability.csv |
| unscoreable test votes | 24.8–75.8%/window | rq3_unscoreable_by_window.csv |
| RQ2b excess | z² 1.57/1.65, 13–14% \|z\|>2 | rq2b_excess_covariates.csv |
| tie-band half-widths | 2.84–2.93 vs 1.67–1.68 | rq4_tie_band_table.csv |
| entanglement excess | ≈+2.2×MPD decisive LL | rq4_bothbad_entanglement.csv |
| ridge cold-start protection | θ̂(pplx) +0.378→0.000 over λ∈[1,3]; ~5.5×MPD window gain | ridge_sweep_episode.csv |
| 2025 replication (RQ3-style) | equivalence + BT lean −0.084/−0.175/−0.176×MPD | arena2025_verdicts.csv |
| artifact attribution | old interp +5.792×MPD vs HEAD +0.017×MPD, same window | RESEARCH_LOG 2026-07-13 |
| MPDs | RQ3 4e-4; RQ4 3e-4 nats/vote | pre-analysis docs |

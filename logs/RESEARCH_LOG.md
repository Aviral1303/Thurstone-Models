# Research Log — Lattice-Thurstone vs Bradley-Terry on Chatbot Arena data

Running log of what was tried, what worked, what didn't, and why. Written for a
future collaborator (and the eventual methodology section). Newest entries at
the bottom. Times are local (America/New_York region assumed from machine).

---

## 2026-07-09 — Phase 0: project setup

**Structure.** Created `data/{raw,processed}`, `src`, `scripts`,
`results/{figures,tables}`, `logs`, `vendor`. Raw data is gitignored
(re-downloadable); processed artifacts selectively gitignored.

**Thurstone package: vendored + editable install.** Cloned
`microprediction/thurstone` into `vendor/thurstone` at commit
`69ed52c8e5817ead9a32140bf7ff9d2e27461f91` and installed with
`pip install -e vendor/thurstone`. Rationale: PyPI has 0.1.0 which may lag the
repo; the editable install from a pinned commit gives us (a) exact
reproducibility, (b) readable/patchable internals, (c) the same import path as
a normal install. If we ever patch the vendored code, we'll fork rather than
carry local diffs silently.

**Environment.** Python 3.14.4 venv at `.venv`. Full pin in
`requirements-lock.txt`. Key versions: numpy 2.5.1, pandas 3.0.3, scipy (latest
at install), choix (pure-python BT/PL MLE), datasets 5.0.0, matplotlib,
pyarrow.

**Theory reading notes** (from BACKGROUND.md, MALLOWS_CRITIQUE.md, DYNAMIC.md,
LITERATURE.md and source):

- Core machinery: `Density` on a `UniformLattice`; `Race.state_prices()` builds
  winner-of-many field distribution once (O(N)); ties get multiplicity-aware
  dead-heat mass (default HalfPointTie splits 50/50 — for pairwise that's
  exactly "half credit each").
- `AbilityCalibrator`: forward (abilities → win prices) via
  `state_prices_from_ability`; inverse (prices → abilities) via monotone
  interpolation on a cached lookup curve, iterated `n_iter` times.
- `GlobalAbilityCalibrator` (global_fit.py): Gauss-Newton over cached curves,
  fits one θ per contestant + optional per-race bias b_r across many races.
- `GlobalLSCalibrator` (global_ls.py): faster baseline — invert each race,
  median-center, slope-weighted LS stitch. Diagonal design → closed form.
- `dynamic.py` / `kalman_tracker.py`: random-walk prior on θ(t), learn σ(Δt)
  from Var(Δθ) ≈ 2τ² + αΔt, then MAP-smooth trajectories. Relevant if we track
  ability drift over Arena history (RQ1 adjacent).

**⚠ Sign convention (verified by smoke test).** The lattice model is a
race-TIME model: `winner_of_many` takes the field MINIMUM, so LOWER
ability value = better. Smoke test: abilities [0.5, −0.5] → win probs
[0.24, 0.76]. Also verified inverse: prices [0.75, 0.25] → abilities
[−0.38, +0.575]. When comparing to BT scores (higher=better) we must flip
sign or we'll silently invert every ranking.

**Caution noted per project brief.** The repo's `research/` +
DIFFEOMORPHISMS.md and RESEARCH_EXECUTION_PLAN.md contain pre-written
"expected findings" scaffolding with no completed study. We are not using any
of that content. Nothing in this project gets written as a result until it has
actually been computed from data.

**Key modeling consideration flagged early (to resolve in Phase 3, noting now
while fresh):** every Arena battle is a 2-entrant race. For a *single* pair,
Thurstone-probit and BT-logit differ only in link-function shape — the
interesting divergence is in how the *global stitching across many
overlapping pairs* behaves and how tie mass is generated. Also: per-battle
"prices" from a single Bernoulli outcome are 0/1, which a monotone-inverse
calibrator can't invert directly (0 and 1 sit at the curve's clipped
endpoints). We will need to aggregate votes into per-matchup empirical win
rates (model_a, model_b) → (wins_a, wins_b, ties) and treat each *matchup
aggregate* as one race with prices = smoothed win rates. This is analogous to
what BT MLE consumes anyway, so the comparison stays fair. Alternative (MLE
on the Thurstone likelihood directly per-vote) would require new code —
decide after Phase 1 data audit.

---

## 2026-07-09 — Phase 1: data scoping

**What we searched.** HF datasets (lmarena-ai/lmsys orgs), the Kaggle
competition mirror, LMSYS's GCS bucket for the leaderboard-notebook battle
files, and the leaderboard HF space for published rating snapshots.

**Primary find: `clean_battle_20240814_public.json`** (2.13 GB, HTTP 200 on
LMSYS's public GCS bucket). 1,799,991 anony pairwise battles, 129 models,
sub-second timestamps 2023-04-24 → 2024-08-14, winner ∈ {model_a, model_b,
tie, tie (bothbad)}, 35.7% ties. Converted to 47.5 MB parquet
(`scripts/01_convert_clean_battle.py`, streaming ijson — 2 GB JSON won't fit
comfortably through pd.read_json). Audit in `scripts/02_audit_clean_battle.py`.
⚠ Earlier-dated snapshots on the same bucket now 403 — this file could vanish;
raw JSON kept locally, parquet is the working artifact. Consider mirroring.

**Dead ends / dropped candidates.** Kaggle 55k mirror: no timestamps, dropped.
lmsys/chatbot_arena_conversations (33k): gated + subsumed by the big file,
dropped. leaderboard_table_*.csv files: turned out to hold only model metadata
(MT-bench, MMLU, license), NOT ratings — the published BT ratings are inside
elo_results_*.pkl (136 snapshots, 2023→2025-08). elo_results_20240813.pkl is
the natural Phase 2 replication target.

**Secondary: `lmarena-ai/arena-human-preference-140k`** verified by streaming
(schema: model_a/b, winner incl. tie/both_bad, µs timestamps,
evaluation_session_id; 2025-04→2025-07, 53 models). Reserved as an
out-of-era robustness set; not fully downloaded yet.

**Verdict on RQs** (full audit: `logs/DATA_AUDIT_phase1.md`): RQ1, RQ3, RQ4
answerable as stated; RQ2 answerable but needs a time-confound-aware test
design agreed before we look at outcomes; listwise Plackett-Luce comparison is
out (no public listwise Arena data exists). Four modeling issues flagged for
Phase 3 (0/1-price inversion → aggregate to matchup rates; sign convention;
lattice-unit ↔ tie-mass coupling; Davidson-BT needs ~50 lines of custom MLE,
choix doesn't ship it).

**Status: Phase 0+1 complete. STOPPED, awaiting user review per ground rule 3.**

---

## 2026-07-09 — Phase 1 review decisions (user)

1. Commit Phase 0/1 (done; push blocked — machine credential
   `aviralpoddar-gsmc` lacks write access to `Aviral1303/Thurstone-Models`).
2. Sign convention must be pinned by unit tests, extended with the wrapper
   test in Phase 3.
3. **Matchup-aggregation plan REJECTED → direct per-vote likelihood.**
   Rationale (recording for methodology section): windowed aggregation either
   hands BT and Thurstone different-resolution data (unfair comparison) or,
   applied symmetrically, throws away the sub-second timestamps we secured
   specifically for RQ1. Instead: precompute p_win(Δθ), p_tie(Δθ) lookup
   curves once via the forward Race/pricing machinery (pairwise races are
   translation-invariant → strictly 1D in the ability gap), then fit
   abilities by direct MLE over individual votes — exactly mirroring how
   choix fits BT, same per-vote data for both models, only the link differs.
   AbilityCalibrator's price-inversion path is NOT used for fitting (it's
   built for continuous market odds, not sparse 0/1 votes).
4. RQ2 gets a dedicated confound-aware design doc (event-study/DiD around
   model-entry events, pooled across events) BEFORE implementation; doesn't
   block Phase 2.
5. RQ4 additions: (a) compare MLE-optimal Davidson ν's implied tie-band vs
   the lattice unit's implied tie-band — divergence decides whether "no
   extra parameter" is honest or relabeled; (b) explicit decision needed on
   merging "tie" vs "tie (bothbad)" — conceptually different (equally good vs
   equally poor); flag recommendation, don't silently pick.
6. RQ1/RQ3 results must be reported both full-population and restricted to
   models with ≥1000 votes (Gini 0.554 → thin-data models are noisy under
   both methods).

Phase 3 implementation HELD until RQ2 design doc reviewed + direct-likelihood
approach validated on synthetic ground truth.

---

## 2026-07-09 — Phase 2: BT baseline replication — PASSED

**Published target.** `elo_results_20240813.pkl` from the leaderboard HF
space. Unpickling required stubbing plotly classes (embedded Figures were
built with pre-6.0 plotly; `heatmapgl` no longer exists). Extracted
`text/full/leaderboard_table_df`: 128 models, `rating_system="bt"`,
last_updated 2024-08-12 09:20:51 PDT. Saved to
`results/tables/published_bt_20240813.csv` (script 03).

**Method.** Faithful port of fastchat's `compute_mle_elo` (script 04): BT MLE
via weighted logistic regression; each decisive battle = weight 2 in its
direction, each tie (both subtypes) = weight 1 in each direction; ratings =
400/ln(10)·coef + 1000, anchored mixtral-8x7b-instruct-v0.1 = 1114. Battles
truncated to the pickle's own last_updated_tstamp (our battle file runs one
day later). One rename needed: battle log's `chatgpt-4o-latest` = published
`chatgpt-4o-latest-2024-08-08`.

**Results** (`results/tables/bt_replication_20240813.csv`):

| variant | spearman | kendall | MAE (Elo pts) | max|Δ| |
|---|---|---|---|---|
| fastchat-LR, dedup_sampled | 0.99997 | 0.9988 | **0.18** | 2.95 |
| fastchat-LR, all battles | 0.99907 | 0.9818 | 4.59 | 16.69 |
| choix BT, decisive-only | 0.99894 | 0.9811 | 41.4 | 150.8 |

- **Gating checkpoint PASSED**: with the `dedup_sampled` filter we reproduce
  the published board to MAE 0.18 rating points, all 128 models covered.
  Residual deviations concentrate in models that entered in the final weeks
  (llama-3.1-8b, gemma-2-2b, athene-70b) — consistent with the one-day
  snapshot mismatch. Conclusion: data pipeline is sound; the published
  pipeline did use the dedup filter (4.6 MAE without it).
- **choix cross-check**: rank order agrees (ρ=0.999) but decisive-only
  likelihood shifts magnitudes badly in the tail (llama-13b off by 150 pts —
  weak models live on "both bad" ties, dropping ties inflates their spread).
  Lesson recorded: every BT baseline in later phases must model ties the
  same way its comparator does; decisive-only fits are not interchangeable
  with tie-aware fits.
- First-fit bug worth remembering: my "simplified" one-row-per-direction LR
  design collapsed to a single class (all Y=1) — sklearn rightly refused.
  fastchat's paired Y=1/Y=0 rows with directional weights is the correct
  encoding; kept faithful to it.

**Sign-convention tests** (decision #2): `tests/test_sign_convention.py`,
4 tests pinning the race-time (lower=better) convention + forward/inverse
round-trip; extend with the sign-flip wrapper + hand-checked real-data subset
in Phase 3. All passing.

**RQ2 design doc** written (`logs/RQ2_DESIGN.md`), not implemented. Core
reframe worth flagging: for pairwise battles BOTH BT and 2-entrant lattice-
Thurstone are gap-link models, so pool-composition effects falsify both;
the doc splits RQ2 into (a) proximity-DiD event study around entries with
event fixed effects (identification from ability-proximity heterogeneity
within events, not calendar time), and (b) a link-shape triple-additivity
test. Synthetic-null + power validation specified before real outcomes are
touched. Awaiting review.

**Status: Phase 2 done, RQ2 design awaiting review, Phase 3 HELD.**

---

## 2026-07-09 — Phase 2 review decisions (user) + scope reframe

- Phase 2 approved. **Standing rule from the choix lesson: every BT-vs-lattice
  comparison must use identical tie treatment on both sides.** Phase 3 fit
  must include ties fastchat-style, not decisive-only.
- RQ2 design approved (RQ2a/RQ2b split confirmed) + one addition: model-family
  clustering robustness (same-lab successors are ability-proximal for
  non-context reasons). Added as §3.6 of RQ2_DESIGN.md: family map committed
  before outcomes are loaded, family×proximity covariate, same-family
  exclusion rerun, downgrade rule if the effect halves.
- **Scope reframe** (logs/SCOPE_REFRAME.md): Arena is pairwise-only, so
  Cotton's simultaneous multi-entrant field-coherence property is never
  exercised. We test (1) global-refit stability of two gap-link models under
  population growth (RQ1) and (2) link-shape/tie-mechanism comparisons
  (RQ2b/RQ3/RQ4). Paper positioning must not claim validation of the
  winner-of-many result; that needs listwise data which doesn't exist
  publicly.
- **Tie categories resolved (RQ4)**: "tie" = dead-heat category for the main
  analysis; "tie (bothbad)" EXCLUDED from it (equally-good ≠ equally-poor;
  dead-heat models closeness of draws, both-bad is an absolute-bar judgment).
  Implemented as `include_both_bad` flag (default False) in src/fit.py so the
  robustness rerun needs no rework. NB: the fastchat-equivalent half-tie mode
  keeps pooling both tie types — that's part of replicating BT's treatment
  faithfully (fastchat pools them); the tie-category distinction applies to
  the NATIVE dead-heat likelihood.

## 2026-07-09 — Phase 3: fitting machinery + synthetic validation PASSED

**Modules.**
- `src/lattice_link.py` — LatticeLink: 1D curves W(g), D(g), L(g) computed
  once from the forward Race/pricing machinery (801 gap points, 0.07s);
  higher-is-better sign flip happens exactly here and nowhere else.
  Curves verified: exact symmetry W(g)=L(−g), D even, W strictly increasing,
  W+D+L=1 to 1e-8. Two likelihood views: decisive link F=W/(W+L) (for the
  fastchat-equivalent mode — BT's logistic has no tie mass, so the fair
  comparison must condition on decisive outcomes) and native trinomial
  (W,D,L) for RQ4. LogisticLink in the same interface for cross-checks.
- `src/fit.py` — per-vote MLE, L-BFGS with analytic gradients via lookup
  interpolation; modes half_tie / native; include_both_bad flag per the tie
  decision; objective normalized by total weight (conditioning).
- `src/bt_baseline.py` — compute_mle_elo extracted from script 04 (shared by
  replication + synthetic checks); defaults stay faithful to fastchat
  (tol=1e-6), cross-checks pass tighter tolerances. Script 04 re-run after
  refactor: MAE 0.18324 unchanged.

**Synthetic validation** (`scripts/05_synthetic_validation.py`, seed
20260709, 30 models, θ*~N(0,1), 200k votes, Arena-like Dirichlet exposure
skew; all pre-stated thresholds met — `results/tables/synthetic_validation.csv`):

| check | spearman | pearson | RMSE | note |
|---|---|---|---|---|
| A lattice-gen → native fit | 0.9916 | 0.9950 | 0.092 raw | ≤0.10 ✓ |
| B lattice-gen → half-tie fit | 0.9964 | 0.9960 | 0.080 | slope 1.000 |
| C BT-gen → our logistic vs choix | 0.9982 | — | max|Δθ| 3.1e-4 | ≤1e-3 ✓ |
| D BT+30% random ties → ours vs fastchat LR | — | — | max resid 0.012 Elo pts | ≤0.1 ✓ |
| E BT-gen → lattice fit (misspec) | 0.9987 | 0.9974 | 0.065 scaled | slope 1.174 |

Debugging notes for the record: (1) initial check C failed at 8e-3 — cause
was optimizer slack on both sides (choix default tol=1e-5; our unnormalized
objective stopping on ftol), not an objective mismatch; fixed by comparing
pure MLEs with tight tolerances. (2) check D initially failed at 0.24 Elo
pts — sklearn LR tol=1e-6 slack in the fastchat port; parametrized tol,
defaults unchanged. (3) check E's slope 1.17 is the expected logit-vs-
probit-like scale factor under misspecification (ranks unaffected), noted so
nobody reads raw-scale differences as a finding later.

**Observation for RQ4 (not tuned, just noted):** with default unit=0.1 the
lattice dead-heat mass at gap 0 is 2.8%, far below Arena's 18.7% quality-tie
share. The native tie band will need a much coarser unit (or the tie-band
comparison will simply show this) — exactly what the Davidson-ν vs lattice-
unit analysis is for.

**Sign tests extended** (decision #2 complete): 7 tests passing, including
the hand-checked real subset (gpt-4-1106-preview vs vicuna-13b, 318 battles,
197/42/79 — counts pinned in the test) through both fit modes. NB this
touches one real pair purely as a sign check, per the decision; no research
fitting on real data has been run.

**Status: synthetic validation PASSED. STOPPED before real-data fitting,
awaiting go-ahead.**

---

## 2026-07-10 — Phase 3 real-data step (after user review)

**Review corrections applied first:**

1. **RQ4 fitted-vs-fitted fix.** The 2.8%-vs-18.7% note in the previous entry
   was a sanity observation about an UNFIT default and must not be cited as a
   preliminary result (now stated in logs/RQ4_DESIGN.md). Davidson's ν is an
   MLE quantity; the lattice unit must be too. Implemented
   `src/fit.py::profile_lattice_unit` (unit enters only via curves → joint
   MLE reduces to a cheap 1D profile; grid + quadratic refinement; chose
   profile over joint because curve rebuild is 0.07s and the profile gives
   the likelihood curve for free). Synthetic check
   (`scripts/06_unit_profile_validation.py`, seed 20260710, true unit 0.5,
   ~189k votes): fitted 0.513 (within 10% criterion), θ recovery Spearman
   0.997 / RMSE 0.039, interior convex minimum. PASSED.
2. **Anchoring convention settled BEFORE any rolling refit** (src/anchoring.py):
   **gpt-4-0613 pinned to 0.0 in native units, both methods, every fit.**
   Reasoning: fixed dated checkpoint (no silent drift), 96,284 votes, active
   2023-06-27 → 2024-08-14 (present in every cumulative window; mixtral-8x7b,
   BT production's anchor, only enters 2023-12-11 — too late). Same anchor for
   both methods so cross-method numbers never mix conventions. Pre-specified
   caveat + sensitivity: single-anchor propagates the anchor's own noise as a
   common offset into |Δθ| metrics → RQ1 will also report median-alignment on
   common incumbents; rank metrics immune by construction.

**Full-population side-by-side fits** (`scripts/07_fit_full_real.py`;
dedup_sampled, full file, 1,670,250 battles, 129 models; identical half-tie
treatment both sides; anchored gpt-4-0613=0):

- BT (logistic link) 4.5s; lattice 12.0s. Ratings:
  `results/tables/full_fits_20240814.csv`, scatter in results/figures.
- **Consistency check (user gate): Spearman 0.99993, Kendall 0.9978,
  max rank move 2 positions** (mistral-7b-instruct 112↔114,
  gemma-1.1-2b-it 114↔112, three others ±2/±1; all mid/low-table).
  The ≥1000-votes subset row is IDENTICAL to full population because every
  model in the full file has ≥1,312 votes — the subset distinction only
  bites in RQ1's shorter windows.
- **Unit profile on real data (RQ4 groundwork, not the RQ4 comparison):**
  fitted unit 0.800 (interior, clean profile), implying D(0)=22.6% at-zero
  dead-heat mass vs 20.45% observed quality-tie share among non-bothbad
  votes. Descriptive closeness only — no Davidson comparator fitted yet, no
  held-out evaluation; the RQ4 protocol remains pending.
- **Unit-(in)sensitivity of the half-tie fit, measured:** ranks invariant
  (Spearman 0.999933 between unit 0.1 and 0.800 fits) but magnitudes rescale
  (max|Δθ|=0.30). Consequence recorded in RQ4_DESIGN.md: RQ1/RQ3 hold the
  unit fixed across all fits within an experiment; proposal = 0.800
  everywhere with unit-0.1 sensitivity rerun of RQ1 headlines.

**Status: side-by-side fits in place and consistent. STOPPED before the
rolling-refit experiment, awaiting user review per instruction.**

---

## 2026-07-10 — RQ1 spec + scale convention (post-review corrections)

**Unit=0.800 for RQ1 rejected by user — look-ahead leak** (full-sample fit
would inject month-16 information into month-3 snapshots; same discipline as
the anchor-timing choice). Replaced per direction with a slope-matching
convention, with one mathematical wrinkle discovered on the way:

- **The logistic's toss-up slope (0.25) is unattainable by any unit.**
  Measured slope of the lattice decisive link at Δθ=0: 0.2837 as unit→0
  (probit-like limit), rising monotonically with unit (0.2899 at 0.1,
  0.3370 at 0.8). Carving a wider dead-heat band always *steepens* the
  conditional link. So "pick the unit that matches slopes" has no solution;
  the intent is implemented the way 400/ln10 works for BT — a reporting
  constant: **θ_matched = θ·(slope(0)/0.25)** via
  `LatticeLink.slope_match_factor()` (test #8). Fits themselves use unit=0.1
  fixed across all checkpoints; magnitude metrics reported slope-matched;
  unit=0.8 rerun kept as robustness of the magnitude metric only.
- **Secondary check → real finding.** Unit fitted on first 3 months only:
  0.5855 vs full-sample 0.8002. Not close — and it tracks a genuine
  composition drift: quality-tie share (non-bothbad denominator) rose from
  13.1% (first 3 months) to 20.45% (full period). Tie propensity increased
  substantially over the 16 months. Flagged as its own finding (feeds RQ4
  interpretation; also reinforces unit-as-convention for RQ1).
- **Tie-share numbers reconciled** (user sanity check): 18.69% (Phase 1
  audit) = quality ties / ALL votes, full file. 20.45% (scripts/07) =
  quality ties / non-bothbad votes, dedup file. Chain: 18.69% → 22.53%
  (denominator excludes both-bad) → 20.45% (dedup filter removes
  proportionally more quality ties — duplicated high-frequency prompts tie
  more often). Both numbers correct; definitions now stated wherever used.
- **Optimizer robustness fix** found by the first-3-months fit: L-BFGS can
  end ABNORMAL at kinks of the piecewise-linear interpolated log-curves
  even when converged (max|grad| 3.6e-5, same scale as fits that report
  CONVERGENCE). fit_gaplink now accepts termination iff max|grad| < 1e-4 on
  the normalized objective, else raises. All 8 tests pass.

**RQ1 full spec written** (`logs/RQ1_SPEC.md`) per review directives:
monthly cumulative checkpoints 2023-07-31 → 2024-08-12 (14); validation
track against published elo_results pkls at their own timestamps (BT-era
gated at Phase-2-level match; online-Elo era Spearman-only; dedup epoch
mapped empirically by fitting both filter variants); incumbents = ≥1000
cumulative votes at T (all-models table alongside); horizons δ=1 and δ=3
months; rank metrics primary (unit-invariant, stated); slope-matched
magnitude metrics secondary with anchor-aligned AND median-aligned variants;
paired per-window method comparison with sign-consistency headline and
full-table transparency. **Awaiting go-ahead before running.**

---

## 2026-07-10 — RQ1 executed (after spec approval + 0.5855 variant added)

**Bug caught by the validation track before it contaminated anything.** First
run's BT-era validation MAEs came out 13–19 Elo pts instead of the ~0.2
expected from Phase 2. Cause: fastchat's compute_mle_elo POOLS both tie
labels into the half-tie weighting, but fit_gaplink defaults
include_both_bad=False and scripts 07/08 never overrode it — our fits
silently dropped the 17% both-bad votes. Synthetic check D never caught it
because that world only emitted 'tie' labels. The internal BT-vs-lattice
comparison was still fair (identical data both sides) but deviated from the
documented fastchat-equivalence requirement. Fixed: half_tie fits now pass
include_both_bad=True everywhere; regression test added
(tests/test_fastchat_equivalence.py — mixed-label synthetic must match
compute_mle_elo AND must diverge without the flag; 10 tests passing).
Native-mode RQ4 tie category decision unaffected (dead-heat = 'tie' only).

**Consistency gate re-run with corrected treatment (scripts/07):** improved —
Spearman 0.999978 (was 0.99993), Kendall 0.99927, max rank move 2, mean
0.047. Unit-insensitivity of half-tie ranks re-confirmed (0.999933); unit
0.1→0.8 magnitude rescale now max|Δθ|=0.254.

**Validation track (14 snapshots, 2023-08 → 2024-08):**
- Online-Elo era (pre-2023-12, rating_system unlabeled): Spearman 0.90–0.97,
  MAE 20–25 — expected methodological mismatch, labeled approximate-only.
- BT era, era-appropriate filter: **MAE 0.18–1.01, Spearman ≥0.9986 on 10/10
  snapshots** (worst: 20240403 at MAE 2.26 — noted, uninvestigated).
- **Dedup epoch mapped empirically**: no_dedup matches better for every
  snapshot 2023-12-06 → 2024-05-01 (MAE 0.27–2.26 vs dedup 2.4–4.0); dedup
  matches better from 2024-06-02 onward (0.18–0.46 vs 3.9–4.1). Production
  dedup switched on between 2024-05-01 and 2024-06-02 — consistent with
  LMSYS's public timeline. RQ1 experiment track itself stays uniformly
  dedup_sampled (declared convention; the validation demonstrates the
  pipeline reproduces production under either filter).

**RQ1 experiment (14 checkpoints, 4 methods, δ∈{1,3}, both incumbent sets,
both alignments — full tables in results/tables/rq1_*.csv):**

Headline (incumbents ≥1000 votes, δ=1, across 13 windows):

| method | mean τ_b | mean spearman | mean\|Δθ\|_med (Elo-eq) | p95 (Elo-eq) |
|---|---|---|---|---|
| BT | 0.98358 | 0.99816 | 1.77 | 6.22 |
| lattice u0.1 | 0.98434 | 0.99829 | 1.73 | 6.14 |
| lattice u0.5855 | 0.98374 | 0.99838 | 1.74 | 6.18 |
| lattice u0.8 | 0.98379 | 0.99818 | 1.74 | 6.13 |

Paired per-window differences (lattice u0.1 − BT): Kendall better in 5/13
windows, worse in 3, EQUAL in 5; median difference +0.00000. Magnitude
smaller in 7/13; median −0.02 Elo-eq pts. δ=3: better in 5/11, median
+0.00000. Same picture at u0.5855 and u0.8, and for the all-models
incumbent set. frac_move_gt5 = 0 in every window for every method.

**Face-value reading (pre-interpretation, for review): a null.** Under
identical data, tie treatment, anchoring, and per-vote MLE machinery, the
lattice link's refit stability is indistinguishable from BT's — per-window
differences are 1-2 orders of magnitude smaller than the window-to-window
variation itself, and sign-inconsistent. No stability advantage for either
method at either horizon. (Interpretation, framing, and whether to slice by
entrant intensity: held for user review per instruction.)

**Tie-propensity drift** (13.1% → 20.45%) is part of RQ1's write-up context
per review directive — the three unit variants exist because the "right"
band drifts; their near-identical stability metrics show the RQ1 result is
robust to that drift.

---

## 2026-07-10 — RQ1 findings written; post-hoc entrant slice: nothing

Per user review of the raw tables:

- **Post-hoc entrant-intensity slice** (scripts/09) — explicitly NOT in the
  pre-registered RQ1_SPEC.md, labeled post-hoc exploratory in the script,
  the output table, and RQ1_FINDINGS.md. Result: nothing. All |Spearman| ≤
  0.37 (descriptive p ≥ 0.22, n=13) for both covariates (n_entrants,
  entrant_vote_share) against both per-method τ_b and the paired method
  differences. Even stability itself isn't visibly entrant-modulated at
  this n; calendar maturation dominates. Noted and moved on — does not
  qualify the headline null.
- **logs/RQ1_FINDINGS.md written** stating the null as a first-class,
  well-powered result (5 better / 3 worse / 5 exactly tied, median diff
  +0.00000, reproduced across 3 unit conventions × 2 horizons × 2 incumbent
  sets); the secondary sample-size-governs-stability finding (τ_b window
  range 0.968→0.997 vs between-method ≤0.007); and the fixed framing
  paragraph tying to SCOPE_REFRAME.md — the paper is an empirical audit of
  where the alternative does and doesn't measurably differ, RQ1's answer is
  "it doesn't, here," reported first-class regardless of RQ3/RQ4 outcomes.

**Status: RQ1 complete and written up. HOLDING before RQ3 — narrative to be
confirmed with user first, per instruction (framing fixed before further
results are generated).**

---

## 2026-07-11 — RQ3 pre-analysis commitment (no real-data numbers computed)

Per user directive: full pre-commitment BEFORE any real calibration number
exists. Deliverables: `logs/RQ3_PREANALYSIS.md` + synthetic validation
(scripts/10, 11; src/rq3_eval.py; tests/test_rq3_classifier.py). Real-data
RQ3 NOT run. Key decisions and what the synthetic work uncovered:

- Design: rolling-origin on RQ1's 14 checkpoints (13 time-disjoint test
  windows); decisive-only conditional scoring (tie prediction deferred to
  RQ4 — BT has no native tie model); recent-entrant = first training battle
  within 28 days before T_k, fixed now; primary lattice unit 0.5855
  (look-ahead-free), 0.1/0.8 sensitivity.
- MPD derived, not asserted: log-loss cost of a uniform 10-Elo gap error =
  ½·0.25·(10·ln10/400)² = 4.14e-4 → **MPD = 4e-4 nats/vote** (Brier 2e-4).
  Inference: window-cluster bootstrap, effective N = 13 windows (stated);
  nested-training dependence flagged. Sign consistency bar 10/13.
- Verdict rules (a/b1/b2/c) implemented + unit-tested in classify() before
  outcomes exist; sub-practical directional leans get reported as such
  inside "equivalence", never promoted, never hidden.
- **Discovery 1 (analytic, scripts/11): in-family effect ceiling ≈1.56×MPD**
  — no plausible lattice link can beat a best-fit logistic by more than
  ~1.5×MPD at population level; realistic parameters 0.2–0.8×MPD.
- **Discovery 2 (World P2): plugin-noise reversal.** In the MOST favorable
  lattice world (strong link, correctly-specified fit), realized held-out
  delta is NEGATIVE (−1.41×MPD): steeper links' plugin-MLE predictions are
  overconfident under ability-estimation noise, costing more than BT's
  shape misfit. Consequence pre-committed in §4: lattice_positive at ≥MPD
  is close to theoretically unachievable here; if it happens anyway, the
  steepness/regularization confound must be analyzed before any generative
  claim — and equivalence is the a-priori expected verdict. Also
  pre-committed: flattest-unit-wins ordering on real data would indicate
  plugin-noise dominance, not tie-band truth.
- **Synthetic gates all PASS** (exit 0): near-zero world → no call; 0.5×MPD
  world → correct direction, CI>0, not promoted; ceiling world → ≥MPD
  effect detected with correct sign (bt_positive, matching noise-free
  truth); logistic world → equivalence, no false lattice call. Statistical
  sensitivity demonstrated down to ~0.3×MPD true effects. 15 tests passing.
- Machinery hardening en route: fit_gaplink restart-from-stall for L-BFGS
  kink stalls; steep-skewed links need g_step=0.005 (real-data links don't,
  remedy kept anyway).

**Status: awaiting user review of the pre-commitment (esp. §4) before any
real-data RQ3 fitting.**

---

## 2026-07-11 — RQ3 pre-commitment review: real-input grounding + additions

User review caught that the ceiling/reversal rested on illustrative inputs
(generic Gaussian gap spreads; synthetic-world noise). Both recomputed from
real quantities already in the repo (scripts/12; NO held-out calibration
number touched — training fits and hypothetical-truth expectations only):

- **Ceiling, empirical gaps**: real vote-weighted gap distribution is far
  narrower than assumed (median |gap| 0.31 vs illustrative σ_g≥2.1) →
  ceiling collapses 1.56×MPD → **≤0.23×MPD** (plausible symmetric units
  0.005–0.08×MPD). Conclusion strengthened ~7×.
- **Reversal, real SEs**: Fisher SEs implemented (known-null-space
  deflation — bare pinv inverted the gauge direction and produced garbage
  on first run, caught immediately: SEs of 45k nats); validated vs
  published bootstrap SDs (Spearman 0.983, ratio 0.80). Real median SEs
  1.6–4.5 Elo. **P2's plugin-noise reversal does NOT transfer**: under
  lattice-truth with real noise, expected delta stays positive
  (+0.05…+0.27×MPD realistic truth; ≤+1.14×MPD extreme truth, earliest
  regime). Corrected honestly in §6.1: equivalence remains the a-priori
  expectation, but via the effect-ceiling route, not the reversal route;
  the overconfidence mechanism stays live for the recent-entrant stratum
  (P2-like noise there).
- **§4.1 added**: pre-specified steepness-confound procedure (uncertainty-
  integrated re-prediction with training-fit Fisher SEs, identical
  correction both methods, full pipeline re-run; fixed wording for
  shrinks-below-MPD vs survives-correction vs inconclusive).
- **§4.2 added**: sub-MPD significant leans reported as "practically
  equivalent despite a detectable directional lean" — effect size is the
  governing word, rule written before any real p-value exists.
- **Paper note added**: plugin-overconfidence crossover flagged as a
  standalone methodological contribution (general to ranking/reward-model
  deployments with steep links and no uncertainty correction).

**Status: real-input grounding done, doc updated. STILL NO real-data RQ3
fitting. Awaiting user go-ahead.**

---

## 2026-07-11 — RQ3 pre-commitment, second review round (items 1–4)

- **SE calibration settled before first use**: §4.1's correction now
  specifies bootstrap-calibrated SEs (Fisher/0.80, the measured ratio vs
  published bootstrap SDs) as primary, raw Fisher as labeled sensitivity.
  Machinery: `rq3_eval.fisher_se_calibrated` (fisher_se moved from
  scripts/12 into src/rq3_eval.py as official §4.1 machinery).
- **Regime-noise mapping corrected (user suspicion confirmed)**: the §6.1
  reversal table was era-level (per-model SEs but pooled-vote averaging).
  Recomputed on the actual 28-day recent-entrant cohort per checkpoint
  (scripts/13; stratum composition from last 28 TRAINING days — strictly
  pre-test): early-2023 cohorts were genuinely noisy (median SE 17.7 Elo vs
  era 5.6) and small (3 models, 1.4k votes/28d); by 2024 Arena's entrant
  oversampling matured (cohort 3.1 vs era 2.7 Elo). Stratum achievable
  effect under lattice-truth: **+1.58×MPD early / +0.24 mid / +0.11 full**
  (realistic truth); +2.38/+0.34/+0.17 (stress bound). No P2-style reversal
  at any calibrated real noise level.
- **u1.2/skew6 labeled a stress-test upper bound** (skewness ≈0.89, near
  the skew-normal max; selected as sweep maximum, not from data).
- **Pre-committed expectation paragraph added** to the framing (RQ1-style):
  full-population equivalence expected; any genuine effect would most
  plausibly appear in early-window recent-entrant strata, and only under
  a more-skewed-than-typical generating process — recorded before real
  fitting so stratified framing can't be read as post hoc.

**Status: all review items done. NO real-data RQ3 numbers computed.
Awaiting go-ahead.**

---

## 2026-07-12 — RQ3 REAL-DATA RESULTS (run exactly per pre-registration)

Pooling spec §5.1 finalized and pushed first (noise-based bins from
per-checkpoint cohort-SE ratios, scripts/14; IV weights within bins; min-N
500/window, 1,500/bin). Then scripts/15: 13 windows × (BT + 3 lattice
units), 484,599 scoreable decisive test votes total. Tables:
results/tables/rq3_*.csv.

**Full-population verdicts (pre-committed classifier): INCONCLUSIVE at all
three units.** Pooled (vote-weighted, §3): +0.25×MPD (u0.5855) with CI
(−0.17, +1.22)×MPD — the CI straddles +MPD. Structure behind it: **11–12 of
13 windows show tiny BT-leaning deltas** (−0.01 to −0.6×MPD), and ONE
window (2023-11-30) is +5.8×MPD, dragging the vote-weighted pooled mean
positive against the sign pattern.

**Post-hoc diagnostic (labeled, no re-verdict): the outlier is a single
cold-start event.** ~80% of the 2023-11-30 window's total delta traces to
pairs involving pplx-7b-online (+pplx-70b-online), which entered with **2
training votes**. BT extrapolated it to mid-table (+0.432); the lattice's
steeper link — which needs smaller gaps for the same win rates, i.e.,
implicit shrinkage — kept it at +0.004. The model was then mass-sampled in
December and performed poorly; BT paid ~68 nats over the window. This is
the §6.1/6.2 steepness mechanism appearing in ESTIMATION form (shrinkage),
not the prediction-overconfidence form — and it favored the lattice, once,
by a lot, in exactly one cold-start window.

**Strata (pre-committed reads):**
- Recent stratum, IV-pooled across all windows: sub-practical BT lean with
  CI excluding 0 at every unit (u0.5855: −0.12×MPD, CI (−0.18, −0.06)) →
  per §4.2: "practically equivalent despite a detectable directional lean
  (BT)". Note the pre-committed weightings diverge instructively: IV
  pooling downweights the cold-start window (huge per-vote variance), so
  the stratum read is BT-leaning while vote-weighted full-pop is dragged
  lattice-positive by the same window.
- High-noise bin, recent stratum (the ONE place §6.2 said a real lattice
  effect could appear, predicted +1.58×MPD under lattice-truth): measured
  **−0.50×MPD, CI (−1.10, −0.06)** — CI-bearing (n=4,488) and negative.
  Evidence against lattice-truth in its best-case location.
- All other bin×stratum cells: between −0.5 and +0.05×MPD, none clearing
  MPD in either direction.

**Reliability diagnostic:** slopes ≈1.455 for ALL four models — everything
is systematically UNDERconfident on next-month data by the same large
margin (true outcome logits ~1.45× predicted). Shared nonstationarity/
pool-maturation dwarfs any link-shape difference; between-method
calibration differences are second-order against it.

**Operational fact (pre-committed to report):** 25–75% of decisive test
votes per window were unscoreable (a brand-new model with zero training
votes on one side) — cold-start coverage, not model choice, is the binding
constraint on next-month predictability.

**§4.1 correction: NOT triggered** (no positive verdict fired).

**What would resolve the inconclusive (§4c obligation):** the verdict
hinges on one cold-start event. A minimum-training-votes scoring filter
(e.g., exclude votes where either model has <30 training votes) would
resolve it — NOT run, since it is not pre-registered; proposed to the user
as a labeled post-hoc sensitivity, RQ1-entrant-slice style.

---

## 2026-07-12 — RQ3 findings doc + post-hoc decomposition (user-reviewed
framing)

User review: INCONCLUSIVE verdict correct, classifier did its job; filter
approved but reframed as a DECOMPOSITION answering a narrower question —
never superseding the pre-registered headline. Cold-start/shrinkage event
elevated to first-class finding (n=1 caveat mandatory); best-case-location
negative result and reliability-slope/coverage findings stated first-class;
sign convention stated explicitly in the findings doc.

**Post-hoc <30-training-votes filter run (scripts/16, labeled)**: the
outlier window collapses +5.8×MPD → +0.04×MPD, confirming it was entirely
cold-start models. Filtered verdicts at all three units: **equivalence
with sub-practical BT lean** (pooled −0.106/−0.141/−0.217×MPD at
u0.5855/u0.1/u0.8, CIs inside the band and excluding 0; BT better in
11–13/13 windows) — matching the expectation recorded before the run.
CORRECTION (2026-07-10 review round): the earlier note here and in the
report claimed the lean "grows monotonically with unit steepness" — false;
steepness order is u0.1 < u0.5855 < u0.8 (slopes 0.290/0.324/0.337) while
leans are −0.141/−0.106/−0.217. The steepest unit does show the largest
lean by a wide margin (suggestive of the overconfidence face) but the two
shallow units are inverted; RQ3_FINDINGS §4 now states the qualified
version.

**logs/RQ3_FINDINGS.md written** (RQ1_FINDINGS structure): §1 pre-registered
inconclusive headline + sign convention; §2 CI-bearing strata (best-case-
location −0.50×MPD against lattice-truth; recent-stratum sub-practical BT
lean; weighting-divergence note); §3 generalizable findings (reliability
slope 1.455 all methods + 25–75% unscoreable = coverage & drift bind, not
link choice); §4 cold-start shrinkage case study (two faces of steep-link ×
parameter-uncertainty, regime-dependent, n=1 caveat); §5 post-hoc
decomposition with its separate-question framing; §6 scope.

**Status: RQ3 complete and written up. Holding before RQ4 discussion.**

---

## 2026-07-10 — Review round: dose-response correction; RQ2b run; RQ4
pre-analysis written; RQ2a HELD

**Correction of my own prior claim.** The requested "monotone dose-response"
sentence could not be written as requested: checking against slopes
(0.290/0.324/0.337 for u0.1/u0.5855/u0.8) vs leans (−0.141/−0.106/−0.217),
the pattern is NOT monotone — u0.1 and u0.5855 are inverted. My earlier
report claimed monotonicity by reading the table in unit-label order; the
user's instruction echoed that error. RQ3_FINDINGS §4 now carries the
qualified version (steepest unit largest lean by a wide margin = suggestive
of the overconfidence face; shallow-unit ordering within noise), with the
correction noted inline.

**RQ2b run as pre-registered** (scripts/17; RQ2_DESIGN §5). 60-day blocks,
stable models, ≥100 decisive/leg → 5 usable blocks, 786 triples.
- Synthetic calibration gates PASSED (true-link mean z²≈1.02–1.03) — but
  the same check shows the four links differ by only ~0.01–0.04 in mean z²
  in BOTH synthetic worlds: **the additivity test has essentially no
  power to distinguish these links at real gap scales** (consistent with
  the RQ3 effect ceiling; now demonstrated for RQ2b's statistic too).
- Real data: mean z² = 1.20 (logit) / 1.23 (u0.5855) / 1.25 (u0.1) / 1.28
  (u0.8); lattice lower in 1/5 blocks. Between-link differences (~0.03–
  0.08) are the same order as the synthetic no-power spread → **no link
  can honestly be crowned**; pre-registered "whichever residualizes
  better" answer: indistinguishable.
- **Convergence framing (added at review): this is a SECOND,
  independently-derived effect-ceiling finding.** RQ3 bounded the
  link-shape effect via held-out calibration (analytic KL ceiling
  ≤0.23×MPD on empirical gaps); RQ2b now reaches the same conclusion via
  a structurally different statistic (within-window triple-additivity
  residuals — no held-out scoring, no MLE plugin, no time splits). Two
  tests with different failure modes converging on "no discriminable
  link-shape effect at real gap scales" is stronger evidence than either
  alone; the paper should state this convergence explicitly.
- The real finding: **all links show mean z² ≈ 1.2–1.3 > 1** — a modest
  additivity excess SHARED by every gap-link, driven by blocks 02 and 04
  (z²≈1.6–1.7; ~7–9% of triples with |z|>2 vs ~4.6% nominal). Consistent
  with within-window drift / heterogeneity / genuine set effects straining
  ANY static gap-link — the RQ2a-flavored phenomenon, visible without the
  DiD machinery. Caveat: triples share legs; block-level reads only.

**RQ4_PREANALYSIS.md written (doc only — Davidson NOT implemented, no
synthetic run, no real data).** Key decisions: WINDOWED fitting on the
RQ1/RQ3 grid with per-window profiled ν and unit (pooled-fit-with-caveat
rejected: pooled tie parameters average the 13.1→20.45% drift and the
caveat would do all the work); MPD_RQ4 = 3e-4 nats (1pp tie-probability
error at the 20% tie level, derivation in doc); RQ3 classifier reused;
tie-curve-shape grade + effective-tie-band (relabeling-vs-divergence)
table pre-specified; synthetic gates specified (§6) to run after doc
review. Grounding (scripts/18, empirical-gap boundary, no outcomes):
**tie-mechanism ceiling ≤0.31×MPD in both truth directions** →
equivalence a-priori expected; the informative outputs will be the
parameter-drift trajectories and the tie-band answer.

**RQ2a: HELD per user decision** — descope-vs-run to be weighed after RQ2b
+ RQ4-design review with the four convergent data points in hand.

---

## 2026-07-10 — RQ2b additivity-excess characterization (DESCRIPTIVE /
EXPLORATORY — anomaly characterization, not a pre-registered test)

scripts/19; tables results/tables/rq2b_excess_*.csv. Answers to the four
review questions:

1. **Which blocks.** block02 = 2023-08-22→2023-10-21 (z²=1.57, 14.3% of
   triples |z|>2) and block04 = 2023-12-20→2024-02-18 (z²=1.65, 13.3%).
   block05 (2024-02-18→04-18) moderate at 1.37; block03 and block06 clean
   (1.05, 1.04).
2. **Covariates: nothing checked explains it.**
   - Dedup switchover (May–Jun 2024): overlaps ONLY block06 — the
     cleanest block. Ruled out.
   - Entry activity: excess blocks have FEWER entries (7/16) and LOWER
     entrant vote share (0.62) than the clean/moderate late blocks
     (21–22 entries, 0.89–0.91 share). If anything anti-correlated.
   - English share is high (0.77–0.78) in blocks 02, 03, 04 alike — but
     block03 is clean, so language mix doesn't separate excess from clean.
   - Judges/votes-per-judge/tie-share/high-freq-share: all smooth
     monotone-ish trends across blocks, none tracking the excess pattern.
3. **Concentration: broad, with mild legacy-model tilt, no single driver.**
   Excess triples involve the era's models roughly in proportion to their
   triple counts; mildly overrepresented: gpt-3.5-turbo-1106 (6/16 of its
   triples in excess), claude-1 (4/12), vicuna-13b (3/10), palm-2 (2/6).
   gpt-4-0613 appears most in absolute terms (9 excess triples in block04)
   but at near-base-rate share. Not a one-model artifact.
4. **Half-block check: NOT within-block drift.** block04's excess persists
   undiminished in both 30-day halves (H1 1.74, H2 1.54); block02's is
   concentrated in its second half (H1 0.91, H2 2.04) — i.e., a
   ~30-day-localized episode (late Sep–Oct 2023), not gradual drift
   across the 60 days. (block05's moderate excess sits in H1, adjacent to
   block04.) A drift artifact would have shrunk toward 1 in the halves.

**Net descriptive characterization for the paper**: static gap-link models
(all four links equally) show excess triple-additivity dispersion
(mean z² 1.6–1.7, ~13–14% of triples |z|>2 vs 4.6% nominal), concentrated
in two eras — late-Sep–Oct 2023 and Dec 2023–Feb 2024 — spread broadly
across models with a mild legacy-model tilt, and NOT attributable to entry
intensity, language mix, judge-population size, the dedup pipeline epoch,
or within-window drift. Cross-link invariance means it is a property of
the vote-generating process, not of any link choice. Cause unidentified;
candidates (judge-population composition shifts within those eras,
context/set effects, per-pair heterogeneity) would need RQ2a-style
machinery to separate — this is the concrete content behind the
named-open-question option.

---

## 2026-07-10 — RQ4 §6 synthetic gates PASSED; spline fix for optimizer

**DavidsonLink implemented** (src/davidson_link.py; 5 unit tests): analytic
trinomial gap link; verified that Davidson's conditional decisive link is
EXACTLY logistic for any ν (so ν is purely tie mass — Davidson and vanilla
BT coincide on decisive outcomes; clean structural fact for the paper).
Trinomial eval in src/rq4_eval.py (Δ = ll_dav − ll_lat, positive = lattice
better; MPD_RQ4 = 3e-4).

**Optimizer pathology root-caused and fixed.** Repeated ABNORMAL L-BFGS
stalls (scripts/10 W2, scripts/20; grad 1.2–2.4e-4 after restarts) were
caused by LatticeLink returning piecewise-LINEAR log-curve values with
SMOOTHED gradients — value and gradient mutually inconsistent, breaking
line searches near kinks. Fixed at the source: C1 cubic-Hermite splines
(value and derivative exactly consistent — centered-FD check 1e-5).
Diagnostic dead-end recorded: a first "consistency check" using
np.gradient on dense evaluations flagged errors ~1.0 that turned out to be
genuine staircase noise in the lattice tail curves (W~1e-4 at |g|≈4.8,
unit 0.8) — curve property, not implementation error; tiny-eps centered
differences are the right check. All 20 tests pass; interpolation change
is at 1e-6 level, no result materially affected.

**Gates (scripts/20, results/tables/rq4_synthetic_gates.csv): ALL PASS.**
W1 Davidson-truth: verdict davidson_positive, ν recovered ≤6.4% err;
W2 lattice-truth: lattice_positive, unit ≤3.9% err; W3 realistic-gap
matched world: true Δ −0.01×MPD, verdict equivalence; no false ≥MPD calls.
Noted in the doc: W1/W2's ±4–6×MPD true effects are wide-gap
machinery-power demonstrations (sd 1.0 ≫ real gaps); W3 at real-scale
gaps confirms the ceiling and the equivalence expectation.

**Status: RQ4 gates passed, doc updated. Awaiting user review before real
RQ4 fitting. RQ2a decision pending with characterization in hand.**

---

## 2026-07-10 — RQ4 REAL-DATA RESULTS (run exactly per pre-registration);
RQ2a descoped; excess finding + paper notes written

RQ2a descope written into SCOPE_REFRAME.md (sharpened instrument-mismatch
reasoning); standalone RQ2B_EXCESS_FINDING.md; PAPER_NOTES.md seeded with
the Davidson-decisive-link-is-logistic proof. Then scripts/21, both
pre-committed variants. Tables: results/tables/rq4_*.csv.

**Main variant (dead-heat = 'tie' only): INCONCLUSIVE** — pooled
+0.838×MPD, CI (+0.017, +2.542), 10/13 windows lattice-better. Same §4c
structure as RQ3, and the SAME driver: 2023-11-30 (+9.55×MPD — the pplx
cold-start window, now expressed through the trinomial channel) plus the
small first window (+5.29×MPD, n=3.1k). Remaining 11 windows sit in
±0.8×MPD. The verdict stands as pre-registered; no post-hoc filter run
without approval (the RQ3-style <30-votes decomposition would be the
obvious labeled follow-up).

**Drift trajectories (first-class output): the tie-band drift is now
quantified.** ν̂ rises 0.376→0.559 and û rises 0.594→0.803,
near-monotonically, both tracking the quality-tie-share drift. Strong
consistency check: the windowed û endpoints reproduce the two independent
earlier profile fits (0.5855 on first-3-months data; 0.8002 full-sample)
almost exactly.

**Tie-band answer (the 5a question): DIVERGENT, not relabeled.** The two
fitted mechanisms imply nearly identical at-zero tie mass every window
(P(tie|0): 0.158–0.219 Davidson vs 0.168–0.227 lattice) but genuinely
different band SHAPES: half-max half-width ≈2.85–2.93 ability units
(Davidson, slow sech-like decay) vs ≈1.67–1.68 (lattice, faster
overlap decay) — stable across all 13 windows. Empirical tie-rate-vs-gap
bins: both models undershoot near toss-ups (fitted ≈0.20 vs empirical
≈0.23 at |ĝ|<0.15) and the empirical decay lies between the two fitted
shapes; vote-weighted RMS 0.0225 (lattice) vs 0.0237 (Davidson) —
near-identical fit quality despite the different shapes (the gap range
where they differ most is data-thin).

**Robustness variant (both-bad mapped to tie): INCONCLUSIVE with a
structural caveat that is itself informative.** The lattice unit slammed
into the pre-set grid ceiling (1.4) in 11/13 windows — the dead-heat
mechanism cannot express ~35% tie mass at plausible units, while Davidson
absorbs it trivially (ν̂≈1.0–1.08, interior). Early windows show huge
Davidson advantages (−7 to −24×MPD) exactly where the bound binds; late
windows flip lattice-positive (+4.5 to +7.5). Comparison is unfair to the
lattice at the bound (grid was pre-set; extending it would be post-hoc —
flagged for interpretation discussion, not rerun). Honest reading
recorded: the dead-heat mechanism structurally refuses both-bad-scale tie
mass — which independently supports the original decision that both-bad
is not a dead-heat outcome.

**Status: RQ4 done — the last pre-registered empirical piece. Synthesis
transition flagged to user per instruction (findings inventory + paper
outline next, no fifth investigation without a deliberate decision).**

---

## 2026-07-10 — RQ4 post-hoc round: A MATERIAL CORRECTION, entanglement
diagnostic, findings doc

**Correction (important).** My RQ4 report asserted the 2023-11-30 outlier
was "the pplx cold-start event, now expressed through the trinomial
channel" — asserted without decomposition, and the user's follow-up
directive #1 built on it. The <30-votes filter (scripts/22 Part A)
contradicted it: pplx votes are excluded by the filter yet the window's
delta ROSE (+9.55→+11.5×MPD). Decomposition (scripts/23,
rq4_outlier_decomposition.csv): **pplx pairs = −2.8% of the window delta;
the actual driver is the tie channel on near-peer same-family pairs**
(ties +407 nats lattice-better, mean +0.055/vote; decisive −301
Davidson-better; top pairs gpt-4-0314↔gpt-4-1106-preview,
gpt-4-0314↔gpt-4-0613, claude-1↔claude-2.1) — §2's band-shape divergence
mattering once, in a window where votes concentrated on small-gap sibling
pairs with a high tie rate. RQ3's and RQ4's outliers share a calendar
window, NOT a mechanism. Consequences: (i) the shrinkage mechanism keeps
exactly one empirical instance (RQ3, n=1) — RQ4 is NOT a second data
point; (ii) the corrected cross-RQ statement (verdicts hostage to
single-window episodes at sub-ceiling effect sizes, two different
channels) replaces the one-mechanism version everywhere; (iii) both
findings docs corrected inline with the error acknowledged. Also noted
descriptively: the window abuts the Dec-2023→Feb-2024 additivity-excess
era — shared "unusual era" background is a candidate, unverified.

**Part A (labeled post-hoc)**: filtered verdict still inconclusive
(+0.861×MPD, CI +0.017..+2.693) — unlike RQ3, the filter does not resolve
RQ4's inconclusive (because the outlier isn't cold-start votes).

**Part B (both-bad entanglement diagnostic, no new fitting decisions):**
(a) the 1.4 grid ceiling WAS pre-specified (RQ4_DESIGN geomspace to 1.4)
but never stress-tested at both-bad-scale mass (gates used ≤1.2, 10–20%
ties). (b) Capacity is not the issue (unit 1.4 expresses 40% at-zero tie
mass); the saturation is SHAPE-driven (narrow band must inflate at-zero
mass to cover large-gap ties). (c) **Entanglement confirmed, qualified**:
on identical held-out decisive votes, both-bad-variant fits degrade the
lattice's decisive log-loss by +4.75×MPD (mean, 11 ceiling windows) vs
+2.50×MPD for the Davidson control (which isolates the common
ability-refit effect; Davidson's ν provably never touches its decisive
link) → lattice-specific excess ≈+2.2×MPD. The citable asymmetry:
tie-mass capacity and decisive discrimination are ENTANGLED in the
lattice's single-unit parameterization, orthogonal in Davidson's.
Qualifications: modest, window-heterogeneous, boundary never
stress-tested. The tie/both-bad category decision stands on its original
conceptual reasoning, not on this result.

**logs/RQ4_FINDINGS.md written** (RQ1/RQ3 template): inconclusive headline
with corrected driver; divergent-but-indiscriminable tie-band shapes
stated as the precise claim (§2); drift trajectories + three-way
cross-validation (windowed û endpoints ≈ 0.5855/0.8002 earlier
independent fits) as a methods-validity result (§3); entanglement addendum
per actual evidence (§4); outlier correction + corrected cross-RQ
statement (§5).

---

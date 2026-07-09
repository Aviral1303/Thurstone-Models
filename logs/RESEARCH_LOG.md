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

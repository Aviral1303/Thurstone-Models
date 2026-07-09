# RQ1 Full Specification — Refit stability under population growth

Status: SPEC ONLY, awaiting user go-ahead. Nothing below has been run.
Written 2026-07-10, incorporating review decisions (slope-matched scale, no
look-ahead conventions, monthly cumulative cadence, published-snapshot
validation).

## 0. Question and honest framing

When new models enter the comparison pool, whose global refit perturbs the
already-present models less — BT/logistic or lattice-Thurstone — holding the
fitting machinery, tie treatment, anchor, and data identical? Per
SCOPE_REFRAME.md this is a statement about estimation dynamics of two
gap-link models on a growing vote graph, not about Cotton's multi-entrant
field coherence.

**Primary metrics are rank-based and unit-invariant** (measured: Spearman
0.999933 between lattice fits at unit 0.1 vs 0.8) — the primary result does
not depend on any scale convention. Magnitude metrics are secondary and
convention-dependent, reported under the slope-matched scale defined in §4.

## 1. Data and fits

- Data: `clean_battle_20240814.parquet`, dedup_sampled filter, renames
  applied (chatgpt-4o-latest → chatgpt-4o-latest-2024-08-08).
- Checkpoints T_k: last instant of each calendar month (UTC),
  2023-07-31 → 2024-07-31 (13 checkpoints), plus the published-board cutoff
  2024-08-12 09:20:51 PDT as the final checkpoint. 14 total. Each fit uses
  ALL dedup battles with tstamp ≤ T_k (cumulative — mirrors how the
  published leaderboard actually refits).
- Burn-in rationale: first checkpoint 2023-07-31 gives the pool ~3 months of
  votes and guarantees the anchor (gpt-4-0613, entered 2023-06-27) is
  present with substantial votes. Runtime assertion: anchor has ≥500
  cumulative votes at every checkpoint, else abort (logged, not silently
  skipped).
- Methods, both fitted per checkpoint with `src/fit.py` per-vote MLE,
  half-tie mode (fastchat-equivalent tie treatment on both sides), anchored
  gpt-4-0613 = 0:
  - BT: LogisticLink.
  - Lattice: LatticeLink(unit=0.1) — unit fixed ACROSS ALL checkpoints
    (pure convention, no look-ahead; see §4 for why the unit choice cannot
    matter for primary metrics and how magnitudes are made comparable).
- Sensitivity reruns (magnitude metrics only): all lattice fits repeated at
  unit=0.5855 and unit=0.8. Framed strictly as robustness of the magnitude
  metric to the band convention — NOT as competing estimates of a "real"
  unit. Rationale per 2026-07-10 review: 0.5855 (fitted on the first 3
  months only) is both look-ahead-free and empirically closest to the true
  early-window tie propensity; 0.1 is the arbitrary default; 0.8
  (full-sample fit) is retained for robustness despite its look-ahead
  because it bounds the plausible band range. **The three variants differ
  because tie propensity genuinely drifts over the period (13.1% → 20.45%
  quality-tie share) — this drift finding is part of RQ1's own write-up, as
  required context for the magnitude metrics, not a cross-reference to
  RQ4.**

## 2. Validation track (free check against production history)

For each available published `elo_results_*.pkl` snapshot between 2023-07 and
2024-08 (choose the nearest snapshot within ±10 days of each checkpoint, if
any): fit BT at that pickle's own `last_updated_tstamp` (exact, not
month-end) and compare to its published ratings.

- Snapshots whose `rating_system == "bt"` (Dec 2023 onward): expect the
  Phase 2-level match (Spearman ≥ 0.999, MAE ≤ ~2 Elo pts after identical
  anchoring). A worse match on any snapshot is a data-pipeline red flag to
  investigate before trusting RQ1 numbers from that era.
- Earlier snapshots using online-Elo (pre-BT switchover): comparison is
  methodologically approximate — report Spearman only, no MAE gate.
- Dedup filter epoch: the dedup pipeline was introduced partway through the
  period; for each validation snapshot we fit BOTH dedup and no-dedup
  variants and record which matches the published board better. This maps
  the switchover empirically and documents which filter each era's
  validation uses; the RQ1 experiment track itself stays uniformly
  dedup_sampled (a fixed, declared convention).

## 3. Windows, incumbents, horizons

- Horizons: δ = 1 month (adjacent checkpoints) and δ = 3 months. "One
  month's entrants" and "a quarter's worth" are different questions; both
  reported. (δ=3 windows overlap → serial dependence; see §6.)
- Incumbents for window (T, T+δ): models with ≥1000 cumulative dedup votes
  at T. (Cumulative fits mean every model at T is also in the T+δ fit.)
  Secondary reporting: all models present at T, per standing decision #6 —
  both tables always produced, headline quotes the ≥1000 set.
- Recorded per window (interpretation covariates, not gates): number of
  entrants in (T, T+δ], their share of window votes, total vote growth,
  number of incumbents.

## 4. Scale convention for magnitude metrics (no look-ahead)

Problem: |Δθ| depends on each model's scale convention; BT fixes its scale
by the logistic link (+ 400/ln10 for Elo display). The lattice's decisive
link is steeper at a toss-up (slope at Δθ=0 is ~0.284 in the unit→0 probit
limit, rising with unit; measured 0.2899 at unit=0.1) — and the logistic's
0.25 is NOT attainable by any unit choice, so "pick the matching unit" has
no solution.

Convention adopted (functional-form-derived, zero look-ahead — the exact
analog of BT's 400/ln10 constant): report lattice magnitudes in
**slope-matched units**, θ_matched = θ · (slope_link(0)/0.25), implemented
as `LatticeLink.slope_match_factor()` (tested). After rescaling, dp/dθ at a
toss-up = 0.25 in both methods: "one reported unit near a toss-up" buys the
same win-probability change. Both methods' magnitudes additionally mapped to
Elo-equivalent display via ×400/ln10. All magnitude tables carry a
"convention-dependent" label.

Rejected alternative: unit fitted on the full sample (0.8002) — leaks
month-16 information into month-3 snapshots. Related finding to be reported
separately (already computed, `results/tables/unit_profile_first3mo.csv`):
unit fitted on the first 3 months only = 0.5855 vs 0.8002 full-sample,
tracking a genuine rise in quality-tie share (13.1% → 20.45%,
non-bothbad denominator). Tie propensity drifts over the period; this gets
its own paragraph in the writeup (relevant to RQ4 and to interpreting any
time-varying tie behavior), and it reinforces treating the unit as a fixed
convention rather than a "true" constant.

## 5. Metrics

Per (checkpoint window, horizon, method):

**Primary (rank-based, unit-invariant):**
- Kendall τ_b between incumbent ability orderings at T and T+δ
- Spearman ρ (same pairs)
- max |rank move| among incumbents; fraction of incumbents moving >5 ranks

**Secondary (magnitude, slope-matched units, convention-dependent):**
- max, mean, and 95th-percentile |Δθ_matched| among incumbents
- two alignment variants, both reported: (i) anchor-aligned
  (gpt-4-0613 = 0 both fits — production-like); (ii) median-aligned on
  incumbents (bounds the anchor's own noise contribution; pre-specified in
  src/anchoring.py docstring)

**Method comparison:** per-window paired differences (lattice − BT) on every
metric; summarized by median difference, sign consistency (how many of the
windows favor each method), and Wilcoxon signed-rank as a descriptive
index. Full per-window table always published.

## 6. Inference honesty

- δ=3 windows overlap and all windows share secular structure → no i.i.d.
  assumption holds. The headline claim will rest on sign consistency and
  effect size across windows with the full table shown, not on a single
  p-value. Wilcoxon reported as descriptive with an explicit caveat.
- No outcome-dependent choices remain open: cadence, incumbent threshold,
  horizons, metrics, alignment variants, unit conventions, and the
  validation protocol are all fixed above before the experiment runs.
- Everything deterministic (MLE fits, no sampling) → bit-reproducible given
  the lockfile.

## 7. Outputs

- `results/tables/rq1_metrics.csv` (one row per window × horizon × method ×
  incumbent-set × alignment)
- `results/tables/rq1_validation_published.csv`
- `results/tables/rq1_window_covariates.csv`
- Figures: τ_b vs time (both methods, both horizons); |Δθ_matched|
  distributions; validation MAE vs time with the dedup-epoch annotation.
- Research-log entry with any deviations flagged loudly.

## 8. Compute

14 checkpoints × 2 methods + ~14 validation BT fits (×2 filter variants) +
14-fit unit-0.8 sensitivity ≈ 70 fits, each 5–60s on the current machine —
under an hour total, no parallelism needed.

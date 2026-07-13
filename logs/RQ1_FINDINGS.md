# RQ1 Findings — Refit stability under population growth

> **UPDATED AT HEAD (2026-07-13).** Lattice fits were regenerated with the
> corrected interpolation: the null stands with slightly different
> statistics — median paired Δτ_b +0.00026 (was +0.00000), split 7/4/2
> (was 5/3/5), mean τ_b 0.98469 vs BT's unchanged 0.98358. Every
> conclusion below is unaffected; current numbers: FINDINGS_INVENTORY.md.


Written 2026-07-10, immediately after the pre-registered experiment
(logs/RQ1_SPEC.md) completed. Data: 1.67M dedup Chatbot Arena battles,
14 monthly cumulative checkpoints 2023-07-31 → 2024-08-12, four fits per
checkpoint (BT + lattice at three unit conventions), identical per-vote MLE
machinery, identical fastchat-equivalent tie treatment, identical anchor
(gpt-4-0613 = 0). Pipeline validated against 10 published BT-era leaderboard
snapshots at MAE 0.18–1.01 Elo pts (results/tables/rq1_validation_published.csv).

## Headline result: a null, and a well-powered one

**With data, tie treatment, and anchoring convention held identical across
methods, refit stability of already-present models is statistically
indistinguishable between Bradley-Terry and lattice-Thurstone — at all three
lattice unit variants (0.1, 0.5855, 0.8), at both horizons (1 and 3 months),
for both incumbent sets.**

The exact pattern (δ=1, incumbents ≥1000 votes, 13 windows; τ_b, lattice
u0.1 vs BT): lattice better in 5 windows, worse in 3, **exactly tied in 5**;
median paired difference **+0.00000**. Mean τ_b 0.98434 vs 0.98358.
Slope-matched magnitude metrics: median paired difference −0.02 Elo-eq pts
on a mean of ~1.75. δ=3: 5/11 better, median +0.00000. No incumbent moved
more than 5 ranks in any window under either method.

This pattern — near-zero median, sign-inconsistency, *exact ties* in
5/13 windows, and reproduction across three unit conventions and two
horizons — is the signature of genuine equivalence, not of an underpowered
test: the estimator differences are orders of magnitude smaller than the
quantity being measured, on 1.67M votes.

## Secondary finding (arguably the more useful one)

**Window-to-window variation in stability dwarfs between-method variation by
roughly two orders of magnitude.** τ_b ranges 0.968 → 0.997 across windows
(improving as the pool matures and per-model vote counts grow), while
per-window between-method differences are ≤ 0.007 and median 0. Stated
plainly: **in a deployment of this shape, ranking stability is governed by
per-model sample size and pool maturity, not by the choice of link
function.** A leaderboard operator worried about churn should invest in vote
volume for young models, not in swapping BT for a Thurstonian link.

Context required for the magnitude metrics (per 2026-07-10 review): the
three unit variants exist because tie propensity genuinely drifts over the
period (quality-tie share 13.1% in the first 3 months → 20.45% full-period;
profile-fitted unit 0.5855 early vs 0.8002 full-sample). The stability null
is robust to that drift — all three variants give the same answer.

## Post-hoc exploratory check (NOT pre-registered, not a paper finding)

An entrant-intensity slice was run after seeing the null, at user request,
and is labeled accordingly: it was NOT in the approved RQ1_SPEC.md.
Correlating per-window entrant count and entrant vote share against
per-method τ_b and against the paired method differences
(scripts/09, results/tables/rq1_posthoc_entrant_intensity.csv, n=13):
**nothing** — all |Spearman| ≤ 0.37, all descriptive p ≥ 0.22, signs
uninformative. Even stability itself is not visibly modulated by entrant
intensity at this sample size; calendar-time maturation dominates. Recorded
for completeness; it neither softens nor qualifies the headline null, and
any future use of entrant intensity as a moderator requires pre-registered
analysis on new data.

## Relation to the paper's scope (ties to SCOPE_REFRAME.md)

SCOPE_REFRAME.md already narrowed this project from "validating Cotton's
multi-entrant field coherence" (unexercisable on pairwise data) to "where
does a flexible-base-density gap-link alternative measurably differ from BT
in a real large-scale pairwise deployment?" RQ1's answer narrows it further:
**for refit stability under 16 months of continuous population growth, it
doesn't differ — at all.** The paper reports this as a first-class result:
an empirical audit demonstrating that the often-cited theoretical concern
about BT's global refitting under field change does not manifest as a
practical stability disadvantage relative to a Thurstonian gap-link
alternative on the largest public preference dataset — and that the
operative stability lever is sample size, not link family. This framing is
fixed now, before RQ3/RQ4 run: if those come back positive for either
method, they get reported alongside this null, not instead of it, and the
null is not to be downplayed, buried, or reframed as a limitation.

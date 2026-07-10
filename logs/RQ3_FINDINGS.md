# RQ3 Findings — Held-out calibration, BT vs lattice-Thurstone

Written 2026-07-12, immediately after the pre-registered experiment
(logs/RQ3_PREANALYSIS.md, run exactly as committed by scripts/15). Data:
13 rolling monthly windows on 1.67M dedup Chatbot Arena battles; 484,599
scoreable decisive test votes; BT + lattice at three pre-registered units;
identical fits, tie treatment, and scoring on both sides.

**Sign convention, stated once for every table and number in this
document: Δ = per-vote log-loss(BT) − log-loss(lattice), so POSITIVE ×MPD
values mean the lattice predicted better and negative values mean BT
predicted better; MPD = 4×10⁻⁴ nats/vote (the pre-registered
practical-significance threshold, §3 of the pre-analysis).**

## 1. Pre-registered headline: INCONCLUSIVE — and correctly so

The pre-committed classifier returns **inconclusive at all three lattice
units** (primary u0.5855: pooled +0.25×MPD, 95% window-cluster CI
(−0.17, +1.22)×MPD; u0.1 and u0.8 alike — results/tables/
rq3_pooled_verdicts.csv). The verdict stands as the headline result of
RQ3. It is not a failure to reach an answer: the classifier was built to
refuse a call exactly when effect size and sign pattern disagree, and they
do — **11–12 of 13 windows show sub-practical BT-leaning deltas** (−0.01
to −0.6×MPD) while a single window (2023-11-30) at **+5.8×MPD** drags the
vote-weighted pooled mean positive against the sign pattern, leaving the
CI straddling +MPD. Per the pre-registered §4.2 rule, no summary statement
of this experiment may call either method the winner.

## 2. CI-bearing strata results (pre-registered reads)

**2a. The best-case-location result (strong, citable).** The pre-analysis
(§6.2) localized the only place a practically-significant lattice-truth
effect could plausibly exist: the recent-entrant stratum of the high-noise
bin (windows starting 2023-08-31 and 2023-09-30), where lattice-truth
predicted +1.58×MPD. Measured: **−0.50×MPD, CI (−1.10, −0.06), CI-bearing
(n=4,488 ≥ pooled minimum 1,500)**. Stated as plainly as RQ1's null: *in
the one location theory predicted the best chance of a real lattice-truth
effect, the measured result is negative and CI-bearing — evidence against
lattice-truth in its own best-case location.*

**2b. Recent stratum overall.** IV-pooled across all 13 windows
(pre-registered §5.1 weighting): sub-practical BT lean with CI excluding
zero at every unit (primary: −0.12×MPD, CI (−0.18, −0.06)). Per the
pre-registered §4.2 rule this is reported as: **practically equivalent
despite a detectable directional lean (BT)** — effect size is the
governing word. A weighting note for readers: the IV rule (pre-committed
for strata) downweights the cold-start window's enormous per-vote
variance, while the vote-weighted rule (pre-committed for the
full-population headline) gives it full weight; the two pre-registered
rules therefore pull opposite directions on the same single event. Both
are reported; neither was chosen after the fact.

**2c. Everything else.** All remaining bin × stratum cells lie between
−0.5 and +0.05×MPD (results/tables/rq3_bin_pooled.csv); nothing anywhere
clears MPD in either direction. The §4.1 uncertainty-correction procedure
was not triggered (no positive verdict fired).

## 3. Two generalizable findings (first-class, jointly a methods point)

**3a. Shared miscalibration dwarfs link differences.** Reliability slopes
on held-out predictions are ≈**1.455 for all four models** (BT and all
three lattice units; results/tables/rq3_reliability.csv) — every method is
systematically *underconfident* on next-month votes by the same large
margin (realized outcome logits ≈1.45× predicted). Whatever separates
logit from lattice-probit link shapes is second-order against this shared
nonstationarity/pool-maturation effect.

**3b. Cold-start coverage is the binding constraint.** 24.8–75.8% of
decisive test votes per window were unscoreable because one side had zero
training votes (a brand-new model — per-window counts:
results/tables/rq3_unscoreable_by_window.csv).
Joint methodological point, stated plainly: **in a deployment of this
shape, the binding constraints on next-month predictability are
missing-data coverage and nonstationary drift, not the choice of link
function.** This pairs with RQ1's finding that refit *stability* is
governed by sample size and pool maturity, not link family.

## 4. Case study: the cold-start shrinkage event (n=1 — do not overread)

The single outlier window is mechanistically explained
(post-hoc diagnostic): ~80% of its +5.8×MPD delta traces to
`pplx-7b-online` (plus its 70b sibling), which entered the pool with **two
training votes** at the 2023-11-30 checkpoint. BT extrapolated it to
mid-table (θ̂=+0.432); the lattice placed it at +0.004. The model was
mass-sampled in December, performed poorly, and BT paid ≈68 nats over one
window. The mechanism: a steeper link attains the same win probabilities
at smaller ability gaps, so for near-zero-data models its MLE sits closer
to the field — **implicit shrinkage in estimation**. This is the
estimation-side complement of the prediction-side overconfidence mechanism
identified in the pre-analysis (§6): two faces of one phenomenon — steep
link × parameter uncertainty — one *hurting* the steeper link (plugin
overconfidence when predicting from noisy estimates) and one *helping* it
(shrinkage when estimating from almost no data). Both belong in the
paper's discussion. **Which direction dominates depends on the data
regime; this single event (n=1 window, n=1 model) is suggestive that the
shrinkage direction matters in genuine cold-start deployment — it is not
proof of a general advantage.**

The overconfidence face has a second, independent line of support in the
§5 decomposition — with one honest qualification. The steepest link shows
the largest pooled BT lean by a wide margin (u0.8, slope 0.337: −0.217×MPD
vs −0.106/−0.141 for the two shallower units), directionally what the
prediction-overconfidence mechanism predicts; but the pattern is **not
strictly monotone in steepness** (u0.1, slope 0.290: −0.141 exceeds
u0.5855, slope 0.324: −0.106), so this is suggestive dose-response
evidence, not a clean gradient — the two shallow units differ by little
in slope and their ordering is within noise. (An earlier summary of these
results mis-stated the pattern as monotone by reading the table in
unit-label order; corrected here.) Even so qualified, this is evidence for
the prediction-side overconfidence face, distinct from and complementary
to this case study's evidence for the estimation-side shrinkage face: two
separate pieces of support for the same steep-link × parameter-uncertainty
phenomenon, not one finding repeated.

**Cross-RQ note (added after RQ4, 2026-07-10 — and corrected the same
day):** RQ4's independent trinomial comparison is ALSO inconclusive due to
an outlier in the same 2023-11-30→12-31 window — but decomposition
(scripts/22 era diagnostics) shows RQ4's outlier is a DIFFERENT
phenomenon: pplx pairs contribute −2.8% of that window's RQ4 delta; the
driver there is tie-channel fit on near-peer pairs (RQ4_FINDINGS §5). An
earlier draft of this paragraph claimed the same event drove both RQs;
that was asserted before decomposition and is false. The shrinkage
mechanism's evidence base is THIS case study alone (n=1), plus the
structural analysis in the RQ3 pre-analysis — not two RQs.

## 5. Post-hoc decomposition: excluding extreme cold-start cases
(NOT pre-registered; labeled per the RQ1-entrant-slice precedent)

Separate, narrower question: *excluding extreme cold-start cases, are the
methods practically equivalent?* Implementation: exclude test votes where
either model has <30 training votes at the checkpoint (scripts/16). **This
decomposition does not supersede, soften, or re-adjudicate the
pre-registered INCONCLUSIVE headline in §1, which stands regardless.**

Results (results/tables/rq3_posthoc_coldstart_filter.csv; filter removes
21–76% of test votes per window, results/tables/
rq3_posthoc_filter_exclusions.csv):

| unit | pooled (×MPD) | CI (×MPD) | BT better | verdict (post-hoc) |
|---|---|---|---|---|
| 0.5855 | −0.106 | (−0.177, −0.051) | 11/13 | equivalence, sub-practical BT lean |
| 0.1 | −0.141 | (−0.190, −0.111) | 13/13 | equivalence, sub-practical BT lean |
| 0.8 | −0.217 | (−0.310, −0.157) | 12/13 | equivalence, sub-practical BT lean |

The 2023-11-30 window collapses from +5.8×MPD to +0.04×MPD under the
filter (u0.5855) — confirming the outlier was entirely the cold-start
models, and that this exploratory result matches the expectation recorded
before it was run (report of 2026-07-12: "the remaining 12-window pattern
suggests equivalence-with-sub-practical-BT-lean"). Answer to the narrower
question: **yes — away from extreme cold-start cases, the two methods are
practically equivalent, with a consistent sub-practical BT lean (largest
at the steepest unit, consistent with the §4 mechanism).** The
pre-registered INCONCLUSIVE remains the headline; this subsection answers
a different question.

## 6. Relation to the paper's scope

RQ3's contribution mirrors RQ1's: an empirical audit answer. Where RQ1
found refit stability indistinguishable, RQ3 finds held-out calibration
inconclusive at the pre-registered question (because one cold-start event
dominates), practically equivalent with a sub-practical BT lean everywhere
else, and *negative for lattice-truth in its own best-case location*. The
generalizable outputs are the two mechanisms (overconfidence/shrinkage as
faces of steep-link × parameter-uncertainty) and the deployment finding
(coverage and drift bind; link choice doesn't). None of this is softened
by, or traded against, whatever RQ4 shows.

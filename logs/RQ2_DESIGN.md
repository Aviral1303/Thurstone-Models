# RQ2 Design Doc — Testing pool-composition effects (IIA violations) in Arena votes

Status: DESIGN ONLY, not implemented. For review before any outcome data is
touched. Written 2026-07-09, after Phase 2 replication passed.

## 0. An honest reframing first

The original RQ2 hypothesis ("triples where A's win rate vs B changes with C's
presence, and lattice-Thurstone fits them better") needs a correction before we
design anything:

**For pairwise battles, both BT and 2-entrant lattice-Thurstone are gap-link
models**: P(A beats B) = F(θ_A − θ_B), where F is logistic for BT and the
winner-of-2 curve of the base density for the lattice model. Under *either*
model with static abilities, C's presence in the pool has **zero** effect on an
A-vs-B battle. So:

1. Detecting a pool-composition effect falsifies the static-single-vector
   assumption of **both** models — it is not, by itself, evidence for the
   lattice model over BT.
2. Where the models *can* differ on pairwise data is (a) the shape of F
   (logit tails vs lattice-density tails — testable via triple consistency),
   (b) tie handling (RQ4), and (c) how much a *global fit's* incumbent scores
   get strained/perturbed when the vote graph changes (RQ1).

RQ2 therefore splits into two sub-questions, and we should report them as such:

- **RQ2a (this doc's main design): does the pool's composition causally
  affect pairwise outcomes at all?** (An event-study around model entries.)
- **RQ2b: which link function F is more consistent with observed triples?**
  For dense triples (A,B,C), gap-link additivity means
  F⁻¹(p_AB) + F⁻¹(p_BC) = F⁻¹(p_AC). Both links can be tested on equal
  footing; whichever residualizes triples better wins. This is the honest
  version of "fits them better," and it needs no causal machinery — but it
  tests link shape, not IIA per se.

Mechanisms that would produce a real RQ2a effect: raters recalibrating their
quality bar as the pool improves (context-dependent judgment), and
similarity/substitution effects in how users engage. Mechanisms that would
produce a *spurious* effect: user-mix shifts (a hyped launch brings new voter
populations), prompt-mix shifts, silent version updates of A or B themselves,
and secular drift. The design's job is separating these.

## 1. Setup and notation

- Entry events: E_C = first battle time t_C of model C. ~129 candidate events
  over 2023-04 → 2024-08; usable events require C to accumulate battles
  quickly (we need an early estimate of θ_C). Expect ~60–100 usable.
- For each event E_C and each incumbent pair (A,B) active around t_C, define
  pre/post windows [t_C − w, t_C) and (t_C, t_C + w], w = 14 days (primary;
  7 and 28 as sensitivity).
- Outcome per (pair, event) cell: Δ_AB(C) = logit(p̂_post) − logit(p̂_pre),
  where p̂ = A's decisive-win share vs B (ties excluded in primary analysis;
  tie-inclusive version secondary). Cells require ≥30 decisive battles on
  each side of the event (pre-specified; report sensitivity at ≥50).

## 2. Identification: proximity-DiD, not raw before/after

A raw pooled mean of Δ_AB(C) over events is confounded by anything that moves
with calendar time (drift, user mix). The fix: **every pair active at t_C
experiences the same calendar shock, but pairs differ in how "close" C is to
them in ability.** Context-effect theories (substitution, recalibration) imply
the impact should depend on C's position relative to (A,B); calendar
confounds do not. So the identifying contrast is *within-event, across
pairs*:

  Δ_AB(C) = β₀ + β₁ · prox(C; A,B) + γ_C + ε

- γ_C: event fixed effects (absorb everything common to the moment,
  including the pure-drift component of the shock).
- prox(C; A,B): pre-specified proximity measures, computed from ratings
  fitted on data *strictly before* t_C (no peeking):
  (i) between-ness: 1{θ̂_B < θ̂_C < θ̂_A} (C slots between the pair);
  (ii) min-distance: −min(|θ̂_C − θ̂_A|, |θ̂_C − θ̂_B|);
  (iii) top-proximity: −|θ̂_C − θ̂_A| where A is the pair's stronger member.
- **Primary estimand: β₁ for measure (ii)**, pooled across events with
  precision weights (inverse sampling variance of Δ, delta-method on logits);
  (i) and (iii) secondary. Under both static models, β₁ = 0.
- β₀ (the common shock) is NOT interpretable as an IIA violation — report it,
  but it absorbs drift.

Inference: cluster-robust SEs two-way (by event and by pair) — pairs recur
across events, events hit many pairs. Additionally, **placebo permutation**:
rerun the full pipeline with pseudo-event dates drawn uniformly from periods
≥w away from any real entry (as far as density allows) and from the same
pairs; the real β₁ must clear the permutation distribution's 95th percentile.
If entry density makes clean placebo windows too scarce, fall back to
permuting which model is "C" within each real event date (keeps the calendar
structure, breaks proximity), which is the sharper null anyway.

## 3. Pre-specified confound controls

1. **Silent version drift of A/B**: primary analysis restricted to pairs
   whose members have date-pinned checkpoint names (e.g. `-20240229`,
   `-2024-04-09`); models with mutable endpoints (`-latest`, `bard-*`)
   excluded from pairs (they can still be C — C's identity only needs an
   entry date, which is exact... but θ̂_C then reflects the early version;
   acceptable). Sensitivity: include everything.
2. **Prompt/user-mix shifts**: sensitivity reruns on (a) English-only,
   (b) single-turn-only, (c) judge-level dedup (one vote per judge per pair
   per side of the event). If β₁ flips sign or loses more than half its
   magnitude across these, report as fragile.
3. **Sampling-rate changes**: when C enters, Arena over-samples C's battles,
   thinning A-vs-B counts. Affects cell precision, not bias (handled by
   weights + minimum cell size). Verified empirically in diagnostics
   (battles/day per pair around events) before outcomes are examined.
4. **Multiple events in one window**: entries cluster (2–15/month). Events
   whose windows contain another entry with prox above median get flagged;
   primary spec keeps them (event FEs absorb shared shocks), sensitivity
   drops overlapping events entirely.
5. **Multiple testing**: one primary estimand (β₁, measure ii, w=14,
   decisive-only). Everything else labeled sensitivity/exploratory in the
   paper. No outcome data gets loaded until this doc is approved.
6. **Model-family clustering** (added at review, 2026-07-09): a newly
   entered model that is ability-proximal to a pair is disproportionately
   likely to be a same-lab successor of one of its members (a GPT-4 variant
   entering near other GPT-4 variants). Family effects — shared users
   switching to the successor, correlated prompt styles, press attention to
   the lab — could load on prox(C; A,B) through channels unrelated to
   context-dependent judging. Mitigations, all pre-specified:
   (a) build a model→family map (lab + lineage, e.g. `gpt-4*` OpenAI,
       `claude-*` Anthropic, `llama-*` Meta + finetunes mapped to their base
       lab), committed to the repo before outcome data is loaded;
   (b) add a same-family indicator, 1{family(C) ∈ {family(A), family(B)}},
       as a covariate interacted with proximity in the DiD spec;
   (c) rerun the primary spec excluding all (pair, event) cells where C
       shares a family with A or B (same-family exclusion);
   (d) report β₁ under (b) and (c) alongside the primary; if the proximity
       effect shrinks by more than half or loses significance under the
       exclusion, the headline claim must be downgraded to "cannot separate
       context effects from family-succession dynamics."

## 4. Validation before touching real outcomes

- **Synthetic null check**: simulate the full battle log (same pairs, same
  timestamps, same cell sizes) from a static BT model fitted on real data;
  run the entire pipeline; β₁ must come back null at the nominal false-positive
  rate across 100 simulation seeds. This validates the estimator + inference
  code without using real outcome labels. (Clearly labeled synthetic.)
- **Power check on the same simulations**: inject known context effects of
  varying size (e.g., β₁ ∈ {0.02, 0.05, 0.1} logits) to learn the minimum
  detectable effect. If MDE >> plausible effect sizes, we say so and demote
  RQ2a to descriptive rather than pretending to test it.

## 5. Connecting back to model comparison (RQ2b + RQ1 handoff)

If RQ2a finds a real effect: neither static model is literally true; the
interesting question becomes which model's *global fit* degrades less
gracefully — that is RQ1's stability metric, plus a per-pair deviance-residual
comparison around entry events (equal parameter counts, both fitted by
per-vote MLE per the Phase 3 plan).

Independently, RQ2b (link-shape test) runs on dense triples: for all triples
with ≥100 decisive battles per leg within a stable window (no entry/exit of
A,B,C, window ≤60 days), compare additivity residuals of logit⁻¹ vs
lattice-F⁻¹. Simple, static, and directly answers "which F?". Design detail
to fix at implementation: residual metric (we propose weighted RMS of
F⁻¹(p̂_AC) − F⁻¹(p̂_AB) − F⁻¹(p̂_BC) with delta-method weights; both links get
the same denominator data).

## 6. What we commit to reporting regardless of outcome

- Number of usable events/cells and their coverage bias (which eras/models).
- β₁ with CI, permutation p, all sensitivity variants, and the synthetic
  null/power results.
- Per decision #6 (data quality): all headline numbers repeated on the
  ≥1000-votes-per-model subset.
- If nothing is detectable: RQ2a reported as a bounded null ("effects larger
  than X logits excluded"), not silently dropped.

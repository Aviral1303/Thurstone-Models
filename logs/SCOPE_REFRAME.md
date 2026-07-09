# Scope Reframe — What this project actually tests (and what it doesn't)

Written 2026-07-09 at user direction, after Phase 2. This note constrains the
framing of every downstream writeup. It changes no approved design (RQ1,
RQ2a/b, RQ3, RQ4 stay as scoped); it fixes what we are allowed to claim.

## The distinction

Cotton (2021)'s novel contribution is **simultaneous-field coherence for
multi-entrant races**: N contestants draw performances in ONE race; win
probabilities come from a single winner-of-many field distribution computed
in O(N); and scratching/adding an entrant changes the computation but not the
*kind* of model — the remaining entrants' abilities retain their meaning
without structural refitting. The headline machinery is the field-minimum
distribution over N simultaneous draws.

**Arena has no N>2 races.** Every observation is a 2-entrant battle. In a
2-entrant race the winner-of-many construction reduces to a plain two-way
comparison, and the lattice model collapses to a **gap-link model**:
P(A beats B) = F(θ_A − θ_B), where F is determined by the base density (plus
a lattice-discretization dead-heat band). The simultaneous-field coherence
property is therefore **never exercised by this data**. No result we produce
can confirm or refute Cotton's core multi-entrant claim.

## What we ARE legitimately testing

1. **RQ1 — global-refit stability under population growth.** Both methods
   maintain one global ability vector over an ever-growing set of models
   connected by overlapping pairwise votes. RQ1 asks whose *refitting
   procedure* perturbs incumbents less when new models enter the vote graph.
   This is a real and useful question for leaderboard design — but it is a
   statement about estimation dynamics of two gap-link models on a growing
   comparison graph, NOT about coherence of a simultaneous N-entrant field.
   Any stability advantage found (either direction) is attributable to link
   shape, tie handling, and fitting machinery — not to winner-of-many.

2. **RQ2b / RQ4 — link-function shape comparison.** Triple-additivity
   (RQ2b) tests whether F⁻¹ = logit or F⁻¹ = lattice-probit-like better
   linearizes observed pairwise probabilities. RQ4 tests whether the lattice
   dead-heat band predicts observed tie rates better than a Davidson
   extension. Both are comparisons of **pairwise link + tie mechanisms**
   from the Thurstonian family vs the logistic family — valuable, but they
   would come out identically for a classical Thurstone probit model with a
   tie band; nothing multi-entrant is involved.

3. **RQ2a — whether ANY static gap-link model suffices.** A detected
   pool-composition effect falsifies BT and lattice alike (see RQ2_DESIGN.md
   §0).

4. **RQ3 — held-out calibration of the two links** (with identical tie
   treatment on both sides, per the Phase 2 lesson). Again: link comparison.

## Framing rule for the paper (positioning / related-work section)

The paper must position itself as: *an empirical comparison of Thurstonian
(flexible-base-density, native-tie) gap-link models against the
Bradley-Terry/logit family on the largest public human-preference vote log,
with particular attention to refit stability under continuous population
growth* — using Cotton's lattice machinery as the computational vehicle for
the Thurstonian side (it makes the flexible-density F and its tie band cheap
to compute exactly, and it is the natural bridge if listwise LLM preference
data ever materializes). It must NOT be positioned as a validation of
Cotton's multi-entrant field-coherence result; that claim requires listwise
data (≥3 simultaneous candidates), which does not exist publicly for Arena
(Phase 1 audit). A sentence to this effect belongs in the limitations
section, and the "future work" paragraph is the honest home for the
winner-of-many story (e.g., synthetic listwise extensions, or non-Arena
domains with true multi-entrant contests such as racing data).

## Practical consequences (already consistent with approved designs)

- We stop describing the comparator as "the lattice model's field coherence"
  in logs/results; the accurate label is **"lattice-Thurstone gap-link"**
  (flexible F + native dead-heat band).
- Phase 3's implementation via a 1D p_win(Δθ)/p_tie(Δθ) lookup is not just a
  computational shortcut — it IS the model content being tested, and saying
  so plainly is the correct framing.
- If reviewers ask "why not just classical probit BT with ties?": the honest
  answer is that the lattice machinery generalizes it (arbitrary base
  densities, exact tie mass, listwise-ready) at equal computational cost;
  the flexible density is testable content (we can and should try skewed /
  heavy-tailed bases as a sensitivity axis in RQ3).

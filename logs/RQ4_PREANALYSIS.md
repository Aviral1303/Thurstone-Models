# RQ4 Pre-Analysis Commitment — Tie handling: lattice dead-heat vs Davidson-BT

Status: PRE-COMMITTED 2026-07-10 (extends logs/RQ4_DESIGN.md). Design
document ONLY: Davidson-BT is NOT yet implemented, no synthetic validation
has run, and no real-data RQ4 number of any kind has been computed. The
synthetic gates in §6 must pass, and this document must be reviewed, before
real fitting.

Sign convention (same as RQ3): Δ = per-vote log-loss(Davidson-BT) −
log-loss(lattice); positive ×MPD = lattice better.

## 0. Question and honest scope

Both models produce P(tie) from one fitted tie parameter (Davidson ν;
lattice unit) plus per-model abilities. The retired claim "lattice needs no
extra tie parameter" stays retired (RQ4_DESIGN). What is genuinely at
stake, given RQ2b showed the decisive-link shapes are indistinguishable at
real gap scales: **only the tie-curve shape** P(tie | gap) —
sech-decay-like (Davidson) vs density-overlap (lattice dead-heat) — and
whether the two fitted parameterizations imply the same effective tie band
or genuinely different ones.

## 1. Fitting design: WINDOWED, not pooled (the non-stationarity decision)

**Decision: rolling-origin windowed fitting on the RQ1/RQ3 checkpoint grid
(13 windows). Per window: abilities AND the tie parameter (ν or unit) fit
on training data (≤ T_k) only — tie parameter by 1-D profile MLE, the
identical procedure for both models — then trinomial held-out scoring on
(T_k, T_{k+1}].**

Justification (vs a single full-population fit with a drift caveat):
1. Tie share drifts 13.1% → 20.45% across the period, so a single pooled ν
   and pooled unit each average a moving target; both models would carry
   the same drift misfit, drowning exactly the shape difference RQ4 is
   supposed to measure. A pooled fit's caveat would be doing all the
   interpretive work — rejected for the same reason RQ1 rejected the
   full-sample unit.
2. Windowed fitting is look-ahead-free, matches deployment, and makes RQ4
   directly comparable to RQ1/RQ3 (same grid, same inference machinery).
3. The per-window trajectories ν̂_k and û_k are themselves a first-class
   descriptive output: they quantify the tie-band drift finding and answer
   the relabeling-vs-divergence question (§4) across time rather than at a
   single arbitrary date.

## 2. Models, outcome space, fitting

- **Davidson-BT** (to implement): gap form P(win|g) = e^{g/2} /
  (e^{g/2} + e^{-g/2} + ν), P(tie|g) = ν / (e^{g/2} + e^{-g/2} + ν).
  Per-vote trinomial MLE for abilities; ν by profile (grid + quadratic
  refine — the exact procedure `profile_lattice_unit` uses).
- **Lattice**: native trinomial mode (existing, synthetically validated);
  unit by `profile_lattice_unit` per window.
- **Outcome space**: {A wins, B wins, tie}. `tie (bothbad)` EXCLUDED per
  the standing 2026-07-09 decision; `include_both_bad=True` rerun is the
  pre-committed robustness variant. Unscoreable votes (model absent from
  training) dropped and counted, as in RQ3.

## 3. Metric, MPD, inference

- **Metric**: held-out per-vote trinomial log-loss; paired per-vote deltas
  aggregated to per-window means (Brier secondary).
- **MPD derived**: anchor = a 1-percentage-point absolute error in P(tie)
  at the observed ~20% tie level. Excess log-loss ≈ ½·Δp²·(1/p̄ + 1/(1−p̄))
  = ½·(0.01)²·(1/0.2 + 1/0.8) = 3.1e-4 → **MPD_RQ4 = 3e-4 nats/vote**.
- **Inference**: identical to RQ3 — vote-weighted pooled mean over 13
  window means, window-cluster bootstrap CI (effective N = 13, stated),
  sign-consistency bar 10/13, nested-training caveat carried over.

## 4. Verdict criteria and the tie-band comparison

- The RQ3 classifier is reused verbatim with mpd = 3e-4 (same a/b1/b1′/c
  branches, same sub-practical-lean rule §4.2 of RQ3_PREANALYSIS: any
  significant sub-MPD lean is "practically equivalent despite a detectable
  directional lean", effect size governs).
- **Mechanism/shape grade (RQ4's analog of b2)**: per window, binned
  empirical tie rate vs |ĝ| (bins fixed now: |ĝ| in [0,0.15,0.3,0.5,0.8,
  1.2,2.0], pooled across windows with vote weights) overlaid on each
  model's fitted P(tie|g). A positive verdict may reference mechanism only
  if the winner's fitted curve also tracks the empirical bins better
  (visual + binned RMS, reported).
- **Effective tie-band table (the 5a commitment)**: for each window and
  model, the gap half-widths at which P(tie|g) falls to 50% of P(tie|0),
  and where it crosses 10% and 5% absolute. If ν̂_k and û_k imply
  converging widths → the two parameterizations are relabelings of one
  phenomenon (reported as such); if they diverge → genuinely different
  shape content.

## 5. Pre-registered a-priori expectation (grounded, scripts/18)

Analytic ceiling over the EMPIRICAL gap distribution (same boundary as
RQ3 §6.1 — no outcome data): if the world were exactly lattice (unit
0.5855/0.8), the best-fit Davidson concedes only **0.10/0.19×MPD**; if
exactly Davidson (ν at the observed ~20% tie level), the best-fit lattice
concedes **≤0.31×MPD** (results/tables/rq4_tie_ceiling.csv). So, as with
RQ3, **the two tie mechanisms are near-interchangeable on real matchup
gaps; equivalence is the strongly expected verdict**, and the genuinely
informative outputs are expected to be (i) the ν̂_k/û_k drift trajectories
and (ii) the effective-tie-band convergence/divergence answer. A ≥MPD
verdict in either direction would exceed the in-family ceiling and would
therefore demand a misspecification/confound analysis (both-bad
contamination, drift-within-window, cold-start models) before any
mechanism claim — recorded now, before any number exists.

## 6. Synthetic validation gates (TO RUN after doc review, BEFORE real data)

Mirroring RQ3's structure; all through the real pipeline code:
1. **Davidson-truth world** (realistic ν, real-scale gaps, rolling entry):
   verdict must be davidson-positive or equivalence-with-davidson-lean
   matching the noise-free truth direction; ν recovery within 10% per
   window at RQ3-scale vote counts.
2. **Lattice-truth world**: symmetric requirement; unit recovery within
   10% (already demonstrated once in scripts/06, re-checked per window).
3. **Matched-curves world** (Davidson fitted to the lattice curves, per
   §5 ceiling — true difference ≈0.1×MPD): verdict must be equivalence,
   correct-direction sub-practical lean acceptable.
4. No false ≥MPD call in any world whose true effect is below MPD.

## 7. Outputs

results/tables/rq4_window_table.csv, rq4_pooled_verdict.csv,
rq4_tie_band_table.csv, rq4_tie_curve_bins.csv, parameter-trajectory
figure, findings doc per the RQ1/RQ3 template (headline verdict first,
sign convention stated, drift trajectories and tie-band answer as
first-class sections, robustness rerun labeled).

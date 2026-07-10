# RQ4 Findings — Tie handling: lattice dead-heat vs Davidson-BT

Written 2026-07-10, after the pre-registered experiment (RQ4_PREANALYSIS.md,
gates §6.1, run by scripts/21) and the user-directed post-hoc work
(scripts/22). Tables: results/tables/rq4_*.csv.

**Sign convention: Δ = per-vote trinomial log-loss(Davidson-BT) −
log-loss(lattice); positive ×MPD = lattice better; MPD_RQ4 = 3×10⁻⁴
nats/vote** (1pp tie-probability error at the observed ~20% tie level).

## 1. Pre-registered headline: INCONCLUSIVE — same shape as RQ3, same driver

Main variant (dead-heat = "tie" only; both-bad excluded): pooled
**+0.838×MPD, CI (+0.017, +2.542)**, lattice better in 10/13 windows,
verdict **inconclusive** by the pre-committed classifier. The structure
mirrors RQ3's: eleven windows within ±0.8×MPD, and the 2023-11-30 window
at **+9.55×MPD** (plus the small first window at +5.29, n=3.1k) dragging
the pooled estimate across the band edge. The headline stands as
pre-registered. **The outlier window is the same calendar window as
RQ3's, but a different phenomenon** — see §5; an initial report wrongly
assumed it was the same cold-start event, and the decomposition refuted
that.

## 2. The tie-band result: genuinely different shapes the data cannot
discriminate between — a distinct claim from "equivalent"

The 5a question (are ν and the lattice unit relabelings of one
phenomenon?) gets a precise answer: **no — the two fitted mechanisms are
DIVERGENT in shape, and the data is too thin where they diverge to say
which is right.**

- **Same at-zero mass, every window**: fitted P(tie|0) = 0.158–0.219
  (Davidson) vs 0.168–0.227 (lattice), within ~0.01 of each other in all
  13 windows.
- **Different band shapes, every window**: half-max half-width ≈2.84–2.93
  ability units (Davidson's slow sech-like decay) vs ≈1.67–1.68 (lattice's
  faster density-overlap decay) — a stable, ~1.7× structural difference.
- **The data cannot pick a winner**: binned empirical tie-rate-vs-gap
  decay lies between the two fitted curves (both undershoot near toss-ups:
  ≈0.20 fitted vs ≈0.23 empirical at |ĝ|<0.15); vote-weighted bin RMS
  0.0225 (lattice) vs 0.0237 (Davidson). The gap range where the shapes
  differ most (|ĝ| ≳ 1.2) carries <10% of votes.

Stated exactly: *the two tie parameterizations carry genuinely different
shape content — they are not relabelings — but real matchup gaps
concentrate where the shapes agree, so held-out fit quality cannot
discriminate between them on this dataset.* That is a more informative
conclusion than "equivalent."

## 3. Drift trajectories — and a three-way internal cross-validation

ν̂ rises 0.376 → 0.559 and û rises 0.594 → 0.803 across the 13 windows,
near-monotonically, both tracking the quality-tie-share drift (13.1% →
20.45%) — the RQ1-era tie-drift finding, now quantified in both models'
native parameters. **Methods-worthy internal-validity result**: the
windowed û trajectory's endpoints reproduce the two independently-computed
earlier estimates — 0.5855 (profile fit on first-3-months data, scripts/06
era) and 0.8002 (full-sample profile fit, scripts/07) — to within ~0.01.
Three separately-implemented estimation paths agree on the same drifting
quantity; this belongs in the paper's methods/validity section.

## 4. Both-bad addendum: entanglement confirmed (the sharper structural
asymmetry), with stated qualifications

The robustness variant (both-bad mapped to tie; ~35% tie mass) drove the
lattice unit to the grid ceiling (1.4) in 11/13 windows. Post-hoc
diagnostic (scripts/22, no new fitting decisions — abilities reproduced at
the already-fitted parameters):

- **(a) Ceiling provenance**: the 1.4 cap WAS pre-specified
  (RQ4_DESIGN.md, `geomspace(0.1, 1.4, 12)`) — but chosen with
  quality-tie-scale mass in mind; the synthetic gates exercised units only
  up to 1.2 at 10–20% tie mass and never stress-tested both-bad scale.
  So the per-window fitted values AT the boundary should not be over-read.
- **Capacity is not the issue**: at unit 1.4 the lattice expresses 40%
  at-zero tie mass — more than required. What drives the saturation is
  SHAPE: the lattice's narrow overlap band decays fast with gap, so
  matching both-bad-scale tie mass across the whole gap distribution
  forces extreme units (it overshoots Davidson's fitted at-zero mass,
  0.35–0.40 vs 0.33–0.35, and still pushes for more).
- **(b) Entanglement test**: coarsening the unit provably steepens the
  decisive link (slope at 0: 0.337 at u0.8 → 0.377 at u1.4), while
  Davidson's ν provably never touches its decisive link (exactly logistic
  for any ν — PAPER_NOTES.md). Measured on held-out decisive votes:
  switching from main-variant to both-bad-variant fits degrades the
  lattice's decisive log-loss by **+4.75×MPD** (mean over the 11 ceiling
  windows) vs **+2.50×MPD** for the Davidson control — the control
  isolates the common effect of refitting abilities under the changed
  outcome mix, so the lattice-specific excess is ≈**+2.2×MPD** of decisive
  discrimination lost when tie mass forces the unit coarse.
- **(c) Conclusion (per the actual evidence)**: **tie-mass capacity and
  decisive discrimination are entangled in the lattice's single-unit
  parameterization, and orthogonal in Davidson's** — structurally provable
  and diagnostically visible as ≈2× the control's degradation. This is the
  citable asymmetry between the two tie mechanisms. Qualifications stated:
  the diagnostic excess is modest, window-heterogeneous (2 ceiling windows
  show no lattice excess), and partially confounded by the never-
  stress-tested grid boundary. The original tie/both-bad category decision
  stands on its original conceptual reasoning (closeness vs absolute-bar
  judgments); this result is consistent with it but is not claimed as its
  vindication.

## 5. Post-hoc decomposition — and a correction of the assumed driver

Labeled post-hoc (<30 training votes filter, mirroring RQ3's; scripts/22
Part A): pooled **+0.861×MPD, CI (+0.017, +2.693), still inconclusive**.
Unlike RQ3 — where the filter collapsed the outlier window entirely
(+5.8 → +0.04×MPD) — RQ4's 2023-11-30 window *rises slightly* under the
filter (+9.55 → +11.5×MPD). That contradiction exposed an error worth
recording: the initial RQ4 report assumed this window's outlier was the
same pplx-7b-online cold-start event that drove RQ3's, "expressed through
the trinomial channel." **Decomposition refutes that: pplx pairs
contribute −2.8% of the window's delta** (they are also largely excluded
by the filter, which is why filtering doesn't touch the outlier).

What actually drives it: **the tie channel on near-peer, largely
same-family pairs.** Decisive votes in the window sum to −301 nats
(Davidson better); tie votes sum to **+407 nats** (lattice better, mean
+0.055/vote on 7,403 tie votes). Top contributors: gpt-4-0314 vs
gpt-4-1106-preview (+19.9), gpt-4-0314 vs gpt-4-0613 (+15.1), claude-1 vs
claude-2.1 (+13.6) — small-gap sibling pairs where the window's tie rate
ran high and the lattice's higher near-zero tie mass (P(tie|0) 0.182 vs
Davidson 0.168 that window) plus faster decay fit better. So RQ3's and
RQ4's outliers share a calendar window but not a mechanism: RQ3's is
cold-start shrinkage on the decisive channel; RQ4's is tie-curve shape on
near-peer pairs — which is §2's shape divergence briefly mattering in one
window where the data momentarily concentrated where the shapes differ.
Descriptive observation, not over-read: this window immediately precedes
the Dec-2023→Feb-2024 additivity-excess era (RQ2B_EXCESS_FINDING), so
"something unusual about that era's voting" is a shared candidate
background, unverified.

**Consequence for the cross-RQ shrinkage thread:** the cold-start
shrinkage mechanism keeps exactly ONE empirical instance (RQ3's window,
n=1) plus its structural derivation; RQ4 does NOT provide a second one.
The corrected cross-RQ statement for synthesis: *both pre-registered
calibration comparisons were driven inconclusive by single-window events
in the same calendar month, but through different channels — one
cold-start/decisive, one tie-shape/near-peer — jointly illustrating that
at real effect ceilings (≤0.31×MPD), verdicts are hostage to individual
episodes rather than systematic link differences.* That statement, not
the one-mechanism version, goes in the inventory.

## 6. Relation to scope

RQ4 completes the pre-registered empirical audit. Its contribution
mirrors RQ1/RQ3: no practical tie-handling advantage for either mechanism
at deployment-relevant effect sizes (the pre-registered ceiling ≤0.31×MPD
was respected by the data); the genuinely new content is structural — the
divergent-but-indiscriminable band shapes (§2), the quantified parameter
drift with its three-way cross-validation (§3), and the
entanglement/orthogonality asymmetry (§4).

# RQ4 Design — Tie handling: lattice dead-heat band vs Davidson-extended BT

Status: design + machinery only. The comparison itself has NOT been run.
Written 2026-07-09/10 after user review of Phase 3.

## Correction of an earlier number (do not cite)

The Phase 3 log noted that the lattice default unit (0.1) implies 2.8%
dead-heat mass at gap 0, against Arena's 18.7% observed quality-tie share.
**That was a sanity observation about an unfit default, not a preliminary
result, and must not be cited as evidence about the lattice model's tie
handling.** Davidson's ν is fit to data by MLE; comparing it against an
arbitrary default unit is fitted-vs-default and meaningless. The valid
comparison is fitted-vs-fitted, as specified below.

## The question

Arena votes include an explicit tie outcome (18.7% "tie"; a further 17.0%
"tie (bothbad)" is excluded from the dead-heat analysis per the 2026-07-09
decision — equally-poor is an absolute-bar judgment, not a closeness event;
`include_both_bad` flag exists for the robustness rerun). Both model families
can produce tie probabilities:

- **Davidson-BT**: P(tie|i,j) = ν·sqrt(p_i p_j) / (p_i + p_j + ν·sqrt(p_i p_j)),
  ν fit by MLE (custom implementation needed; choix doesn't ship it).
- **Lattice**: dead-heat mass D(g) from performance draws landing in the same
  lattice cell; band width = the unit parameter, **fit by profile MLE**
  (implemented: `src/fit.py::profile_lattice_unit`, validated on synthetic
  ground truth in `scripts/06_unit_profile_validation.py` — recovered
  0.513 vs true 0.5 with clean convex profile).

So each model has exactly one tie-related parameter fitted the same way.
"No extra parameter" is NOT a claim we can make for the lattice on pairwise
data — the unit plays exactly the role ν plays. What remains testable:

1. **Fit quality at equal parameter count**: held-out log-loss/Brier on the
   three-outcome problem {A wins, B wins, tie}, both models fitted by
   per-vote MLE (identical data, identical tie category), time-based splits
   shared with RQ3.
2. **Shape of the tie curve**: Davidson and the lattice imply different
   dependencies of P(tie) on the ability gap. Binned empirical tie rate vs
   |fitted gap| compared against each model's implied curve — this is where
   the mechanisms genuinely differ (multiplicative-mean form vs
   density-overlap form), not in parameter count.
3. **Tie-band interpretation** (user decision, Phase 2 review item 5a):
   convert fitted ν and fitted unit into comparable "effective tie-band"
   widths — the gap range within which tie probability exceeds a set of
   thresholds (e.g., half its at-zero value; absolute 5%/10%) — and report
   whether the two fitted bands agree or diverge. Agreement means the two
   parameterizations are relabelings of the same phenomenon; divergence
   means the shapes genuinely capture different tie behavior.

## Protocol (pre-specified)

- Data: dedup_sampled battles; both-bad votes dropped from the trinomial
  (both models see identical outcome space). Sensitivity: include_both_bad.
- Fitting: lattice unit profiled on the training split only (grid
  geomspace(0.1, 1.4, 12) + quadratic refinement); Davidson ν by MLE on the
  same split. Abilities refit per split for both.
- Evaluation: held-out trinomial log-loss + Brier; tie-rate-vs-gap curves;
  effective tie-band table. All headline numbers repeated on the
  ≥1000-votes-per-model subset (standing decision #6).
- The lattice unit fitted here is an RQ4 quantity. RQ1/RQ3's half-tie-mode
  comparisons: measured on real data (scripts/07, 2026-07-10), the decisive
  link's RANKING is unit-invariant (Spearman 0.999933 between unit=0.1 and
  unit=0.800 fits) but ability MAGNITUDES rescale (max|Δθ| 0.30 native
  units). Consequence: any experiment comparing fits across time (RQ1) or
  splits (RQ3) must hold the unit FIXED across all its fits; proposed
  convention — unit 0.800 (the full-data profile MLE) everywhere, with
  unit 0.1 as a sensitivity rerun of headline RQ1 metrics.

## Dependencies

- Davidson-BT MLE implementation (small custom scipy; not yet written).
- Shared time-split definitions with RQ3 (not yet fixed).

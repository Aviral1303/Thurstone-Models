# Standalone finding: temporally-localized excess dispersion in pairwise
# preference data under every tested gap-link model

Written 2026-07-10 from the RQ2b run (scripts/17, pre-registered) and its
descriptive characterization (scripts/19, labeled exploratory). Tables:
results/tables/rq2b_*.csv, rq2b_excess_*.csv.

## The result, precisely

Any gap-link model F implies additivity of inverted head-to-head rates
within a stable window: F⁻¹(p_AC) = F⁻¹(p_AB) + F⁻¹(p_BC). Across 786
dense stable triples (all legs ≥100 decisive votes, models present
throughout 60-day blocks, 1.67M-battle dedup Arena log), the standardized
additivity residual z should satisfy mean z² ≈ 1 if the link shape is
right and preferences are stable within the window.

Measured:

- **Two eras show substantial excess**: block 2023-08-22→2023-10-21
  (mean z² = 1.57; localized to its second half, ≈ late Sep–Oct 2023:
  half-block z² = 2.04 vs 0.91 in the first half) and block
  2023-12-20→2024-02-18 (mean z² = 1.65, present in BOTH 30-day halves:
  1.74 / 1.54). In these eras **13–14% of triples exceed |z| > 2**, vs
  ≈4.6% nominal. The adjacent 2024-02-18→04-18 block is moderately
  elevated (1.37, concentrated in its first half); the remaining blocks
  are consistent with calibration (z² ≈ 1.04–1.05). Machinery calibration
  was verified synthetically (true-link mean z² = 1.02–1.03 at matched
  leg sizes).
- **Cross-link-invariant**: the excess is the same under the logit link
  and all three lattice links tested (per-block means within ~0.1 of each
  other; every link elevated in the same blocks). It is therefore a
  property of the vote-generating process, not of any link-shape choice.

## Ruled out (descriptive characterization, scripts/19)

- **Dedup-pipeline transition** (empirically dated May–Jun 2024): overlaps
  only the CLEANEST block. Ruled out.
- **Entry activity**: excess blocks have FEWER model entries (7, 16) and
  LOWER entrant vote share (0.62) than the clean late blocks (21–22
  entries, 0.89–0.91 share) — anti-correlated, if anything.
- **Language mix / judge counts / votes-per-judge / tie share /
  duplicated-prompt share**: none separates excess from clean blocks
  (e.g., the clean 2023-10→12 block matches the excess blocks' language
  mix almost exactly).
- **Single-model artifact**: excess triples span the era's models near
  base rates, with only mild over-representation of legacy/low-traffic
  models (gpt-3.5-turbo-1106 in 6/16 of its triples, claude-1 4/12,
  vicuna-13b 3/10); no model or small set drives it.
- **Within-window preference drift**: a drift artifact would attenuate
  when blocks are halved; the Dec–Feb excess persists undiminished in
  both halves, and the Sep–Oct excess is itself a ~30-day episode.

## Status: open question

The excess is real, temporally localized, and unattributable to any
covariate measured in this dataset. Remaining candidate explanations:

1. **Judge-population composition shifts** within those eras (different
   user cohorts applying different preference orderings — a mixture of
   judges violates additivity even if each judge is perfectly
   gap-link-consistent);
2. **Genuine context/set effects** (votes on a pair influenced by what
   else was concurrently in the pool — the classical IIA-violation
   mechanism).

Distinguishing these requires per-judge panel structure or a natural
experiment that shifts one candidate while holding the other fixed —
which this dataset does not cleanly offer (judge IDs are anonymized
hashes with heavy churn; entry events are anti-correlated with the
excess, so entry-based designs are the wrong instrument, see
SCOPE_REFRAME.md's RQ2a descope). We report it as: **static gap-link
models of every tested shape show temporally-localized excess dispersion
(mean z² 1.6–1.7; 13–14% of triples |z|>2) that we cannot attribute to
any measured covariate** — an honest open question for future work with
better-instrumented preference data.

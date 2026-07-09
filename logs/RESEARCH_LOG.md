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

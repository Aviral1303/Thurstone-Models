# Phase 1 Data Audit — Chatbot Arena public data

Date: 2026-07-09. All numbers below computed from actual downloaded data
(`scripts/02_audit_clean_battle.py`) unless marked "documented" (taken from a
dataset card without local verification).

## Datasets found and verified

### A. `clean_battle_20240814_public.json` ← PRIMARY
- Source: `https://storage.googleapis.com/arena_external_data/public/clean_battle_20240814_public.json`
  (the file LMSYS's own leaderboard/Elo Colab notebook consumed). Verified live
  (HTTP 200, 2.13 GB), downloaded to `data/raw/`, converted to
  `data/processed/clean_battle_20240814.parquet` (47.5 MB, metadata only).
  Note: earlier-dated siblings (e.g. `clean_battle_20231206`, `20240629`) now
  return 403 — this appears to be the last public snapshot left up. Archive it.
- **1,799,991 pairwise battles**, all `anony=True` (the leaderboard-eligible kind).
- Fields: `model_a, model_b, winner, judge` (hashed user id), `turn`, `language`,
  `tstamp`, token-count metadata, category tags. **No conversation text** — fine,
  we only need outcomes.
- **Outcomes**: model_a 576,666 / model_b 580,471 / tie 336,484 / tie-bothbad
  306,370 → **35.7% ties** (18.7% "quality tie" + 17.0% "both bad").
- **Timestamps**: unix seconds with sub-second precision on 100% of rows.
  Range 2023-04-24 → 2024-08-14 (~16 months).
- **Models**: 129 distinct. Votes/model: min 1,312, median 15,565, max 182,232;
  Gini 0.554 (skewed but every model has >1.3k votes).
- **Field growth (RQ1 fuel)**: new models enter every single month, 2–15/month
  (e.g. 15 in 2024-04, 15 in 2024-05). Entry dates reconstructible per model
  from first battle timestamp.
- **Pair coverage (RQ2 fuel)**: 3,460 of 8,256 possible unordered pairs
  observed; 2,448 pairs with ≥100 battles; 947 with ≥500.
- 493,600 distinct judges; 163 languages (58% English).
- License: released publicly by LMSYS for leaderboard reproduction (their
  Colab links it). No explicit license attached to this metadata file itself;
  the related conversation releases state prompts CC-BY-4.0. Treat as
  research-use with attribution; cite the Chatbot Arena paper (arXiv:2403.04132).

### B. `lmarena-ai/arena-human-preference-140k` (HF) ← SECONDARY (2025 era)
- Verified by streaming: 135,634 rows, `model_a/model_b/winner
  (model_a|model_b|tie|both_bad)/timestamp (µs precision)/evaluation_session_id/
  language` + full conversation text. 2025-04-17 → 2025-07-24, 53 models
  (documented). Prompts CC-BY-4.0.
- Use: out-of-era robustness check of whatever we find on A. Not downloaded
  in full yet (1.6 GB, mostly conversation text we don't need).

### C. `lmarena-ai/arena-human-preference-55k` (HF / Kaggle competition)
- Documented: 57,477 rows, winner labels incl. ties, Apache-2.0. **No
  timestamps** → useless for RQ1/RQ2 pool-composition questions, and strictly
  dominated by A for RQ3/RQ4. **Dropped.**

### D. `lmsys/chatbot_arena_conversations` (HF, 33k, Apr–Jun 2023)
- Gated; subsumed by A (same era, ~10× fewer votes). **Dropped.**

### E. Published leaderboard snapshots (HF space `lmsys/chatbot-arena-leaderboard`)
- 129 `leaderboard_table_*.csv` (2023-06-19 → 2025-08-04) — model metadata only,
  no ratings. 136 `elo_results_*.pkl` — contain the **published BT ratings**
  (Arena "Elo" has been BT MLE since Dec 2023). `elo_results_20240813.pkl`
  exists = replication target one day before A's cutoff. (Pickle files —
  load in Phase 2 with care, official LMSYS artifact.)

## The five audit questions

1. **Pairwise or listwise?** Pairwise only, everywhere. No public listwise
   Arena data exists. → Plackett-Luce listwise comparison is OUT; the
   "multientrant" aspect of lattice-Thurstone enters only through the global
   stitching across overlapping pairs, not through N>2 races.
2. **Ties?** Yes, first-class: 35.7% of votes (two subtypes). RQ4 fully
   answerable, with one honest caveat: Arena's *presentation* may conflate
   "tie" and "both bad"; we'll analyze both (pooled and quality-tie-only).
3. **Timestamp granularity?** Sub-second, 100% coverage, 16 continuous months.
   Pool state at any time T is reconstructible (model entry = first battle;
   retirement = last battle). RQ1 fully answerable.
4. **Scale/skew?** 129 models, 1.8M votes, Gini 0.554, min 1,312 votes/model,
   2,448 pairs with ≥100 battles. Ample for held-out calibration (RQ3).
5. **License?** A: public research release, no explicit file license (cite
   paper); B: CC-BY-4.0 prompts; E: official artifacts. No blockers for an
   academic paper; no conversation text redistribution needed anyway.

## RQ-by-RQ verdict

- **RQ1 (stability under field change): ANSWERABLE, strongest setting.**
  16 months of continuous entry (2-15 new models/month). Design: fit both
  methods at T, refit at T+Δ after k new entrants, measure perturbation of
  incumbent scores/ranks (Kendall tau on incumbents, max |Δrating|).
  Bonus: 136 published elo_results snapshots to cross-reference.
- **RQ2 (IIA violations): ANSWERABLE with care.** 947 pairs ≥500 battles give
  many (A,B) pairs whose head-to-head rate can be estimated in windows where
  candidate C is present vs absent. Confound to control: time itself (models
  update, user mix drifts). Needs a pre-registered-style test design before
  we look — will propose one in Phase 2 review.
- **RQ3 (held-out calibration): ANSWERABLE.** Time-based splits (train ≤T,
  test >T), log-loss/Brier, with a "recent entrant" stratum. 1.8M votes is
  plenty.
- **RQ4 (ties): ANSWERABLE.** 643k tie votes. Compare BT+Davidson (fitted ν)
  vs lattice dead-heat mass (no fitted parameter) on held-out tie rates.
  Caveat: lattice tie mass on a continuous-ish density is driven by lattice
  discretization (unit size) — we must be upfront that the lattice unit acts
  as an implicit tie-width parameter; plan a sensitivity analysis over unit.

## Known modeling issues to resolve before Phase 3 (flagged, not solved)

1. **0/1 prices**: single votes can't be inverted by the monotone-interp
   calibrator. Plan: aggregate to per-matchup win rates over a window → each
   matchup = one 2-entrant race with prices = empirical rates (+ smoothing);
   this matches what BT MLE consumes, keeping the comparison fair.
2. **Sign convention**: thurstone is race-time (lower=better). Flip when
   comparing to BT.
3. **Tie mass ↔ lattice unit** coupling (see RQ4 caveat).
4. Davidson-tie BT is not in `choix` — we'll need a small custom MLE for the
   Davidson extension (scipy), while using `choix` for vanilla BT.

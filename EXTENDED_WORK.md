# Extended work: from "the lattice doesn't matter" to a second paper

Will Brodhead, July 2026. Status note for the work that grew out of revising
`paper/main.tex`, and the plan for what becomes of it. Aviral: read this before
touching anything numbered 35+.

---

## 1. Where paper 1 stands

The current paper ("When Better Choice Models Do Not Matter") is revised,
compiled, and pushed. Since the pre-revision baseline (`eee57ce`) the changes
are wording and scoping only — no theorem, verdict, or pre-registered number
moved. The substantive edits:

- The ceiling-confirmation claim is now scoped to the **decisive channel**;
  the tie channel's one excess (+0.838 vs the 0.31 ceiling) is explained as
  drift, not link shape, with a new §4 paragraph stating precisely what the
  ceiling bounds (in-family truth on the *pooled* gap design) and the three
  ways a measurement can legitimately exceed it.
- "Before fitting anything" → "before the comparison is fit" throughout, plus
  a Limitations sentence conceding the mild circularity (the gap distribution
  comes from one baseline ability fit).
- Numbers squared against the data: tie share 20.45 % full-period / 13.1 %
  first three months / 22.5 % final month ("drifts by nearly three quarters");
  the 2024-02 anomaly block quantified (1.37, "a weaker echo in a third");
  §6.1's magnitude claim corrected to 26× the mean / 4× the largest
  between-method difference; slopes 1.455 → 1.459; ridge 5.5 units;
  the 128-vs-129 model-count discrepancy explained; Appendix C's
  exact-vs-rounded MPD anchors footnoted.
- Author block: names/emails only, affiliation removed (independent work).

Deliberately unchanged after re-checking against the repo's own code: the
Fig 1b slopes (0.290–0.337) and tie half-widths (1.67–1.68) are correct as
printed (the skew-normal lattice with fractional shifting reproduces them
exactly; a plain unit-Gaussian recomputation does not, which is what first
made them look wrong).

**Open judgment calls that need Aviral's eyes** (flagged, not decided
unilaterally): the §9 circularity phrasing, the §6.3 slopes-attribution
sentence, and the venue (still TBD).

## 2. Why the extension exists

As written, paper 1's headline is "Bradley–Terry is fine, the lattice buys
nothing." Defensible, but it's a negative result about one comparison. The
paper's real asset is its *method* — effect ceilings and error budgets — and a
method earns its keep only if (a) the budget it prices can actually be
**collected**, and (b) the ceiling can actually **discriminate**, i.e. point
to a domain where a richer model IS worth building. The extension does both.
Everything below is walk-forward / out-of-sample, in the paper's own MPD units
(4e-4 nats/decisive vote unless noted), on the paper's own data plus two new
chess datasets.

## 3. Thread A — pricing the budget (script 35)

`scripts/35_extension_ceilings.py` → `results/tables/extension_budget_windows.csv`.
Thirteen monthly windows on the 2024 Arena data. For each budget line, an
**oracle** (cheats with future information → upper bound on what the line is
worth) and a **realizable** walk-forward estimator (uses only past data →
what you can actually collect).

Pooled, test-vote-weighted, in MPD units:

| line | oracle | realizable | verdict |
|---|---|---|---|
| O1/R2 recalibration (one scalar temperature) | **+12.38** (13/13 windows) | **+11.30** (10/13) | collect ~91 % of the drift line with one walk-forward scalar |
| O2 month-refit (fully dynamic abilities) | +1.55 (9/13) | — | dynamic-ability schemes are **not worth building**; drift is a temperature, not stale abilities |
| O3 cold start (oracle scores unseen models) | **+119.8** on cold votes (26–55 % of each month's decisive votes) | see Thread B | the dominant line by an order of magnitude |
| O4 tie-drift (per-window tie parameter) | +6.28 (13/13) | — | real but second-order |
| R1 time-decay weighting | — | +0.23 (6/13) | worthless |
| R4 ridge shrinkage (established models) | — | −0.00 (0/13) | worthless outside the cold-start episode the paper already covers |

Independent validation: O1's +12.38 reproduces the paper's published +12.4
recalibration gain from a completely separate implementation.

## 4. Thread B — collecting cold start (script 36)

`scripts/36_coldstart_prior.py` → `results/tables/coldstart_prior_windows.csv`.
The budget's biggest line (~+120× oracle) is by definition unreachable by any
ability estimator — a brand-new model has zero votes. But its **name** is not
zero information. At each monthly checkpoint, ridge-regress established
models' fitted abilities on four name-derived features (longest-prefix
predecessor's θ, family-best θ, log₁₀ parameter count, frontier-variant flag;
leave-one-out, walk-forward, no leakage), predict each entrant's ability
before its first vote, and score the month's new-vs-known votes with the
prior.

- Raw prior: **+12.4× MPD** pooled on cold votes (9/13 windows positive).
- Tempered prior: **+33.2×**, **12/13 windows positive**, ~**28 % of the
  oracle ceiling** — from votes the paper declared unscoreable.
- The temper κ is *estimated, not tuned*: κ maximizes the likelihood of all
  previously observed cold votes (same logic as the recalibration scalar).
  A Gaussian posterior-predictive rule tempers too little because prior
  errors are heavy-tailed — frontier releases (Claude 3, Claude 3.5) are
  leaps, not jitter. κ ranges 0.43–1.0 across 2024 checkpoints.

## 5. Thread C — the positive control: chess (scripts 37–39)

The ceiling must be falsifiable as a *decision rule*: it should say "don't
build" where richer models don't help and "build" where they do. Chess draws
provide the test, because the draw rate rises with rating **level** at fixed
gap — something no gap-only tie model (Davidson, Rao–Kupper, the lattice) can
express. Full write-up: `logs/CHESS_POSITIVE_CONTROL.md`.

| domain | data | tie/draw share | ceiling for level-dependent ties |
|---|---|---|---|
| LLM Arena | 1.67 M votes | ~20 % | **≤ 0.31×** MPD → don't build (the paper's verdict) |
| Lichess club blitz/bullet | 1.15 M games | ~4 % | **0.3–0.8×** → marginal, still don't build |
| TWIC master OTB | 181,599 games | 25.1 % | **9.3×** → build it |

At master level the draw rate nearly doubles from Elo 2250 to 2750 at fixed
small gap (0.205 → 0.389). A Davidson model with log ν = a + bL + cL²,
L = (level − 2400)/400, fits ν rising 0.50 → 0.94 across the level range.

**And the ceiling is collected out of sample** (`scripts/38_chess_oos.py`):
chronological walk-forward by TWIC issue gives **+12.0× MPD** forward
(train 1591–1638, test 1639–1650) and **+9.0×** reversed (training on only
36.5 k games). Channel decomposition: ~103 % of the gain is in the tie
channel, −3.5 % in the conditional decisive channel — exactly where the
mechanism says it must be.

This is the three-point arc the method needed: small ceiling → nothing to
collect (Arena); marginal ceiling → nothing worth building (Lichess); large
ceiling → build it, and the predicted gain materializes out of sample (TWIC).

## 6. Replication on the 2025 release (script 40)

`scripts/40_arena2025_budget.py` transports threads A and B to the 135 k-vote
2025 Arena release, weekly windows:

- Recalibration: oracle **+7.04**, realizable one-scalar **+5.96** (~85 %
  collected; 10/12 windows) — the drift result replicates.
- Month/week-refit oracle **+8.49** (12/12) — larger than 2024's +1.55, worth
  a closer look before quoting (weekly windows and a 53-model pool are not
  the 2024 regime).
- Cold start: oracle **+103.7**; pedigree prior **+51.9 raw / +52.3 tempered**
  (~**50 % of oracle**, **9/9 windows positive**). κ estimates run 0.74–1.35:
  on the checkpoint-heavy 2025 entrant mix the prior is under- rather than
  over-confident, and the ML-κ rule adapts in the right direction without
  being told.

## 7. The plan: two papers, one thesis each

Decision (Will, 2026-07-17): paper 1 ships as-is — thesis "the ceiling says
don't build, and the audit confirms it." Paper 2 gets the extension — thesis
"the budget is collectable and the ceiling discriminates." Working title:
**"When Better Choice Models Do Matter."** Skeleton:

1. Recap ceilings/budgets (cite paper 1).
2. Budget lines priced with oracle/realizable pairs (Thread A).
3. Collecting the two big lines: one-scalar recalibration (~91 % of drift)
   and the pedigree prior (~28–50 % of cold start), with the estimated-κ rule.
4. The positive control: Arena / Lichess / TWIC ceiling arc, collected OOS
   with ν(level) Davidson (Thread C).
5. 2025 replication throughout.

Paper 2 drafting is **paused until paper 1 is finished** (Will's call,
2026-07-18). All numbers above are final and committed; the draft is writing,
not research.

## 8. Reproducing everything

- Environment: repo venv `.venv` (numpy/scipy/pandas/pyarrow/ijson/pytest).
- Arena data: `data/processed/*.parquet` is gitignored (48 MB / 2 GB raw);
  rebuild with `scripts/01_convert_clean_battle.py` (2024, LMSYS
  `clean_battle_20240814.json`) and `scripts/27_convert_arena2025.py` (2025).
- Chess: TWIC issues 1591–1650 from
  `https://theweekinchess.com/zips/twicNNNNg.zip` (~410 k games before
  filters); Lichess 2013 monthly PGNs for the club-level check
  (`scripts/39_lichess_feasibility.py`). Raw PGNs are not tracked.
- Run order for the extension: 35 → 36 (imports 35) → 40; 37 → 38
  independently.
- Every table quoted here: `results/tables/extension_budget_windows.csv`,
  `coldstart_prior_windows.csv`, `arena2025_extension_budget.csv`,
  `arena2025_coldstart_prior.csv`; chess numbers in
  `logs/CHESS_POSITIVE_CONTROL.md`.

## 9. Open items

1. **Aviral**: review the paper-1 revision commits (`b04b6b1` → `66adb06`),
   especially the rescoped ceiling claims and the three judgment items in §1
   above; confirm or push back.
2. Venue for paper 1 — undecided.
3. Aviral's listed email is still `aviral.poddar@gsmc.ai`; if the point of
   dropping the affiliation is independence, consider a personal address
   (Will's is already personal).
4. The 2025 week-refit oracle (+8.49 vs 2024's +1.55) needs an explanation
   before paper 2 quotes it.
5. Pedigree-prior features are LLM-name-specific by construction; paper 2
   should say so and frame the prior as an instance of "cheap side
   information beats nothing," not a general recipe.
6. Repo has no README; worth adding once paper 1's venue is settled.

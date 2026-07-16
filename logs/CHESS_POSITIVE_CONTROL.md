# Master-level chess: effect ceiling for level-dependent ties (TWIC)

Question: is the effect ceiling for a level-dependent tie mechanism LARGE (>> 1 practical
threshold) in master-level chess, where draw rates are high? Extends the Lichess analysis
(`lichess_analysis.py`), which found a ceiling of only ~0.3-0.8x threshold at club level.

## Data

- Source: The Week In Chess (TWIC), issues 1591-1650 (60 weekly issues, ~July 2025 - June 2026),
  `https://theweekinchess.com/zips/twicNNNNg.zip`. Chosen over CCRL because TWIC worked on the
  first probe and gives human master play (OTB tournament games with FIDE ratings).
- Parsed games with both Elos and a decisive/draw result: 410,577
- Filters: |gap| <= 600, both Elos in [2200, 2900] (the >2900 tags, ~650 games, are
  engine/anomalous events mixed into TWIC): **n = 181,599**
- Overall draw share: **d = 0.2510** (vs ~0.04 on Lichess 2013 blitz/bullet pool)
- Mean rating 2430, level (average-Elo) range 2200-2828

## Q1: draw rate depends strongly on level at fixed gap

Draw rate by level bin (level = average Elo), restricted to |gap| <= 100:

| level bin | n | draw rate |
|---|---|---|
| 2200-2300 | 17,991 | 0.2053 |
| 2300-2400 | 23,861 | 0.2726 |
| 2400-2500 | 18,096 | 0.3672 |
| 2500-2600 | 11,394 | 0.3594 |
| 2600-2700 |  7,986 | 0.3552 |
| 2700-2800 |  3,236 | 0.3888 |

Draw rate nearly doubles from 2250 to 2750 at fixed (small) gap — no gap-only tie
mechanism can produce this.

## Q2: Davidson MLE fits (aggregated into 25-Elo gap x 100-Elo level cells; 215 cells)

Model: P(win/loss/draw) proportional to (e^{theta/2}, e^{-theta/2}, nu), theta = beta*gap/400 + h,
L = (level - 2400)/400.

| model | log nu | NLL (nats/game) | params |
|---|---|---|---|
| M0 gap-only | a | 0.992730 | beta=2.678, h=0.221, nu=0.7642 |
| M1 +level linear | a + b*L | 0.990502 | beta=2.691, h=0.221, b=+0.487 |
| M2 +level quadratic | a + b*L + c*L^2 | 0.990300 | beta=2.703, h=0.221, b=+0.622, c=-0.388 |

Fitted nu by level:

| level | nu (M1 linear) | nu (M2 quadratic) |
|---|---|---|
| 2200 | 0.5752 | 0.5033 |
| 2400 | 0.7340 | 0.7568 |
| 2600 | 0.9365 | 0.9372 |

Advantage of level-dependent nu over the best gap-only Davidson:

- linear: **0.002228 nats/game**
- quadratic (M0 - M2): **0.002430 nats/game**

Nonparametric sanity check: draw-probability residual (empirical - gap-only model) vs level,
over cells with n >= 200: **corr = +0.474** (gap-only model systematically under-predicts
draws at high level, over-predicts at low level).

## Practical threshold and ceiling

MPD = KL(Bern(d) || Bern(d + 0.01)) at d = 0.2510: **2.61e-04 nats**

Ceiling = advantage / MPD:

- linear: **8.5x threshold**
- quadratic: **9.3x threshold**

## Conclusion

Master-level chess is a LARGE-ceiling domain: the level-dependent tie mechanism is worth
~9 practical thresholds (~0.0024 nats/game vs an MPD of 2.6e-04), an order of magnitude
above the 0.3-0.8x found on Lichess club-level data. High draw rates plus strong
level-dependence of nu (0.50 at Elo 2200 -> 0.94 at Elo 2600) make the effect both large
and practically meaningful.

Analysis script: `master_chess_analysis.py` (run as `./venv/bin/python master_chess_analysis.py`).
Raw run log: `master_run.log`. Data: `twic/twic1591.pgn` ... `twic/twic1650.pgn`.

## Out-of-sample confirmation

Chronological walk-forward split by TWIC issue number (script: `master_chess_oos.py`,
log: `oos_run.log`). Models fitted by MLE on TRAIN cells only (same 25-Elo gap x
100-Elo level cell aggregation as the in-sample analysis), then scored on TEST games
per-game (three-outcome NLL using each test game's own continuous gap and level; no
refitting). MPD recomputed at each test set's own draw share.

### Forward: train issues 1591-1638, test issues 1639-1650

- Train n = 145,045 games (79.9%); test n = 36,554 games (20.1%, chronologically later)
- Train fits: M0 nll=0.993903 (beta=2.684, h=0.219, nu=0.7683);
  M2 nll=0.991567 (beta=2.710, h=0.220, a=-0.269, b=+0.615, c=-0.421)
- Test draw share d = 0.2463; MPD = KL(Bern(d)||Bern(d+0.01)) = 2.646e-04 nats
- Test NLL: M0 = 0.987703, M2 = 0.984528
- **Out-of-sample advantage: 0.003175 nats/game = 12.0x MPD**

### Reversed (robustness): train issues 1639-1650, test issues 1591-1638

- Train n = 36,554; test n = 145,045
- Train fits: M0 nll=0.988023 (beta=2.653, h=0.226, nu=0.7482);
  M2 nll=0.985117 (beta=2.675, h=0.227, a=-0.319, b=+0.651, c=-0.243)
- Test draw share d = 0.2521; MPD = 2.607e-04 nats
- Test NLL: M0 = 0.993847, M2 = 0.991513
- **Out-of-sample advantage: 0.002334 nats/game = 9.0x MPD**

### Where the gain comes from (channel decomposition)

Per-game NLL splits exactly into a binary draw/no-draw channel plus a conditional
win-vs-loss channel (given decisive). Advantage of M2 over M0 by channel:

| direction | tie channel | decisive channel |
|---|---|---|
| forward  | 0.003294 (103.7%) | -0.000119 (-3.7%) |
| reversed | 0.002415 (103.5%) | -0.000081 (-3.5%) |

Essentially all of the gain is in the tie-probability channel, as the mechanism
predicts; the conditional decisive channel is a small wash (M2's slightly different
beta costs a negligible amount there).

### Conclusion

The ~9.3x in-sample ceiling is fully collected out of sample: 12.0x MPD on
chronologically held-out data (and 9.0x in the reversed direction, training on only
36.5k games). The effect is stable across time and comes entirely from the
level-dependent tie probability.

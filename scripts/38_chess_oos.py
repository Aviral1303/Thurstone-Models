"""Out-of-sample (chronological walk-forward) confirmation of the level-dependent
tie-model ceiling on master chess (TWIC).

Split by TWIC issue number: train 1591-1638, test 1639-1650 (and swapped for
robustness). Fit M0 (gap-only Davidson) and M2 (log nu = a + b*L + c*L^2,
L = (level-2400)/400) by MLE on TRAIN cells (same 25-Elo gap x 100-Elo level
cell aggregation as master_chess_analysis.py). Score both on TEST games
per-game (three-outcome NLL, each game's own continuous gap and level; no
refitting). Also decompose the advantage into the binary draw/no-draw channel
and the conditional decisive (win vs loss given decisive) channel.
"""
import re, glob
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize

TWIC_DIR = "twic"

# ---------- parse, keeping issue number ----------
pat = re.compile(r'^\[(\w+) "([^"]*)"\]')
issue_pat = re.compile(r"twic(\d+)\.pgn$")
rows = []  # (issue, welo, belo, res)
files = sorted(glob.glob(f"{TWIC_DIR}/twic*.pgn"))
print(f"pgn files: {len(files)}")
for path in files:
    issue = int(issue_pat.search(path).group(1))
    we = be = res = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("["):
                m = pat.match(line)
                if not m: continue
                k, v = m.groups()
                if k == "Event":
                    we = be = res = None
                elif k == "WhiteElo":
                    we = int(v) if v.isdigit() else None
                elif k == "BlackElo":
                    be = int(v) if v.isdigit() else None
                elif k == "Result":
                    res = v
            elif line.strip():
                if we and be and res in ("1-0", "0-1", "1/2-1/2"):
                    rows.append((issue, we, be, {"1-0": 0, "0-1": 1, "1/2-1/2": 2}[res]))
                we = be = res = None

arr = np.array(rows, dtype=np.int32)
issue, welo, belo, res = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
level = (welo + belo) / 2.0
gap = (welo - belo).astype(float)
keep = (np.abs(gap) <= 600) & (welo >= 2200) & (belo >= 2200) & (welo <= 2900) & (belo <= 2900)
issue, res, level, gap = issue[keep], res[keep], level[keep], gap[keep]
print(f"filtered games (|gap|<=600, both elos in [2200,2900]): {len(res)}")

# ---------- model pieces ----------
def cell_aggregate(gap_, level_, res_):
    """Same aggregation as master_chess_analysis.py: 25-Elo gap bins, 100-Elo level bins."""
    gbin = np.clip(np.round(gap_ / 25), -24, 24).astype(int)
    lbin = np.clip(((level_ - 2200) // 100).astype(int), 0, 6)
    cells = defaultdict(lambda: np.zeros(3))
    for g, l, r in zip(gbin, lbin, res_):
        cells[(g, l)][r] += 1
    keys = sorted(cells)
    G = np.array([k[0] * 25 for k in keys], float)
    L = np.array([(k[1] * 100 + 2250 - 2400) / 400 for k in keys])
    C = np.array([cells[k] for k in keys])
    return G, L, C

def train_nll(params, G, L, C, level_terms):
    beta, h = params[0], params[1]
    lognu = params[2] + (params[3] * L if level_terms >= 1 else 0) \
                      + (params[4] * L**2 if level_terms >= 2 else 0)
    th = beta * G / 400 + h
    Z = 2 * np.cosh(th / 2) + np.exp(lognu)
    logp = np.stack([th / 2 - np.log(Z), -th / 2 - np.log(Z), lognu - np.log(Z)], axis=1)
    return -(C * logp).sum() / C.sum()

def fit(G, L, C, level_terms):
    x0 = [3.0, 0.1, 0.0, 0.0, 0.0][:3 + level_terms]
    r = minimize(lambda p: train_nll(p, G, L, C, level_terms), x0, method="Nelder-Mead",
                 options=dict(maxiter=20000, xatol=1e-8, fatol=1e-12))
    return r.x, r.fun

def per_game_logp(params, level_terms, gap_, level_, res_):
    """Per-game three-outcome log-prob plus channel decomposition.

    Returns (logp_total, logp_tie_channel, logp_decisive_channel) per game, where
    tie channel = binary draw/no-draw log-lik and decisive channel = conditional
    win-vs-loss log-lik given decisive (0 for drawn games). Total = tie + decisive.
    """
    Ls = (level_ - 2400) / 400
    lognu = params[2] + (params[3] * Ls if level_terms >= 1 else 0) \
                      + (params[4] * Ls**2 if level_terms >= 2 else 0)
    th = params[0] * gap_ / 400 + params[1]
    logZ = np.logaddexp(np.logaddexp(th / 2, -th / 2), lognu)
    lp = np.stack([th / 2 - logZ, -th / 2 - logZ, lognu - logZ], axis=1)
    logp_total = lp[np.arange(len(res_)), res_]
    log_pdraw = lp[:, 2]
    log_pnodraw = np.log1p(-np.exp(log_pdraw))
    is_draw = res_ == 2
    logp_tie = np.where(is_draw, log_pdraw, log_pnodraw)
    logp_dec = logp_total - logp_tie   # 0 for draws; log P(observed side | decisive) otherwise
    return logp_total, logp_tie, logp_dec

def kl_bern(p, q):
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

# ---------- walk-forward evaluation ----------
def evaluate(train_mask, test_mask, label):
    print(f"\n=== {label} ===")
    ntr, nte = train_mask.sum(), test_mask.sum()
    print(f"train games: {ntr}, test games: {nte}")
    G, L, C = cell_aggregate(gap[train_mask], level[train_mask], res[train_mask])
    p0, f0 = fit(G, L, C, 0)
    p2, f2 = fit(G, L, C, 2)
    print(f"train fit  M0: nll={f0:.6f}  beta={p0[0]:.3f} h={p0[1]:.3f} nu={np.exp(p0[2]):.4f}")
    print(f"train fit  M2: nll={f2:.6f}  beta={p2[0]:.3f} h={p2[1]:.3f} "
          f"a={p2[2]:+.3f} b={p2[3]:+.3f} c={p2[4]:+.3f}")

    ga, le, re_ = gap[test_mask], level[test_mask], res[test_mask]
    t0, tie0, dec0 = per_game_logp(p0, 0, ga, le, re_)
    t2, tie2, dec2 = per_game_logp(p2, 2, ga, le, re_)
    nll0, nll2 = -t0.mean(), -t2.mean()
    adv = nll0 - nll2
    adv_tie = (-tie0.mean()) - (-tie2.mean())
    adv_dec = (-dec0.mean()) - (-dec2.mean())
    d_test = np.mean(re_ == 2)
    mpd = kl_bern(d_test, d_test + 0.01)
    print(f"test draw share: {d_test:.4f}  ->  MPD = KL(Bern(d)||Bern(d+0.01)) = {mpd:.3e} nats")
    print(f"test NLL  M0: {nll0:.6f}   M2: {nll2:.6f}")
    print(f"out-of-sample advantage: {adv:.6f} nats/game = {adv / mpd:.1f}x MPD")
    print(f"decomposition: tie channel {adv_tie:.6f} ({100*adv_tie/adv:.1f}%), "
          f"decisive channel {adv_dec:.6f} ({100*adv_dec/adv:.1f}%)")
    return dict(label=label, ntr=ntr, nte=nte, p0=p0, p2=p2, f0=f0, f2=f2,
                nll0=nll0, nll2=nll2, adv=adv, adv_tie=adv_tie, adv_dec=adv_dec,
                d_test=d_test, mpd=mpd)

early = issue <= 1638
late = issue >= 1639
fwd = evaluate(early, late, "forward: train 1591-1638, test 1639-1650")
rev = evaluate(late, early, "reversed: train 1639-1650, test 1591-1638")

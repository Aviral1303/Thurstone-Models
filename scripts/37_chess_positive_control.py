"""Master-level chess (TWIC) effect-ceiling check for the level-dependent tie mechanism.

Extends lichess_analysis.py to master play (both Elos >= 2200), where draw rates
are much higher. Fits Davidson three-outcome models by MLE:
  theta = beta*gap/400 + h,   (M0) log nu = a   vs   (M2) log nu = a + b*L + c*L^2
with L = (level - 2400)/400. Reports the NLL advantage of M2 over M0 in nats/game
and compares it to the practical threshold MPD = KL(Bern(d) || Bern(d+0.01)).
"""
import re, sys, glob
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize

PGN_GLOB = sys.argv[1] if len(sys.argv) > 1 else "twic/twic*.pgn"

# ---------- parse ----------
games = []  # (welo, belo, result)
pat = re.compile(r'^\[(\w+) "([^"]*)"\]')
files = sorted(glob.glob(PGN_GLOB))
print(f"pgn files: {len(files)}")
for path in files:
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
                # moves line ends the game record (headers already seen)
                if we and be and res in ("1-0", "0-1", "1/2-1/2"):
                    games.append((we, be, res))
                we = be = res = None

games = np.array([(w, b, {"1-0": 0, "0-1": 1, "1/2-1/2": 2}[r]) for w, b, r in games],
                 dtype=np.int32)
print(f"parsed games with both elos and a result: {len(games)}")

welo, belo, res = games[:, 0], games[:, 1], games[:, 2]
level = (welo + belo) / 2.0
gap = (welo - belo).astype(float)

keep = (np.abs(gap) <= 600) & (welo >= 2200) & (belo >= 2200) & (welo <= 2900) & (belo <= 2900)
welo, belo, res, level, gap = welo[keep], belo[keep], res[keep], level[keep], gap[keep]
d = np.mean(res == 2)
print(f"after filters (|gap|<=600, both elos in [2200, 2900]): {len(res)}, draw share {d:.4f}")
print(f"mean rating {level.mean():.0f}, level range {level.min():.0f}-{level.max():.0f}")

# ---------- Q1: draw rate by level, gap-controlled ----------
print("\ndraw rate by level bin (|gap| <= 100 only):")
m = np.abs(gap) <= 100
for lo in range(2200, 2800, 100):
    sel = m & (level >= lo) & (level < lo + 100)
    if sel.sum() >= 300:
        print(f"  level {lo}-{lo+100}: n={sel.sum():6d}  draw rate {np.mean(res[sel]==2):.4f}")

# ---------- aggregate into (gap, level) cells ----------
gbin = np.clip(np.round(gap / 25), -24, 24).astype(int)          # 25-elo gap bins
lbin = np.clip(((level - 2200) // 100).astype(int), 0, 6)        # 100-elo level bins from 2200
cells = defaultdict(lambda: np.zeros(3))
for g, l, r in zip(gbin, lbin, res):
    cells[(g, l)][r] += 1
keys = sorted(cells)
G = np.array([k[0] * 25 for k in keys], float)                   # gap at bin center
L = np.array([(k[1] * 100 + 2250 - 2400) / 400 for k in keys])   # scaled level, centered 2400
C = np.array([cells[k] for k in keys])                           # counts (win, loss, draw)
N = C.sum()
print(f"\ncells: {len(keys)}, games in cells: {N:.0f}")

# ---------- Davidson fits ----------
def nll(params, level_terms):
    beta, h = params[0], params[1]
    lognu = params[2] + (params[3] * L if level_terms >= 1 else 0) \
                      + (params[4] * L**2 if level_terms >= 2 else 0)
    th = beta * G / 400 + h
    nu = np.exp(lognu)
    Z = 2 * np.cosh(th / 2) + nu
    logp = np.stack([th / 2 - np.log(Z), -th / 2 - np.log(Z), lognu - np.log(Z)], axis=1)
    return -(C * logp).sum() / N

def fit(level_terms):
    x0 = [3.0, 0.1, 0.0, 0.0, 0.0][:3 + level_terms]
    r = minimize(lambda p: nll(p, level_terms), x0, method="Nelder-Mead",
                 options=dict(maxiter=20000, xatol=1e-8, fatol=1e-12))
    return r.x, r.fun

p0, f0 = fit(0)   # M0: gap-only Davidson
p1, f1 = fit(1)   # nu depends on level (linear in log)
p2, f2 = fit(2)   # M2: quadratic

def nu_at(p, terms, elo):
    Ls = (elo - 2400) / 400
    return np.exp(p[2] + (p[3] * Ls if terms >= 1 else 0) + (p[4] * Ls**2 if terms >= 2 else 0))

print("\nDavidson fits (nll in nats/game):")
print(f"  M0 gap-only:     nll={f0:.6f}  beta={p0[0]:.3f} h={p0[1]:.3f} nu={np.exp(p0[2]):.4f}")
print(f"  + level (lin):   nll={f1:.6f}  beta={p1[0]:.3f} h={p1[1]:.3f} b={p1[3]:+.3f}")
print(f"                   nu@2200={nu_at(p1,1,2200):.4f} nu@2400={nu_at(p1,1,2400):.4f} nu@2600={nu_at(p1,1,2600):.4f}")
print(f"  M2 + level (quad): nll={f2:.6f}  beta={p2[0]:.3f} h={p2[1]:.3f} b={p2[3]:+.3f} c={p2[4]:+.3f}")
print(f"                   nu@2200={nu_at(p2,2,2200):.4f} nu@2400={nu_at(p2,2,2400):.4f} nu@2600={nu_at(p2,2,2600):.4f}")
adv1, adv2 = f0 - f1, f0 - f2
print(f"\nadvantage of level-dependent nu over best gap-only Davidson:")
print(f"  linear:    {adv1:.6f} nats/game")
print(f"  quadratic: {adv2:.6f} nats/game")

# ---------- practical threshold ----------
def kl_bern(p, q): return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
mpd = kl_bern(d, d + 0.01)
print(f"\nmaster chess tie MPD (1pp draw-prob error at draw rate {d:.3f}): {mpd:.2e} nats")
print(f"ceiling in threshold units: linear {adv1/mpd:.1f}x, quad {adv2/mpd:.1f}x")

# ---------- nonparametric sanity ----------
big = C.sum(1) >= 200
th = p0[0] * G / 400 + p0[1]; nu = np.exp(p0[2]); Z = 2 * np.cosh(th / 2) + nu
model_d = nu / Z
emp_d = C[:, 2] / C.sum(1)
resid = (emp_d - model_d)[big]
lev = L[big] * 400 + 2400
print(f"\ngap-only Davidson draw-prob residual vs level (cells with n>=200): corr = {np.corrcoef(lev, resid)[0,1]:.3f}")

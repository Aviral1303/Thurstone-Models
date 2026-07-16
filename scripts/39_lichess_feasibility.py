"""Lichess 2013-01 feasibility check for the chess positive control.

Q1: does draw rate depend on rating LEVEL at fixed gap? (If yes, no gap-only
    tie mechanism -- Davidson or lattice -- can be correct.)
Q2: how big is the effect ceiling for level-dependent ties over the best
    gap-only Davidson model, in nats/game and in threshold units?
"""
import re, sys
import numpy as np
from collections import defaultdict
from scipy.optimize import minimize

PGN = "lichess_2013-01.pgn"

# ---------- parse ----------
games = []  # (welo, belo, result, speed)
we = be = res = spd = None
pat = re.compile(r'^\[(\w+) "([^"]*)"\]')
with open(PGN, encoding="utf-8", errors="replace") as f:
    for line in f:
        if line.startswith("["):
            m = pat.match(line)
            if not m: continue
            k, v = m.groups()
            if k == "Event":
                we = be = res = None
                spd = ("bullet" if "Bullet" in v else
                       "blitz" if "Blitz" in v else
                       "slow" if ("Classical" in v or "Standard" in v or "Correspondence" in v) else "other")
            elif k == "WhiteElo":
                we = int(v) if v.isdigit() else None
            elif k == "BlackElo":
                be = int(v) if v.isdigit() else None
            elif k == "Result":
                res = v
        elif line.strip() and not line.startswith("["):
            # moves line ends the game record (headers already seen)
            if we and be and res in ("1-0", "0-1", "1/2-1/2"):
                games.append((we, be, res, spd))
            we = be = res = None

games = np.array([(w, b, {"1-0": 0, "0-1": 1, "1/2-1/2": 2}[r], {"bullet":0,"blitz":1,"slow":2,"other":3}[s])
                  for w, b, r, s in games], dtype=np.int32)
print(f"parsed games with both elos and a result: {len(games)}")

welo, belo, res, spd = games[:,0], games[:,1], games[:,2], games[:,3]
level = (welo + belo) / 2.0
gap = (welo - belo).astype(float)

keep = (np.abs(gap) <= 600) & (level >= 900) & (level <= 2300)
welo, belo, res, spd, level, gap = welo[keep], belo[keep], res[keep], spd[keep], level[keep], gap[keep]
print(f"after filters (|gap|<=600, level 900-2300): {len(res)}, draw share {np.mean(res==2):.4f}")
for s, name in [(0,"bullet"),(1,"blitz"),(2,"slow")]:
    m = spd == s
    print(f"  {name:7s}: n={m.sum():6d} draw share {np.mean(res[m]==2):.4f}")

# ---------- Q1: draw rate by level, gap-controlled ----------
print("\ndraw rate by level bin (|gap| <= 100 only, all speeds):")
m = np.abs(gap) <= 100
for lo in range(1000, 2200, 200):
    sel = m & (level >= lo) & (level < lo+200)
    if sel.sum() >= 300:
        print(f"  level {lo}-{lo+200}: n={sel.sum():6d}  draw rate {np.mean(res[sel]==2):.4f}")

print("\nsame, blitz only:")
for lo in range(1000, 2200, 200):
    sel = m & (spd==1) & (level >= lo) & (level < lo+200)
    if sel.sum() >= 300:
        print(f"  level {lo}-{lo+200}: n={sel.sum():6d}  draw rate {np.mean(res[sel]==2):.4f}")

# ---------- aggregate into (gap, level) cells ----------
gbin = np.clip(np.round(gap/25), -24, 24).astype(int)         # 25-elo gap bins
lbin = np.clip(((level-900)//100).astype(int), 0, 13)          # 100-elo level bins
cells = defaultdict(lambda: np.zeros(3))
for g, l, r in zip(gbin, lbin, res):
    cells[(g, l)][r] += 1
keys = sorted(cells)
G = np.array([k[0]*25 for k in keys], float)          # gap at bin center
L = np.array([(k[1]*100 + 950 - 1500)/400 for k in keys])  # scaled level
C = np.array([cells[k] for k in keys])                # counts (win, loss, draw)
N = C.sum()
print(f"\ncells: {len(keys)}, games in cells: {N:.0f}")

# ---------- Davidson fits ----------
def nll(params, level_terms):
    beta, h = params[0], params[1]
    lognu = params[2] + (params[3]*L if level_terms >= 1 else 0) + (params[4]*L**2 if level_terms >= 2 else 0)
    th = beta*G/400 + h
    nu = np.exp(lognu)
    Z = 2*np.cosh(th/2) + nu
    logp = np.stack([th/2 - np.log(Z), -th/2 - np.log(Z), lognu - np.log(Z)], axis=1)
    return -(C * logp).sum() / N

def fit(level_terms):
    x0 = [3.0, 0.1, -2.0, 0.0, 0.0][:3+level_terms]
    r = minimize(lambda p: nll(p, level_terms), x0, method="Nelder-Mead",
                 options=dict(maxiter=20000, xatol=1e-8, fatol=1e-12))
    return r.x, r.fun

p0, f0 = fit(0)   # gap-only Davidson
p1, f1 = fit(1)   # nu depends on level (linear in log)
p2, f2 = fit(2)   # quadratic

print("\nDavidson fits (nll in nats/game):")
print(f"  gap-only:        nll={f0:.6f}  beta={p0[0]:.3f} h={p0[1]:.3f} nu={np.exp(p0[2]):.4f}")
print(f"  + level (lin):   nll={f1:.6f}  b={p1[3]:+.3f}   nu@1100={np.exp(p1[2]+p1[3]*(-1)):.4f} nu@1500={np.exp(p1[2]):.4f} nu@1900={np.exp(p1[2]+p1[3]):.4f}")
print(f"  + level (quad):  nll={f2:.6f}")
adv1, adv2 = f0-f1, f0-f2
print(f"\nadvantage of level-dependent nu over best gap-only Davidson:")
print(f"  linear:    {adv1:.6f} nats/game")
print(f"  quadratic: {adv2:.6f} nats/game")

# ---------- thresholds for comparison ----------
d = np.mean(res == 2)
def kl_bern(p, q): return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))
mpd_tie_chess = kl_bern(d, d+0.01)
print(f"\nchess tie MPD (1pp draw-prob error at draw rate {d:.3f}): {mpd_tie_chess:.2e} nats")
print(f"ceiling estimate in chess threshold units: linear {adv1/mpd_tie_chess:.1f}x, quad {adv2/mpd_tie_chess:.1f}x")
print(f"Arena paper tie ceiling for reference: 0.31 x 3e-4 = {0.31*3e-4:.2e} nats -> chess adv is {adv2/(0.31*3e-4):.0f}x that")

# ---------- nonparametric sanity: per-cell empirical KL improvement ----------
# compare gap-only model vs empirical rates in well-populated cells
big = C.sum(1) >= 200
th = p0[0]*G/400 + p0[1]; nu = np.exp(p0[2]); Z = 2*np.cosh(th/2)+nu
model_d = nu/Z
emp_d = C[:,2]/C.sum(1)
resid = (emp_d - model_d)[big]
lev = L[big]*400+1500
print(f"\ngap-only Davidson draw-prob residual vs level (cells with n>=200): corr = {np.corrcoef(lev, resid)[0,1]:.3f}")

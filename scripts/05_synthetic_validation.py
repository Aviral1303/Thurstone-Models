"""SYNTHETIC ground-truth sanity check for the Phase 3 fitting pipeline.

*** All data in this script is simulated from known abilities. Nothing here
*** is a research result about Chatbot Arena. Its only purpose is to verify
*** that the per-vote MLE machinery recovers known ground truth before we
*** trust it on real data.

Pass criteria (pre-stated):
  A. Lattice-generated world, native-mode fit:     Spearman>=0.99, Pearson>=0.99, RMSE<=0.10
  B. Lattice-generated world, half-tie-mode fit:   Spearman>=0.99 (scale not thresholded)
  C. BT world (decisive): our logistic MLE == choix.opt_pairwise, max|dtheta|<=1e-3
  D. BT world + gap-independent random ties: our logistic half-tie fit ==
     fastchat compute_mle_elo up to affine map, max residual <= 0.1 Elo pts
  E. Misspecification (BT-generated, lattice-fitted): Spearman>=0.99 (ranks robust)

Writes results/tables/synthetic_validation.csv.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import choix  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

from bt_baseline import compute_mle_elo  # noqa: E402
from fit import fit_gaplink  # noqa: E402
from lattice_link import LatticeLink, LogisticLink  # noqa: E402

SEED = 20260709
rng = np.random.default_rng(SEED)

N_MODELS = 30
N_VOTES = 200_000

theta_true = rng.normal(0.0, 1.0, N_MODELS)
theta_true -= theta_true.mean()
models = [f"m{k:02d}" for k in range(N_MODELS)]

# Skewed matchup exposure (Arena-like): popularity weights from a Dirichlet
popularity = rng.dirichlet(np.full(N_MODELS, 0.5))


def sample_pairs(n):
    a = rng.choice(N_MODELS, size=n, p=popularity)
    b = rng.choice(N_MODELS, size=n, p=popularity)
    ok = a != b
    return a[ok], b[ok]


def battles_frame(ia, ib, outcome):
    return pd.DataFrame({
        "model_a": np.array(models)[ia],
        "model_b": np.array(models)[ib],
        "winner": outcome,
    })


checks = []


def aligned_stats(est: pd.Series, name: str):
    v = est.reindex(models).to_numpy()
    v = v - v.mean()
    sp = spearmanr(theta_true, v).statistic
    pe = pearsonr(theta_true, v).statistic
    slope = float(np.dot(v, theta_true) / np.dot(v, v))  # free link-scale
    rmse_scaled = float(np.sqrt(np.mean((slope * v - theta_true) ** 2)))
    rmse_raw = float(np.sqrt(np.mean((v - theta_true) ** 2)))
    rec = {"check": name, "spearman": sp, "pearson": pe,
           "rmse_raw": rmse_raw, "rmse_scaled": rmse_scaled, "slope": slope}
    checks.append(rec)
    print("   ", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()})
    return rec


link = LatticeLink()
logistic = LogisticLink()

# ---------- World A: lattice-generated (native trinomial outcomes) ----------
ia, ib = sample_pairs(N_VOTES)
g = theta_true[ia] - theta_true[ib]
W, D = link.p_win(g), link.p_tie(g)
u = rng.uniform(size=len(g))
outcome = np.where(u < W, "model_a", np.where(u < W + D, "tie", "model_b"))
bat_A = battles_frame(ia, ib, outcome)
print(f"[A] lattice world: {len(bat_A):,} votes, tie share {np.mean(outcome == 'tie'):.4f}")
rA = aligned_stats(fit_gaplink(bat_A, link, mode="native"), "A: lattice-gen, native fit")
assert rA["spearman"] >= 0.99 and rA["pearson"] >= 0.99 and rA["rmse_raw"] <= 0.10, "A FAILED"

print("[B] same world, half-tie fit")
rB = aligned_stats(fit_gaplink(bat_A, link, mode="half_tie"), "B: lattice-gen, half-tie fit")
assert rB["spearman"] >= 0.99, "B FAILED"

# ---------- World B: BT-generated, decisive only ----------
ia, ib = sample_pairs(N_VOTES)
g = theta_true[ia] - theta_true[ib]
p = 1.0 / (1.0 + np.exp(-g))
outcome = np.where(rng.uniform(size=len(g)) < p, "model_a", "model_b")
bat_B = battles_frame(ia, ib, outcome)

# pure MLE on both sides (no regularization) with tight tolerances, so the
# comparison isolates the likelihood/optimizer, not penalty definitions
est_log = fit_gaplink(bat_B, logistic, mode="half_tie", l2=0.0)
midx = {m: k for k, m in enumerate(models)}
pairs = [(midx[r.model_a], midx[r.model_b]) if r.winner == "model_a"
         else (midx[r.model_b], midx[r.model_a]) for r in bat_B.itertuples()]
theta_choix = choix.opt_pairwise(N_MODELS, pairs, alpha=0.0, tol=1e-10)
theta_choix = theta_choix - theta_choix.mean()
diff_choix = float(np.max(np.abs(est_log.reindex(models).to_numpy() - theta_choix)))
print(f"[C] BT world: ours vs choix max|dtheta| = {diff_choix:.2e}")
rC = aligned_stats(est_log, "C: BT-gen, our logistic fit")
checks.append({"check": "C2: ours vs choix max|dtheta|", "spearman": np.nan, "pearson": np.nan,
               "rmse_raw": diff_choix, "rmse_scaled": np.nan, "slope": np.nan})
assert diff_choix <= 1e-3, "C FAILED (optimizer mismatch vs choix)"
assert rC["spearman"] >= 0.99, "C FAILED (recovery)"

# ---------- World C: BT + gap-independent random ties, vs fastchat ----------
ia, ib = sample_pairs(N_VOTES)
g = theta_true[ia] - theta_true[ib]
p = 1.0 / (1.0 + np.exp(-g))
u = rng.uniform(size=len(g))
tie_rate = 0.30
outcome = np.where(u < tie_rate, "tie",
                   np.where(rng.uniform(size=len(g)) < p, "model_a", "model_b"))
bat_C = battles_frame(ia, ib, outcome)

# our half-tie fit must equal fastchat's compute_mle_elo up to the affine
# Elo map (same objective, different parameterization/optimizer). Ties in
# our fit: include the 'tie' label (there is no both-bad here). fastchat
# pools tie labels identically.
est_ours = fit_gaplink(bat_C, logistic, mode="half_tie", l2=0.0)
elo_fastchat = compute_mle_elo(bat_C, anchor=None, tol=1e-12, max_iter=5000)
ours_elo_units = est_ours * 400 / math.log(10)
merged = pd.DataFrame({"ours": ours_elo_units, "fc": elo_fastchat}).dropna()
merged -= merged.mean()
resid = float(np.max(np.abs(merged["ours"] - merged["fc"])))
print(f"[D] ours(half-tie) vs fastchat LR, max residual after centering = {resid:.4f} Elo pts")
checks.append({"check": "D: ours vs fastchat max resid (Elo pts)", "spearman": np.nan,
               "pearson": np.nan, "rmse_raw": resid, "rmse_scaled": np.nan, "slope": np.nan})
assert resid <= 0.1, "D FAILED (half-tie objective mismatch vs fastchat)"

# ---------- E: misspecification — BT-generated data, lattice fit ----------
print("[E] BT-generated decisive world, lattice half-tie fit (link misspec)")
rE = aligned_stats(fit_gaplink(bat_B, link, mode="half_tie"), "E: BT-gen, lattice fit")
assert rE["spearman"] >= 0.99, "E FAILED"

out = pd.DataFrame(checks)
out_path = ROOT / "results" / "tables" / "synthetic_validation.csv"
out.to_csv(out_path, index=False)
print(f"\nALL CHECKS PASSED — wrote {out_path}")

"""Ground the RQ3 pre-analysis ceiling & plugin-noise-reversal in REAL inputs.

Replaces the illustrative inputs of scripts/10-11 (generic Gaussian gap
spreads; synthetic-world noise levels) with quantities measured from
already-completed work:

  (a) Empirical gap distribution: vote-weighted ability gaps over the actual
      1.67M dedup battles, from the completed full-population fits
      (results/tables/full_fits_20240814.csv).
  (b) Real estimation noise: per-model standard errors from the Fisher
      information of REAL TRAINING fits at three regimes (early checkpoint
      2023-08-31, mid 2023-12-31, full 2024-08-12), for BOTH methods;
      BT full-fit SEs cross-checked against the published bootstrap
      variance (elo_results_20240813).

IMPORTANT BOUNDARY: no held-out real-data calibration number is computed.
All expectations below are under HYPOTHETICAL lattice-truth generative
assumptions; fits used are training-style fits of the kind already run for
RQ1. RQ3's real evaluation remains untouched.

Outputs:
  results/tables/rq3_ceiling_empirical.csv
  results/tables/rq3_fisher_se_summary.csv
  results/tables/rq3_reversal_real_inputs.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fit import VoteData, fit_gaplink  # noqa: E402
from lattice_link import LatticeLink, LogisticLink  # noqa: E402

SEED = 20260711
MPD = 4e-4
ELO_PER_NAT = 400 / np.log(10)
rng = np.random.default_rng(SEED)

# ---------------- data & completed fits ----------------
battles = pd.read_parquet(ROOT / "data" / "processed" / "clean_battle_20240814.parquet")
RENAMES = {"chatgpt-4o-latest": "chatgpt-4o-latest-2024-08-08"}
battles = battles.assign(model_a=battles["model_a"].replace(RENAMES),
                         model_b=battles["model_b"].replace(RENAMES))
dedup = battles[battles["dedup_sampled"]]

fits_full = pd.read_csv(ROOT / "results" / "tables" / "full_fits_20240814.csv", index_col="model")
theta_bt_full = fits_full["bt_theta"]
theta_lat_full = fits_full["lattice_theta"]

# ---------------- (a) empirical vote-weighted gap distribution ----------------
g_bt = (theta_bt_full.reindex(dedup["model_a"]).to_numpy()
        - theta_bt_full.reindex(dedup["model_b"]).to_numpy())
g_lat = (theta_lat_full.reindex(dedup["model_a"]).to_numpy()
         - theta_lat_full.reindex(dedup["model_b"]).to_numpy())
print(f"empirical |gap| (BT units): median={np.median(np.abs(g_bt)):.3f} "
      f"mean={np.abs(g_bt).mean():.3f} p95={np.percentile(np.abs(g_bt), 95):.3f} "
      f"max={np.abs(g_bt).max():.3f}")

# histogram weights on a fine grid for the KL sweep
grid = np.linspace(-8, 8, 401)
hist, edges = np.histogram(g_lat, bins=400, range=(-8, 8))
w_emp = hist / hist.sum()
centers = 0.5 * (edges[:-1] + edges[1:])


def min_kl_vs_logistic_empirical(gen_link):
    q = np.clip(gen_link.f_decisive(centers), 1e-12, 1 - 1e-12)

    def kl_at(s):
        p = np.clip(1 / (1 + np.exp(-centers / s)), 1e-12, 1 - 1e-12)
        return float(np.sum(w_emp * (q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p)))))

    r = minimize_scalar(kl_at, bounds=(0.2, 5.0), method="bounded")
    return r.fun, r.x


rows = []
for (u, a) in [(0.1, 0.0), (0.4, 0.0), (0.5855, 0.0), (0.8, 0.0), (1.2, 0.0),
               (0.8, 4.0), (1.2, 6.0)]:
    kl, s = min_kl_vs_logistic_empirical(LatticeLink(unit=u, skew_a=a))
    rows.append({"unit": u, "skew_a": a, "min_kl_nats": kl, "x_mpd": kl / MPD,
                 "best_logistic_scale": s})
ceiling = pd.DataFrame(rows)
print("\n(a) ceiling over EMPIRICAL gap distribution:")
print(ceiling.round(6).to_string(index=False))
ceiling.to_csv(ROOT / "results" / "tables" / "rq3_ceiling_empirical.csv", index=False)

# ---------------- (b) real Fisher SEs at three training regimes ----------------
CUTS = {
    "early_2023-08-31": pd.Timestamp("2023-09-01", tz="UTC").timestamp(),
    "mid_2023-12-31": pd.Timestamp("2024-01-01", tz="UTC").timestamp(),
    "full_2024-08-12": 1723479651.547,
}


def fisher_se(train: pd.DataFrame, link, theta: pd.Series) -> pd.Series:
    """Per-model SEs from expected information of the half-tie objective.
    I_ij = sum over weighted rows of F'(g)^2/(F(1-F)) with +/- design; gauge
    fixed by pseudo-inverse (mean-zero convention, matches our fits)."""
    data = VoteData.from_battles(train)
    from fit import _rows_half_tie
    w_idx, l_idx, wts = _rows_half_tie(data, include_both_bad=True)
    order = {m: i for i, m in enumerate(data.models)}
    th = theta.reindex(data.models).to_numpy()
    g = th[w_idx] - th[l_idx]
    eps = 1e-4
    F = np.clip(link.f_decisive(g), 1e-9, 1 - 1e-9)
    dF = (link.f_decisive(g + eps) - link.f_decisive(g - eps)) / (2 * eps)
    contrib = wts * dF ** 2 / (F * (1 - F))
    n = len(data.models)
    info = np.zeros((n, n))
    np.add.at(info, (w_idx, w_idx), contrib)
    np.add.at(info, (l_idx, l_idx), contrib)
    np.add.at(info, (w_idx, l_idx), -contrib)
    np.add.at(info, (l_idx, w_idx), -contrib)
    # The information matrix has a KNOWN null space (the constant vector —
    # translation gauge). Deflate it explicitly: add c*(11'/n) with huge c,
    # whose inverse contributes a negligible 1/(c*n) to the diagonal and
    # leaves all gauge-orthogonal directions exact. (A bare pinv inverted
    # the numerically-near-null direction and produced garbage.)
    c = 1e6 * np.trace(info) / n
    cov = np.linalg.inv(info + c * np.ones((n, n)) / n)
    return pd.Series(np.sqrt(np.clip(np.diag(cov), 0, None)), index=data.models)


se_store: dict[tuple[str, str], pd.Series] = {}
theta_store: dict[tuple[str, str], pd.Series] = {}
summary = []
for regime, cut in CUTS.items():
    train = dedup[dedup["tstamp"] <= cut]
    for method, link in (("bt", LogisticLink()), ("lattice_u0.5855", LatticeLink(unit=0.5855))):
        th = fit_gaplink(train, link, mode="half_tie", include_both_bad=True)
        se = fisher_se(train, link, th)
        se_store[(regime, method)] = se
        theta_store[(regime, method)] = th
        summary.append({"regime": regime, "method": method, "n_models": len(se),
                        "median_se_nat": float(se.median()),
                        "p90_se_nat": float(se.quantile(0.9)),
                        "max_se_nat": float(se.max()),
                        "median_se_elo": float(se.median() * ELO_PER_NAT)})
        print(f"[{regime}] {method}: median SE {se.median():.4f} nat "
              f"({se.median()*ELO_PER_NAT:.1f} Elo), p90 {se.quantile(0.9):.4f}, "
              f"max {se.max():.4f}")

# cross-check BT full SEs vs published bootstrap SD
pub = pd.read_csv(ROOT / "results" / "tables" / "published_bt_20240813.csv", index_col="model")
pub_sd_nat = np.sqrt(pub["variance"]) / ELO_PER_NAT
ours = se_store[("full_2024-08-12", "bt")]
common = pub_sd_nat.index.intersection(ours.index)
ratio = (ours.reindex(common) / pub_sd_nat.reindex(common))
from scipy.stats import spearmanr  # noqa: E402
sp = spearmanr(ours.reindex(common), pub_sd_nat.reindex(common)).statistic
print(f"\ncross-check vs published bootstrap SD: spearman={sp:.3f}, "
      f"median ratio ours/published={ratio.median():.3f}")
summary.append({"regime": "crosscheck_vs_published_bootstrap", "method": "bt",
                "n_models": len(common), "median_se_nat": float(ratio.median()),
                "p90_se_nat": float(sp), "max_se_nat": np.nan, "median_se_elo": np.nan})
pd.DataFrame(summary).to_csv(ROOT / "results" / "tables" / "rq3_fisher_se_summary.csv", index=False)

# ---------------- (b2) reversal check with REAL noise ----------------
# Hypothetical truth: lattice link at fitted abilities (per regime). Both
# methods predict with plugin theta-hat = theta + real-SE noise; BT uses the
# empirical best-fit logistic scale for that truth link. Expected delta
# (ll_bt - ll_lat) computed under the truth q, MC over noise draws.
N_VOTES_MC = 200_000
N_DRAWS = 100

rev_rows = []
for regime, cut in CUTS.items():
    train = dedup[dedup["tstamp"] <= cut]
    sub = train.sample(n=min(N_VOTES_MC, len(train)), random_state=SEED)
    th_lat = theta_store[(regime, "lattice_u0.5855")]
    se_lat = se_store[(regime, "lattice_u0.5855")]
    se_bt = se_store[(regime, "bt")]
    models = th_lat.index
    ia = sub["model_a"].map({m: i for i, m in enumerate(models)}).to_numpy()
    ib = sub["model_b"].map({m: i for i, m in enumerate(models)}).to_numpy()
    th = th_lat.to_numpy()
    for (u, a) in [(0.5855, 0.0), (1.2, 6.0)]:
        gen = LatticeLink(unit=u, skew_a=a, g_step=0.005)
        kl0, s_star = min_kl_vs_logistic_empirical(gen)
        g_true = th[ia] - th[ib]
        q = np.clip(gen.f_decisive(g_true), 1e-12, 1 - 1e-12)
        d_sum = 0.0
        se_l = se_lat.reindex(models).to_numpy()
        se_b = se_bt.reindex(models).to_numpy()
        for _ in range(N_DRAWS):
            eps_l = rng.normal(0, se_l)
            eps_b = rng.normal(0, se_b)
            g_hat_l = g_true + eps_l[ia] - eps_l[ib]
            g_hat_b = g_true + eps_b[ia] - eps_b[ib]
            from scipy.special import expit
            p_lat = np.clip(gen.f_decisive(g_hat_l), 1e-12, 1 - 1e-12)
            p_bt = np.clip(expit(g_hat_b / s_star), 1e-12, 1 - 1e-12)
            ll_lat = -(q * np.log(p_lat) + (1 - q) * np.log(1 - p_lat))
            ll_bt = -(q * np.log(p_bt) + (1 - q) * np.log(1 - p_bt))
            d_sum += float(np.mean(ll_bt - ll_lat))
        d = d_sum / N_DRAWS
        rev_rows.append({"regime": regime, "truth_unit": u, "truth_skew": a,
                         "population_kl_x_mpd": kl0 / MPD,
                         "expected_delta_x_mpd_with_real_noise": d / MPD})
        print(f"[{regime}] truth u{u}/a{a}: population ceiling {kl0/MPD:+.2f}x MPD, "
              f"with real noise {d/MPD:+.2f}x MPD "
              f"({'lattice' if d > 0 else 'BT'} favored)")

rev = pd.DataFrame(rev_rows)
rev.to_csv(ROOT / "results" / "tables" / "rq3_reversal_real_inputs.csv", index=False)
print("\nDONE")

"""RQ4 synthetic validation gates (RQ4_PREANALYSIS.md section 6).

*** All data simulated from known ground truth. Not a research result.
*** No real-data RQ4 number is computed here.

Rolling worlds mirroring the real experiment (13 windows, staggered entry,
cumulative training). Per window BOTH models are fit exactly as
pre-committed: abilities by native trinomial MLE, tie parameter (nu or
unit) by 1-D profile on training data. Held-out trinomial scoring; RQ3
classifier reused with MPD_RQ4=3e-4; delta = ll_davidson - ll_lattice
(positive = lattice better).

Worlds and gates:
  W1 Davidson-truth (nu=0.5, sd 1.0): verdict davidson_positive or
     equivalence-with-davidson-lean; nu recovery within 10% on windows
     with cumulative train >= 50k votes (interpretation of the section-6
     wording, stated).
  W2 Lattice-truth (u=0.8, sd 1.0): symmetric; unit recovery within 10%
     (same window rule).
  W3 Realistic matched world (Davidson nu=0.558, real-scale gaps sd 0.35):
     verdict equivalence.
  W4 (gate on all worlds): no >=MPD positive call anywhere — all true
     effects are sub-MPD by the section-5 ceiling.

Output: results/tables/rq4_synthetic_gates.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from davidson_link import DavidsonLink  # noqa: E402
from fit import profile_lattice_unit  # noqa: E402
from lattice_link import LatticeLink  # noqa: E402
from rq3_eval import classify, pooled_estimate  # noqa: E402
from rq4_eval import (  # noqa: E402
    MPD_RQ4,
    evaluate_window_trinomial,
    relabel_verdict,
    window_table_trinomial,
)

SEED = 20260714
DAY = 86400.0
N_MONTHS = 14
NUS = np.geomspace(0.15, 1.5, 7)
UNITS = np.geomspace(0.2, 1.2, 7)


def simulate_world(gen_link, theta_sd, rng):
    entry_month = [0] * 10
    m = 0
    while len(entry_month) < 40 and m < N_MONTHS:
        m += 1
        entry_month += [m] * int(2 + (m % 2))
    entry_month = np.array(entry_month[:40])
    theta = rng.normal(0.0, theta_sd, len(entry_month))
    theta -= theta.mean()
    names = np.array([f"m{k:02d}" for k in range(len(entry_month))])
    popularity = rng.dirichlet(np.full(len(entry_month), 0.6))
    frames = []
    for month in range(N_MONTHS):
        active = np.where(entry_month <= month)[0]
        w = popularity[active] / popularity[active].sum()
        n_votes = min(int(8000 * 1.22 ** month), 20_000)
        a = rng.choice(active, size=n_votes, p=w)
        b = rng.choice(active, size=n_votes, p=w)
        ok = a != b
        a, b = a[ok], b[ok]
        g = theta[a] - theta[b]
        W, D = gen_link.p_win(g), gen_link.p_tie(g)
        u = rng.uniform(size=len(g))
        winner = np.where(u < W, "model_a", np.where(u < W + D, "tie", "model_b"))
        frames.append(pd.DataFrame({
            "model_a": names[a], "model_b": names[b], "winner": winner,
            "tstamp": (month * 30 + rng.uniform(0, 30, len(a))) * DAY}))
    return pd.concat(frames, ignore_index=True), theta, list(names)


def true_expected_delta(theta_true, names, gen_link, fits, test):
    """Noise-free expected trinomial delta on scoreable test votes."""
    (th_d, lk_d), (th_l, lk_l) = fits
    sub = test[test["winner"].isin(("model_a", "model_b", "tie"))]
    known = sub["model_a"].isin(th_d.index) & sub["model_b"].isin(th_d.index)
    sub = sub[known]
    idx = {m: k for k, m in enumerate(names)}
    g_true = np.array([theta_true[idx[r.model_a]] - theta_true[idx[r.model_b]]
                       for r in sub.itertuples()])
    q = np.vstack([gen_link.p_win(g_true), gen_link.p_loss(g_true), gen_link.p_tie(g_true)])

    def e_ll(th, lk):
        g = th.reindex(sub["model_a"]).to_numpy() - th.reindex(sub["model_b"]).to_numpy()
        P = np.vstack([lk.p_win(g), lk.p_loss(g), lk.p_tie(g)])
        P = np.clip(P / np.maximum(P.sum(axis=0), 1e-300), 1e-12, 1)
        return -np.sum(q * np.log(P), axis=0)

    return float(np.mean(e_ll(th_d, lk_d) - e_ll(th_l, lk_l)))


def run_world(name, gen_link, theta_sd, seed, true_param=None, param_kind=None):
    rng = np.random.default_rng(seed)
    battles, theta_true, names = simulate_world(gen_link, theta_sd, rng)
    cutoffs = [(m + 1) * 30 * DAY for m in range(N_MONTHS)]
    results, true_deltas, recov = [], [], []
    for k in range(N_MONTHS - 1):
        train = battles[battles["tstamp"] <= cutoffs[k]]
        test = battles[(battles["tstamp"] > cutoffs[k]) & (battles["tstamp"] <= cutoffs[k + 1])]
        # g_step=0.005: the finer curve grid is the established remedy for
        # L-BFGS stalls at interpolation kinks (see scripts/10 World P2).
        nu_hat, th_dav, _ = profile_lattice_unit(
            train, NUS, make_link=lambda v: DavidsonLink(nu=float(v)), mode="native")
        u_hat, th_lat, _ = profile_lattice_unit(
            train, UNITS, make_link=lambda v: LatticeLink(unit=float(v), g_step=0.005),
            mode="native")
        lk_d, lk_l = DavidsonLink(nu=nu_hat), LatticeLink(unit=u_hat, g_step=0.005)
        results.append(evaluate_window_trinomial(th_dav, lk_d, th_lat, lk_l, test, f"w{k:02d}"))
        true_deltas.append(true_expected_delta(theta_true, names, gen_link,
                                               ((th_dav, lk_d), (th_lat, lk_l)), test))
        recov.append({"window": k, "cum_votes": len(train), "nu_hat": nu_hat, "u_hat": u_hat})
    wt = window_table_trinomial(results)
    pooled = pooled_estimate(wt)
    verdict = classify(wt, pooled, mpd=MPD_RQ4)
    verdict["verdict"] = relabel_verdict(verdict["verdict"])
    true_eff = float(np.average(true_deltas, weights=wt["n"]))
    rec = pd.DataFrame(recov)
    param_ok = None
    if true_param is not None:
        col = "nu_hat" if param_kind == "nu" else "u_hat"
        big = rec[rec["cum_votes"] >= 50_000]
        errs = (big[col] - true_param).abs() / true_param
        param_ok = bool((errs <= 0.10).all())
        print(f"  {param_kind} recovery (cum>=50k windows): "
              f"{[round(v, 3) for v in big[col]]} vs true {true_param} -> "
              f"max rel err {errs.max():.3f} ({'OK' if param_ok else 'FAIL'})")
    print(f"[{name}] true delta {true_eff/MPD_RQ4:+.2f}x MPD | realized "
          f"{pooled['pooled']/MPD_RQ4:+.2f}x MPD CI "
          f"({pooled['ci_lo']/MPD_RQ4:+.2f},{pooled['ci_hi']/MPD_RQ4:+.2f}) | "
          f"verdict={verdict['verdict']} {verdict.get('directional_note', '')}")
    return {"world": name, "true_x_mpd": true_eff / MPD_RQ4,
            "pooled_x_mpd": pooled["pooled"] / MPD_RQ4,
            "ci_lo_x_mpd": pooled["ci_lo"] / MPD_RQ4, "ci_hi_x_mpd": pooled["ci_hi"] / MPD_RQ4,
            "verdict": verdict["verdict"], "note": verdict.get("directional_note", ""),
            "param_recovery_ok": param_ok}


rows = []
rows.append(run_world("W1 davidson-truth nu0.5", DavidsonLink(nu=0.5), 1.0,
                      SEED, true_param=0.5, param_kind="nu"))
rows.append(run_world("W2 lattice-truth u0.8", LatticeLink(unit=0.8), 1.0,
                      SEED + 1, true_param=0.8, param_kind="unit"))
rows.append(run_world("W3 realistic matched (davidson nu0.558, sd0.35)",
                      DavidsonLink(nu=0.558), 0.35, SEED + 2))

out = pd.DataFrame(rows)
out.to_csv(ROOT / "results" / "tables" / "rq4_synthetic_gates.csv", index=False)

print("\n===== GATES =====")
g1 = (rows[0]["verdict"] == "davidson_positive"
      or (rows[0]["verdict"] == "equivalence" and "bt" in rows[0]["note"])) \
     and rows[0]["param_recovery_ok"]
g2 = (rows[1]["verdict"] == "lattice_positive"
      or (rows[1]["verdict"] == "equivalence" and "lattice" in rows[1]["note"])) \
     and rows[1]["param_recovery_ok"]
g3 = rows[2]["verdict"] == "equivalence"
g4 = all(abs(r["true_x_mpd"]) >= 1.0 or r["verdict"] in
         ("equivalence", "inconclusive") for r in rows)
for label, ok in (("G1 davidson-truth: direction + nu recovery", g1),
                  ("G2 lattice-truth: direction + unit recovery", g2),
                  ("G3 realistic matched: equivalence", g3),
                  ("G4 no false >=MPD call on sub-MPD truth", g4)):
    print(f"{label}: {'PASS' if ok else 'FAIL'}")
if not (g1 and g2 and g3 and g4):
    sys.exit(1)
print("ALL GATES PASS")

"""RQ4 post-hoc work (user review 2026-07-10). Two labeled parts:

PART A — POST-HOC EXPLORATORY decomposition of the main RQ4 variant,
  mirroring RQ3's: exclude test votes where either model has <30 training
  votes. Same framing: answers the narrower "away from extreme cold-start,
  are the tie mechanisms equivalent?" — the pre-registered INCONCLUSIVE
  headline stands regardless.

PART B — DIAGNOSTIC characterization of the both-bad grid-ceiling
  saturation (no new fitting decisions: abilities are reproduced
  deterministically at the ALREADY-FITTED per-window parameters recorded
  in rq4_param_trajectories.csv; no grid extension). Question: is tie-mass
  capacity ENTANGLED with decisive discrimination in the lattice's
  single-unit parameterization? Test: held-out DECISIVE-only conditional
  log-loss on the same decisive votes, comparing each model's main-variant
  fit vs its both-bad-variant fit. Davidson is the control (its decisive
  link is exactly logistic for any nu — proven orthogonality), so any
  lattice-specific degradation when the unit is forced coarse is the
  entanglement signature.

Outputs: results/tables/rq4_posthoc_coldstart_filter.csv,
         rq4_bothbad_entanglement.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from davidson_link import DavidsonLink  # noqa: E402
from fit import fit_gaplink  # noqa: E402
from lattice_link import LatticeLink  # noqa: E402
from rq3_eval import classify, pooled_estimate  # noqa: E402
from rq4_eval import (  # noqa: E402
    MPD_RQ4,
    evaluate_window_trinomial,
    relabel_verdict,
    window_table_trinomial,
)

FINAL_CUTOFF = 1723479651.547
MIN_TRAIN_VOTES = 30

battles = pd.read_parquet(ROOT / "data" / "processed" / "clean_battle_20240814.parquet")
RENAMES = {"chatgpt-4o-latest": "chatgpt-4o-latest-2024-08-08"}
battles = battles.assign(model_a=battles["model_a"].replace(RENAMES),
                         model_b=battles["model_b"].replace(RENAMES))
dedup = battles[battles["dedup_sampled"]]

month_ends = pd.date_range("2023-08-01", "2024-08-01", freq="MS", tz="UTC")
cutoffs = [t.timestamp() for t in month_ends] + [FINAL_CUTOFF]
labels = [str((t - pd.Timedelta(days=1)).date()) for t in month_ends] + ["2024-08-12"]

traj = pd.read_csv(ROOT / "results" / "tables" / "rq4_param_trajectories.csv")
params = {(r.variant, r.window): (r.nu_hat, r.unit_hat) for r in traj.itertuples()}

# reproduce all fits deterministically at recorded parameters
fits = {}
for k in range(len(cutoffs) - 1):
    train = dedup[dedup["tstamp"] <= cutoffs[k]]
    for variant, include_bb in (("main_tie_only", False), ("robustness_bothbad", True)):
        nu_hat, u_hat = params[(variant, labels[k])]
        lk_d = DavidsonLink(nu=nu_hat)
        lk_l = LatticeLink(unit=u_hat, g_step=0.005)
        th_d = fit_gaplink(train, lk_d, mode="native", include_both_bad=include_bb)
        th_l = fit_gaplink(train, lk_l, mode="native", include_both_bad=include_bb)
        fits[(variant, labels[k])] = (th_d, lk_d, th_l, lk_l)
    print(f"[{labels[k]}] fits reproduced")

# ---------------- PART A: cold-start filter on main variant ----------------
print("\n=== PART A: post-hoc <30-training-votes decomposition (main variant) ===")
results = []
for k in range(len(cutoffs) - 1):
    train = dedup[dedup["tstamp"] <= cutoffs[k]]
    test = dedup[(dedup["tstamp"] > cutoffs[k]) & (dedup["tstamp"] <= cutoffs[k + 1])]
    votes = pd.concat([train["model_a"], train["model_b"]]).value_counts()
    ok = votes[votes >= MIN_TRAIN_VOTES].index
    test_f = test[test["model_a"].isin(ok) & test["model_b"].isin(ok)]
    th_d, lk_d, th_l, lk_l = fits[("main_tie_only", labels[k])]
    results.append(evaluate_window_trinomial(th_d, lk_d, th_l, lk_l, test_f, labels[k]))

wt = window_table_trinomial(results)
pooled = pooled_estimate(wt)
verdict = classify(wt, pooled, mpd=MPD_RQ4)
verdict["verdict"] = relabel_verdict(verdict["verdict"])
print(wt.round(6).to_string(index=False))
print(f"pooled {pooled['pooled']/MPD_RQ4:+.3f}x MPD "
      f"CI ({pooled['ci_lo']/MPD_RQ4:+.3f},{pooled['ci_hi']/MPD_RQ4:+.3f}); "
      f"lattice better {verdict['n_lattice_better']}/13; "
      f"verdict={verdict['verdict']} {verdict.get('directional_note', '')}")
wt.assign(pooled_x_mpd=pooled["pooled"] / MPD_RQ4,
          ci_lo_x_mpd=pooled["ci_lo"] / MPD_RQ4, ci_hi_x_mpd=pooled["ci_hi"] / MPD_RQ4,
          verdict=verdict["verdict"], note=verdict.get("directional_note", "")).to_csv(
    ROOT / "results" / "tables" / "rq4_posthoc_coldstart_filter.csv", index=False)

# ---------------- PART B: both-bad entanglement diagnostic ----------------
print("\n=== PART B: decisive-only log-loss, main-fit vs bothbad-fit (entanglement) ===")
rows = []
for k in range(len(cutoffs) - 1):
    test = dedup[(dedup["tstamp"] > cutoffs[k]) & (dedup["tstamp"] <= cutoffs[k + 1])]
    dec = test[test["winner"].isin(("model_a", "model_b"))]
    rec = {"window": labels[k],
           "unit_main": params[("main_tie_only", labels[k])][1],
           "unit_bothbad": params[("robustness_bothbad", labels[k])][1],
           "at_ceiling": params[("robustness_bothbad", labels[k])][1] >= 1.399}
    for variant in ("main_tie_only", "robustness_bothbad"):
        th_d, lk_d, th_l, lk_l = fits[(variant, labels[k])]
        for model, th, lk in (("dav", th_d, lk_d), ("lat", th_l, lk_l)):
            known = dec["model_a"].isin(th.index) & dec["model_b"].isin(th.index)
            d2 = dec[known]
            g = th.reindex(d2["model_a"]).to_numpy() - th.reindex(d2["model_b"]).to_numpy()
            p = np.clip(lk.f_decisive(g), 1e-12, 1 - 1e-12)
            y = (d2["winner"] == "model_a").to_numpy(dtype=float)
            ll = float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
            rec[f"ll_dec_{model}_{'bb' if variant.startswith('rob') else 'main'}"] = ll
    rec["lat_degradation"] = rec["ll_dec_lat_bb"] - rec["ll_dec_lat_main"]
    rec["dav_degradation"] = rec["ll_dec_dav_bb"] - rec["ll_dec_dav_main"]
    rows.append(rec)
    print(f"[{labels[k]}] unit {rec['unit_main']:.3f}->{rec['unit_bothbad']:.3f}"
          f"{' CEIL' if rec['at_ceiling'] else '     '} "
          f"lat dec-LL delta {rec['lat_degradation']/MPD_RQ4:+7.2f}x MPD | "
          f"dav {rec['dav_degradation']/MPD_RQ4:+7.2f}x MPD")

ent = pd.DataFrame(rows)
ent.to_csv(ROOT / "results" / "tables" / "rq4_bothbad_entanglement.csv", index=False)
ceil = ent[ent["at_ceiling"]]
print(f"\nceiling windows (n={len(ceil)}): lattice decisive degradation "
      f"mean {ceil['lat_degradation'].mean()/MPD_RQ4:+.2f}x MPD; "
      f"Davidson control {ceil['dav_degradation'].mean()/MPD_RQ4:+.2f}x MPD")
print("DONE")

"""RQ4 REAL-DATA experiment — run exactly per logs/RQ4_PREANALYSIS.md
(gates passed scripts/20; user go-ahead 2026-07-10).

13 rolling windows on the RQ1/RQ3 checkpoint grid. Per window, BOTH models
fit on training data only: abilities by native trinomial per-vote MLE, tie
parameter by 1-D profile (Davidson nu; lattice unit) — identical procedure.
Held-out trinomial log-loss on {A wins, B wins, tie}; both-bad excluded
(main) / mapped to tie (pre-committed robustness variant). RQ3 classifier
verbatim with MPD_RQ4=3e-4; delta = ll_davidson - ll_lattice (positive =
lattice better). Drift trajectories nu_hat_k / unit_hat_k and effective
tie-band table are first-class outputs.

Outputs: results/tables/rq4_window_table.csv, rq4_pooled_verdict.csv,
rq4_tie_band_table.csv, rq4_tie_curve_bins.csv, rq4_param_trajectories.csv,
results/figures/rq4_trajectories.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from davidson_link import DavidsonLink  # noqa: E402
from fit import profile_lattice_unit  # noqa: E402
from lattice_link import LatticeLink  # noqa: E402
from rq3_eval import classify, pooled_estimate  # noqa: E402
from rq4_eval import (  # noqa: E402
    MPD_RQ4,
    evaluate_window_trinomial,
    predict_and_score_trinomial,
    relabel_verdict,
    window_table_trinomial,
)

FINAL_CUTOFF = 1723479651.547
NUS = np.geomspace(0.15, 1.5, 8)
UNITS = np.geomspace(0.2, 1.4, 8)

battles = pd.read_parquet(ROOT / "data" / "processed" / "clean_battle_20240814.parquet")
RENAMES = {"chatgpt-4o-latest": "chatgpt-4o-latest-2024-08-08"}
battles = battles.assign(model_a=battles["model_a"].replace(RENAMES),
                         model_b=battles["model_b"].replace(RENAMES))
dedup = battles[battles["dedup_sampled"]]

month_ends = pd.date_range("2023-08-01", "2024-08-01", freq="MS", tz="UTC")
cutoffs = [t.timestamp() for t in month_ends] + [FINAL_CUTOFF]
labels = [str((t - pd.Timedelta(days=1)).date()) for t in month_ends] + ["2024-08-12"]

GAP_BINS = [0.0, 0.15, 0.3, 0.5, 0.8, 1.2, 2.0]


def band_widths(link):
    """Effective tie-band half-widths (ability units): 50% of P(tie|0),
    and absolute 10% / 5% crossings (NaN if never above)."""
    gs = np.linspace(0, 8, 4001)
    pt = link.p_tie(gs)
    p0 = float(pt[0])
    out = {"p_tie_0": p0}
    for name, level in (("half_of_p0", 0.5 * p0), ("abs_10pct", 0.10), ("abs_5pct", 0.05)):
        below = np.where(pt <= level)[0]
        out[f"width_{name}"] = float(gs[below[0]]) if len(below) and p0 > level else np.nan
    return out


def run_variant(variant: str):
    include_bb = variant == "robustness_bothbad"
    results, traj, bin_acc = [], [], []
    for k in range(len(cutoffs) - 1):
        train = dedup[dedup["tstamp"] <= cutoffs[k]]
        test = dedup[(dedup["tstamp"] > cutoffs[k]) & (dedup["tstamp"] <= cutoffs[k + 1])]
        if include_bb:
            test = test.assign(winner=test["winner"].replace({"tie (bothbad)": "tie"}))
        nu_hat, th_dav, _ = profile_lattice_unit(
            train, NUS, make_link=lambda v: DavidsonLink(nu=float(v)),
            mode="native", include_both_bad=include_bb)
        u_hat, th_lat, _ = profile_lattice_unit(
            train, UNITS, make_link=lambda v: LatticeLink(unit=float(v), g_step=0.005),
            mode="native", include_both_bad=include_bb)
        lk_d, lk_l = DavidsonLink(nu=nu_hat), LatticeLink(unit=u_hat, g_step=0.005)
        wr = evaluate_window_trinomial(th_dav, lk_d, th_lat, lk_l, test, labels[k])
        results.append(wr)
        rec = {"variant": variant, "window": labels[k], "nu_hat": nu_hat, "unit_hat": u_hat,
               "n_scoreable": wr.n_scoreable,
               "n_unscoreable": wr.n_in_outcome_space - wr.n_scoreable}
        for model, lk in (("davidson", lk_d), ("lattice", lk_l)):
            for key, val in band_widths(lk).items():
                rec[f"{model}_{key}"] = val
        traj.append(rec)
        print(f"[{variant} {labels[k]}] nu={nu_hat:.4f} unit={u_hat:.4f} "
              f"scoreable={wr.n_scoreable:,} "
              f"d_mean={wr.per_vote['d_logloss'].mean()/MPD_RQ4:+.3f}x MPD")

        # tie-curve bins (each model's own fitted gaps)
        sub = test[test["winner"].isin(("model_a", "model_b", "tie"))]
        for model, th, lk in (("davidson", th_dav, lk_d), ("lattice", th_lat, lk_l)):
            known = sub["model_a"].isin(th.index) & sub["model_b"].isin(th.index)
            s2 = sub[known]
            g = np.abs(th.reindex(s2["model_a"]).to_numpy()
                       - th.reindex(s2["model_b"]).to_numpy())
            tie = (s2["winner"] == "tie").to_numpy(dtype=float)
            pred = lk.p_tie(g) / np.maximum(
                lk.p_win(g) + lk.p_tie(g) + lk.p_loss(g), 1e-300)
            binned = pd.DataFrame({"bin": pd.cut(g, GAP_BINS), "tie": tie, "pred": pred})
            agg = binned.groupby("bin", observed=True).agg(
                n=("tie", "size"), emp=("tie", "mean"), fit=("pred", "mean"))
            agg = agg.assign(variant=variant, window=labels[k], model=model)
            bin_acc.append(agg.reset_index())

    wt = window_table_trinomial(results)
    pooled = pooled_estimate(wt)
    verdict = classify(wt, pooled, mpd=MPD_RQ4)
    verdict["verdict"] = relabel_verdict(verdict["verdict"])
    print(f"\n=== {variant} ===")
    print(wt.round(6).to_string(index=False))
    print(f"pooled {pooled['pooled']/MPD_RQ4:+.3f}x MPD "
          f"CI ({pooled['ci_lo']/MPD_RQ4:+.3f},{pooled['ci_hi']/MPD_RQ4:+.3f}) "
          f"[{pooled['n_windows']} windows, {pooled['n_votes']:,} votes]")
    print(f"VERDICT: {verdict['verdict']} {verdict.get('directional_note', '')}")
    return wt.assign(variant=variant), pd.DataFrame(traj), pd.concat(bin_acc), {
        "variant": variant, "pooled_x_mpd": pooled["pooled"] / MPD_RQ4,
        "ci_lo_x_mpd": pooled["ci_lo"] / MPD_RQ4, "ci_hi_x_mpd": pooled["ci_hi"] / MPD_RQ4,
        "n_lattice_better": verdict["n_lattice_better"],
        "n_davidson_better": verdict["n_bt_better"],
        "verdict": verdict["verdict"], "note": verdict.get("directional_note", "")}


wt_main, traj_main, bins_main, v_main = run_variant("main_tie_only")
wt_rob, traj_rob, bins_rob, v_rob = run_variant("robustness_bothbad")

pd.concat([wt_main, wt_rob]).to_csv(ROOT / "results" / "tables" / "rq4_window_table.csv",
                                    index=False)
pd.DataFrame([v_main, v_rob]).to_csv(ROOT / "results" / "tables" / "rq4_pooled_verdict.csv",
                                     index=False)
traj = pd.concat([traj_main, traj_rob])
traj.to_csv(ROOT / "results" / "tables" / "rq4_param_trajectories.csv", index=False)
traj[[c for c in traj.columns if "width" in c or "p_tie_0" in c or c in
      ("variant", "window", "nu_hat", "unit_hat")]].to_csv(
    ROOT / "results" / "tables" / "rq4_tie_band_table.csv", index=False)
pd.concat([bins_main, bins_rob]).to_csv(
    ROOT / "results" / "tables" / "rq4_tie_curve_bins.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
t = traj_main
axes[0].plot(t["window"], t["nu_hat"], marker="o", label="Davidson ν̂")
axes[0].plot(t["window"], t["unit_hat"], marker="s", label="lattice û")
axes[0].set_title("fitted tie parameters (main variant)")
axes[1].plot(t["window"], t["davidson_p_tie_0"], marker="o", label="Davidson P(tie|0)")
axes[1].plot(t["window"], t["lattice_p_tie_0"], marker="s", label="lattice P(tie|0)")
axes[1].set_title("implied at-zero tie probability")
axes[2].plot(t["window"], t["davidson_width_half_of_p0"], marker="o", label="Davidson")
axes[2].plot(t["window"], t["lattice_width_half_of_p0"], marker="s", label="lattice")
axes[2].set_title("half-max tie-band half-width (ability units)")
for ax in axes:
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(ROOT / "results" / "figures" / "rq4_trajectories.png", dpi=150)
print("\nDONE")

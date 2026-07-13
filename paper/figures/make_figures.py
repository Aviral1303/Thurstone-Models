"""Generate all paper figures from the ACTUAL result CSVs and model code.

Every data point is read from results/tables/*.csv (frozen at commit
1e7c13d) or computed from the fitted model implementations in src/ — no
illustrative/fabricated data anywhere. Color scheme (entity-consistent,
dataviz-skill validated): red #e34948 = logit family (BT/Davidson);
blue ordinal ramp #86b6ef/#2a78d6/#104281 = lattice family (units
0.1/0.5855/0.8); ink = empirical data; every series direct-labeled.

Run from repo root: .venv/bin/python paper/figures/make_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
FIG = ROOT / "paper" / "figures"
T = ROOT / "results" / "tables"

from davidson_link import DavidsonLink  # noqa: E402
from lattice_link import LatticeLink, LogisticLink  # noqa: E402

# ---- palette / style (dataviz reference instance, light surface) ----
RED = "#e34948"          # logit family (BT, Davidson)
BLUES = {"0.1": "#86b6ef", "0.5855": "#2a78d6", "0.8": "#104281"}
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.grid": True, "axes.grid.axis": "y",
    "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 2.0,
})

MPD3, MPD4 = 4e-4, 3e-4


def month_ticks(ax, labels, every=2):
    ax.set_xticks(range(len(labels))[::every])
    ax.set_xticklabels([l[:7] for l in labels][::every], rotation=0)


# ================= Figure M: model construction + links =================
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))

ax = axes[0]
u = 0.5855
lk_small = LatticeLink(unit=u)
from thurstone import Density, UniformLattice  # noqa: E402

lat = UniformLattice(L=12, unit=u)  # coarse lattice so cells are visible
dA = Density.skew_normal(lat, loc=0.0, scale=1.0, a=0.0)
g_show = 0.8
dB = Density.skew_normal(lat, loc=g_show, scale=1.0, a=0.0)
grid = lat.grid
w = u * 0.42
ax.bar(grid - w / 2, dA.p, width=w, color=BLUES["0.5855"], label="model $A$")
ax.bar(grid + w / 2, dB.p, width=w, color=BLUES["0.1"], label="model $B$")
overlap = np.minimum(dA.p, dB.p)
ax.bar(grid, overlap, width=u * 0.92, color="none", edgecolor=INK2,
       hatch="////", linewidth=0.0, label="dead-heat mass")
ax.set_xlim(-3.4, 4.2)
ax.set_xlabel("latent performance (ability units)")
ax.set_ylabel("cell probability")
ax.set_title(f"(a) lattice performances, unit $u={u}$, gap $g={g_show}$")
ax.legend(loc="upper left", handlelength=1.2, frameon=True,
          facecolor="white", framealpha=0.85, edgecolor="none")

ax = axes[1]
gs = np.linspace(-3, 3, 601)
ax.plot(gs, 1 / (1 + np.exp(-gs)), color=RED, label="logistic (BT)")
for uu, c in BLUES.items():
    lk = LatticeLink(unit=float(uu))
    ax.plot(gs, lk.f_decisive(gs), color=c, linewidth=1.6,
            label=f"lattice $u={uu}$")
ax.axhline(0.5, color=GRID, linewidth=0.6)
ax.set_xlabel("ability gap $g$")
ax.set_ylabel(r"$P(A \succ B \mid \mathrm{decisive})$")
ax.set_title("(b) conditional decisive links")
ax.legend(loc="lower right", handlelength=1.2)
fig.tight_layout()
fig.savefig(FIG / "fig_model.pdf")
print("fig_model: slopes at 0:",
      {uu: round(LatticeLink(unit=float(uu)).slope_at_zero(), 4) for uu in BLUES},
      "logistic 0.25")

# ================= Figure 1: RQ1 stability =================
m = pd.read_csv(T / "rq1_metrics.csv")
s = m[(m.delta == 1) & (m.incumbents == "votes>=1000")]
piv = s.pivot(index="T", columns="method", values="kendall")
labels = list(piv.index)
fig, ax = plt.subplots(figsize=(6.4, 2.7))
x = np.arange(len(labels))
ax.plot(x, piv["bt"], color=RED, marker="o", markersize=4, label="BT (logistic)")
ax.plot(x, piv["lattice_u0.1"], color=BLUES["0.5855"], marker="s",
        markersize=4, label="lattice ($u=0.1$)")
month_ticks(ax, labels)
ax.set_ylabel(r"incumbent Kendall $\tau_b$,  $T$ vs $T{+}1$ mo")
ax.set_xlabel("checkpoint $T$")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "fig_rq1.pdf")
d = piv["lattice_u0.1"] - piv["bt"]
print(f"fig_rq1: median diff {d.median():+.5f}, split "
      f"{(d>0).sum()}/{(d<0).sum()}/{(d==0).sum()}")

# ================= Figure 2: RQ2b blocks =================
b = pd.read_csv(T / "rq2b_blocks.csv")
fig, ax = plt.subplots(figsize=(6.4, 2.7))
x = np.arange(len(b))
link_cols = [("mean_zsq_logit", RED, "logit", "o"),
             ("mean_zsq_lat_u0.1", BLUES["0.1"], "lattice $u{=}0.1$", "s"),
             ("mean_zsq_lat_u0.5855", BLUES["0.5855"], "lattice $u{=}0.5855$", "^"),
             ("mean_zsq_lat_u0.8", BLUES["0.8"], "lattice $u{=}0.8$", "D")]
for col, c, lab, mk in link_cols:
    ax.plot(x, b[col], color=c, marker=mk, markersize=4.5, linewidth=1.4, label=lab)
ax.axhline(1.0, color=INK2, linewidth=0.8, linestyle="--")
ax.annotate("calibrated ($\\bar{z^2}=1$)", (0.02, 1.02), color=INK2, fontsize=7.5)
ax.set_ylim(0.93, 1.95)
for i, row in b.iterrows():
    if row["mean_zsq_logit"] > 1.5:
        ytop = max(b.loc[i, c] for c in ["mean_zsq_logit", "mean_zsq_lat_u0.1",
                                          "mean_zsq_lat_u0.5855", "mean_zsq_lat_u0.8"])
        ax.annotate("excess", (x[i], ytop + 0.07), ha="center", color=INK,
                    fontsize=8, fontweight="bold")
labels2 = [f"{lbl.split('_')[1]}" for lbl in b["block"]]
ax.set_xticks(x)
ax.set_xticklabels([f"{l[:7]}\n(n={n})" for l, n in zip(labels2, b['n_triples'])],
                   fontsize=7.5)
ax.set_ylabel(r"mean $z^2$ of triple-additivity residuals")
ax.set_xlabel("60-day block (start month, triples)")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16),
          columnspacing=1.0, handlelength=1.4)
fig.tight_layout()
fig.savefig(FIG / "fig_rq2b.pdf")
print("fig_rq2b: block means",
      b[["block", "mean_zsq_logit"]].round(3).to_dict("records"))

# ================= Figure 3: RQ3 window deltas =================
w3 = pd.read_csv(T / "rq3_window_table_u0.5855.csv")
fig, ax = plt.subplots(figsize=(6.4, 2.7))
x = np.arange(len(w3))
vals = w3["mean_d_logloss"] / MPD3
ax.axhspan(-1, 1, color=GRID, alpha=0.55, zorder=0)
ax.annotate("practical-equivalence band ($\\pm$MPD)", (0.0, -0.85),
            color=INK2, fontsize=7.5)
ax.bar(x, vals, width=0.62, color=BLUES["0.5855"], zorder=2)
ax.axhline(0, color=BASE, linewidth=0.8)
k = int(vals.abs().idxmax())
ax.annotate(f"largest window: ${vals[k]:+.2f}\\times$MPD",
            (x[k] + 0.35, vals[k] - 0.12), color=INK, fontsize=7.5)
ax.set_ylim(-1.15, 1.15)
month_ticks(ax, list(w3["window"]))
ax.set_ylabel(r"window mean $\Delta$ ($\times$MPD)")
ax.set_xlabel("test window (lattice better $>0$, BT better $<0$)")
fig.tight_layout()
fig.savefig(FIG / "fig_rq3.pdf")
print(f"fig_rq3: outlier window {w3.loc[k,'window']} at {vals[k]:+.2f}xMPD")

# ================= Figure 4: RQ4 tie curves =================
bins = pd.read_csv(T / "rq4_tie_curve_bins.csv")
bins = bins[bins.variant == "main_tie_only"]
mids = {"(0.0, 0.15]": 0.075, "(0.15, 0.3]": 0.225, "(0.3, 0.5]": 0.4,
        "(0.5, 0.8]": 0.65, "(0.8, 1.2]": 1.0, "(1.2, 2.0]": 1.6}
pool = bins.groupby(["model", "bin"], observed=True).apply(
    lambda d: pd.Series({"n": d["n"].sum(),
                         "emp": np.average(d["emp"], weights=d["n"]),
                         "fit": np.average(d["fit"], weights=d["n"])}),
    include_groups=False).reset_index()
pool["mid"] = pool["bin"].map(mids)
pool = pool.dropna(subset=["mid"]).sort_values("mid")

# small multiples: each model binned by ITS OWN fitted gaps, with its own
# empirical series — a shared empirical curve would silently mix binnings.
traj = pd.read_csv(T / "rq4_param_trajectories.csv")
last = traj[traj.variant == "main_tie_only"].iloc[-1]
gs = np.linspace(0, 2.0, 300)
dv = DavidsonLink(nu=last.nu_hat)
lt = LatticeLink(unit=last.unit_hat, g_step=0.005)
den_d = dv.p_win(gs) + dv.p_tie(gs) + dv.p_loss(gs)
den_l = lt.p_win(gs) + lt.p_tie(gs) + lt.p_loss(gs)
panels = (
    ("davidson", RED, dv.p_tie(gs) / den_d,
     f"(a) Davidson  ($\\hat\\nu={last.nu_hat:.3f}$, final window)"),
    ("lattice", BLUES["0.5855"], lt.p_tie(gs) / den_l,
     f"(b) lattice  ($\\hat u={last.unit_hat:.3f}$, final window)"),
)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
for ax, (model, c, curve, title) in zip(axes, panels):
    ax.axvspan(1.2, 2.0, color=GRID, alpha=0.55, zorder=0)
    ax.annotate("$<10\\%$ of votes", (1.28, 0.02), color=INK2, fontsize=7)
    ax.plot(gs, curve, color=c, linewidth=1.6, label="fitted tie curve")
    p = pool[pool.model == model]
    ax.plot(p["mid"], p["fit"], color=c, marker="o", markersize=4,
            linestyle="none", label="fitted (pooled bins)")
    ax.plot(p["mid"], p["emp"], color=INK, marker="o", markersize=5,
            linestyle="none", markerfacecolor="white", markeredgewidth=1.3,
            label="empirical tie rate")
    ax.set_xlabel(r"fitted ability gap $|\hat g|$")
    ax.set_title(title, fontsize=8)
    ax.set_ylim(0, 0.27)
axes[0].set_ylabel(r"$P(\mathrm{tie})$")
axes[0].legend(fontsize=6.8, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "fig_rq4_curves.pdf")
for model in ("davidson", "lattice"):
    p = pool[pool.model == model]
    rms = np.sqrt(np.average((p["emp"] - p["fit"]) ** 2, weights=p["n"]))
    print(f"fig_rq4_curves: {model} pooled bin RMS = {rms:.4f}")

# ================= Figure 5: RQ4 drift trajectories =================
tm = traj[traj.variant == "main_tie_only"].reset_index()
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
x = np.arange(len(tm))
ax = axes[0]
ax.plot(x, tm["nu_hat"], color=RED, marker="o", markersize=4,
        label=r"Davidson $\hat\nu$")
ax.plot(x, tm["unit_hat"], color=BLUES["0.5855"], marker="s", markersize=4,
        label=r"lattice $\hat u$")
for ref, lab in ((0.5855, "first-3-months profile fit"),
                 (0.8002, "full-sample profile fit")):
    ax.axhline(ref, color=MUTED, linewidth=0.7, linestyle=":")
    ax.annotate(lab, (4.6, ref + 0.012), color=MUTED, fontsize=6.8)
month_ticks(ax, list(tm["window"]), every=3)
ax.set_ylabel("fitted tie parameter")
ax.set_title("(a) per-window tie parameters")
ax.legend(loc="lower right", fontsize=7)
ax = axes[1]
ax.plot(x, tm["davidson_width_half_of_p0"], color=RED, marker="o",
        markersize=4, label="Davidson")
ax.plot(x, tm["lattice_width_half_of_p0"], color=BLUES["0.5855"], marker="s",
        markersize=4, label="lattice")
month_ticks(ax, list(tm["window"]), every=3)
ax.set_ylabel("half-max half-width (ability units)")
ax.set_ylim(0, 3.3)
ax.set_title("(b) implied tie-band width")
ax.legend(loc="lower right", fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "fig_rq4_traj.pdf")
print("fig_rq4_traj: nu", tm.nu_hat.iloc[0].round(3), "->", tm.nu_hat.iloc[-1].round(3),
      "| u", tm.unit_hat.iloc[0].round(3), "->", tm.unit_hat.iloc[-1].round(3),
      "| widths", tm.davidson_width_half_of_p0.min().round(2), "-",
      tm.davidson_width_half_of_p0.max().round(2), "vs",
      tm.lattice_width_half_of_p0.min().round(2), "-",
      tm.lattice_width_half_of_p0.max().round(2))
print("DONE")

# ================= Figure C: conceptual error budget =================
# Schematic (lengths NOT to scale — log-flavored visual ordering); every
# annotation is a verified number from FINDINGS_INVENTORY / HEAD re-runs.
fig, ax = plt.subplots(figsize=(6.6, 2.4))
items = [
    ("link-family choice\n(logistic vs any lattice width)",
     1.0, "$\\leq 0.23$–$0.31\\times$MPD by construction\n(effect ceilings, computed before fitting)", BLUES["0.5855"]),
    ("single-window episodes\n(tie channel, RQ4)",
     2.6, "up to $9.6\\times$MPD in one window", "#8a8781"),
    ("shared nonstationary drift\n(all methods alike)",
     4.2, "reliability slope $\\approx 1.46$;\ntie share 13.1%$\\to$20.4% over 16 mo", "#8a8781"),
    ("cold-start coverage\n(no ability model can score)",
     5.8, "24.8–75.8% of next-month\ndecisive votes unscoreable", INK2),
]
for y, (label, length, note, c) in zip([3, 2, 1, 0], items):
    ax.barh(y, length, height=0.52, color=c)
    ax.text(-0.12, y, label, ha="right", va="center", fontsize=8, color=INK)
    ax.text(length + 0.12, y, note, ha="left", va="center", fontsize=7.2, color=INK2)
ax.set_xlim(0, 10.4)
ax.set_ylim(-0.6, 3.7)
ax.set_yticks([])
ax.set_xticks([])
ax.spines[["left", "bottom"]].set_visible(False)
ax.grid(False)
ax.set_title("What limits next-month ranking quality here (schematic; bar lengths not to scale)",
             fontsize=8.5, color=INK)
fig.tight_layout()
fig.savefig(FIG / "fig_concept.pdf")
print("fig_concept written")

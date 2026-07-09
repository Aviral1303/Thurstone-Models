"""Lattice-Thurstone gap link for pairwise battles (higher-is-better wrapper).

Per SCOPE_REFRAME.md: with 2-entrant data the lattice model IS a gap-link
model. This module computes that link exactly once, from the vendored
package's forward Race/pricing machinery, as 1D curves over the ability gap:

    W(g) = P(i beats j | theta_i - theta_j = g)
    D(g) = P(dead-heat  | theta_i - theta_j = g)
    L(g) = P(j beats i  | ... ) = W(-g)

Sign convention: THIS module is higher-is-better (theta up = stronger). The
underlying package is race-time (lower = better); the flip happens exactly
once, here, by shifting the first entrant's performance density DOWN by g.
Pinned by tests/test_sign_convention.py.

Two likelihood views (see logs/RESEARCH_LOG.md, Phase 3 decisions):
- decisive link  F(g) = W/(W+L): conditional-on-decisive two-outcome link,
  used in the fastchat-equivalent "half-tie" fit so BT and lattice see
  literally identical outcome spaces (BT's logistic has no tie mass).
- native trinomial (W, D, L): used for RQ4's dead-heat analysis.

The dead-heat mass D comes from performance draws landing on the same lattice
cell, so the lattice `unit` is an implicit tie-band width (RQ4 sensitivity
axis — do not tune it against tie-rate fit outside that analysis).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from thurstone import STD_L, STD_UNIT, Density, UniformLattice
from thurstone.pricing import conditional_win_draw_loss

_EPS = 1e-300


@dataclass
class LatticeLink:
    scale: float = 1.0
    skew_a: float = 0.0
    unit: float = STD_UNIT
    L: int = STD_L
    g_max: float = 8.0  # curve support in ability units (Arena spread ~3)
    g_step: float = 0.02
    _gaps: np.ndarray = field(init=False, repr=False)
    _W: np.ndarray = field(init=False, repr=False)
    _D: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        lattice = UniformLattice(L=self.L, unit=self.unit)
        base = Density.skew_normal(lattice, loc=0.0, scale=self.scale, a=self.skew_a)
        cdf_base = base.cdf()
        gaps = np.arange(-self.g_max, self.g_max + self.g_step / 2, self.g_step)
        W = np.empty_like(gaps)
        D = np.empty_like(gaps)
        for k, g in enumerate(gaps):
            # higher-is-better: entrant with advantage g gets race-time loc -g
            d_i = base.shift_fractional(-g / self.unit)
            win, draw, _ = conditional_win_draw_loss(d_i, cdf_base)
            W[k] = float(np.sum(win))
            D[k] = float(np.sum(draw))
        self._gaps = gaps
        self._W = W
        self._D = D

    # ---- raw curves (vectorized in g) ----
    def p_win(self, g):
        return np.interp(g, self._gaps, self._W)

    def p_tie(self, g):
        return np.interp(g, self._gaps, self._D)

    def p_loss(self, g):
        return np.interp(-np.asarray(g, dtype=float), self._gaps, self._W)

    # ---- decisive (conditional) link and log-derivatives for MLE ----
    def f_decisive(self, g):
        w = self.p_win(g)
        lo = self.p_loss(g)
        return w / np.maximum(w + lo, _EPS)

    def _log_curve_and_grad(self, curve: np.ndarray):
        logc = np.log(np.maximum(curve, _EPS))
        dlogc = np.gradient(logc, self._gaps)
        return logc, dlogc

    def log_f_decisive(self, g):
        """log F(g) and d/dg log F(g), interpolated (vectorized)."""
        w = self._W
        lo = self._W[::-1]  # p_loss on the same grid = W mirrored
        f = w / np.maximum(w + lo, _EPS)
        logf, dlogf = self._log_curve_and_grad(f)
        return np.interp(g, self._gaps, logf), np.interp(g, self._gaps, dlogf)

    def log_p_win(self, g):
        logw, dlogw = self._log_curve_and_grad(self._W)
        return np.interp(g, self._gaps, logw), np.interp(g, self._gaps, dlogw)

    def log_p_tie(self, g):
        logd, dlogd = self._log_curve_and_grad(self._D)
        return np.interp(g, self._gaps, logd), np.interp(g, self._gaps, dlogd)


class LogisticLink:
    """BT/logit link in the same interface, for implementation cross-checks.

    No native tie mass: p_tie == 0, f_decisive == sigmoid.
    """

    def f_decisive(self, g):
        return 1.0 / (1.0 + np.exp(-np.asarray(g, dtype=float)))

    def p_win(self, g):
        return self.f_decisive(g)

    def p_loss(self, g):
        return self.f_decisive(-np.asarray(g, dtype=float))

    def p_tie(self, g):
        return np.zeros_like(np.asarray(g, dtype=float))

    def log_f_decisive(self, g):
        g = np.asarray(g, dtype=float)
        # log sigmoid(g) = -log(1+exp(-g)); d/dg = sigmoid(-g)
        return -np.logaddexp(0.0, -g), self.f_decisive(-g)

"""Davidson-extended BT as a gap link (RQ4 comparator).

Davidson (1970): P(i beats j) = p_i / (p_i + p_j + nu*sqrt(p_i p_j)).
With p = e^theta and g = theta_i - theta_j, dividing by sqrt(p_i p_j):

    denom(g) = e^{g/2} + e^{-g/2} + nu = 2 cosh(g/2) + nu
    P(win|g) = e^{g/2} / denom,  P(tie|g) = nu / denom,
    P(loss|g) = e^{-g/2} / denom

Conditional decisive link is EXACTLY logistic (independent of nu):
W/(W+L) = sigmoid(g) — so Davidson and vanilla BT coincide on decisive
outcomes, and nu is purely the tie-mass parameter. Same interface as
LatticeLink (native trinomial + decisive modes, analytic gradients).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-300


@dataclass
class DavidsonLink:
    nu: float = 0.5

    def _denom(self, g):
        g = np.asarray(g, dtype=float)
        return 2.0 * np.cosh(g / 2.0) + self.nu

    def p_win(self, g):
        g = np.asarray(g, dtype=float)
        return np.exp(g / 2.0) / self._denom(g)

    def p_tie(self, g):
        return self.nu / self._denom(g)

    def p_loss(self, g):
        g = np.asarray(g, dtype=float)
        return np.exp(-g / 2.0) / self._denom(g)

    # ---- decisive (conditional) link: exactly logistic ----
    def f_decisive(self, g):
        g = np.asarray(g, dtype=float)
        return 1.0 / (1.0 + np.exp(-g))

    def log_f_decisive(self, g):
        g = np.asarray(g, dtype=float)
        return -np.logaddexp(0.0, -g), self.f_decisive(-g)

    # ---- native trinomial log-probs with analytic gradients ----
    def log_p_win(self, g):
        g = np.asarray(g, dtype=float)
        denom = self._denom(g)
        val = g / 2.0 - np.log(np.maximum(denom, _EPS))
        grad = 0.5 - np.sinh(g / 2.0) / denom
        return val, grad

    def log_p_tie(self, g):
        g = np.asarray(g, dtype=float)
        denom = self._denom(g)
        val = np.log(max(self.nu, _EPS)) - np.log(np.maximum(denom, _EPS))
        grad = -np.sinh(g / 2.0) / denom
        return val, grad

    def slope_at_zero(self) -> float:
        return 0.25  # decisive link is logistic

    def slope_match_factor(self) -> float:
        return 1.0

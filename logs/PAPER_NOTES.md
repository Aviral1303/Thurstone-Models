# Paper notes — theory/background facts and framing items

Running notes feeding the paper's theory/background and discussion
sections. Started 2026-07-10. (Findings live in their own docs:
RQ1_FINDINGS, RQ3_FINDINGS, RQ2B_EXCESS_FINDING, RESEARCH_LOG; framing
constraints in SCOPE_REFRAME.)

## Structural fact: Davidson's decisive link is exactly logistic, for any ν

Davidson (1970) extends BT with P(i beats j) = p_i / (p_i + p_j +
ν√(p_i p_j)) and P(tie) = ν√(p_i p_j) / (p_i + p_j + ν√(p_i p_j)).
Write p = e^θ and g = θ_i − θ_j, divide through by √(p_i p_j):

    P(win | g)  = e^{g/2} / (2 cosh(g/2) + ν)
    P(tie | g)  = ν       / (2 cosh(g/2) + ν)
    P(loss | g) = e^{-g/2} / (2 cosh(g/2) + ν)

so the conditional-on-decisive probability is

    P(win | decisive, g) = e^{g/2} / (e^{g/2} + e^{-g/2}) = σ(g),

**the logistic link exactly, independent of ν.** Consequences worth
stating in the theory/background section (alongside the general framing
of gap-link/random-utility families, not buried in a results table):

1. ν is *purely* a tie-mass parameter: Davidson-BT and vanilla BT are the
   same model on decisive outcomes. Any decisive-outcome comparison
   between "BT" and "Davidson-BT" is vacuous by construction.
2. The lattice model does NOT share this factorization: carving dead-heat
   mass out of a continuous performance density steepens the conditional
   decisive link (slope at 0 rises from ≈0.284 toward the coarse-unit
   limit; measured 0.290→0.337 over units 0.1→0.8). Tie handling and
   decisive-link shape are entangled in the Thurstonian family but
   orthogonal in the Davidson family — a clean structural contrast
   between the two tie mechanisms, independent of any empirical result.
3. Practical corollary used throughout our design: comparing tie
   mechanisms at equal parameter count (RQ4) is only meaningful in the
   full trinomial likelihood; on decisive votes the Davidson side
   degenerates to BT. (Verified in code: DavidsonLink.f_decisive ≡
   sigmoid; tests/test_davidson_link.py.)

(Proof verified numerically for ν ∈ {0.1, 0.5, 1.5} to 1e-12.)

## Other discussion-section items already flagged elsewhere (pointers)

- Two faces of steep-link × parameter-uncertainty: prediction-side
  overconfidence (RQ3 pre-analysis §6, qualified dose-response in
  RQ3_FINDINGS §4) and estimation-side shrinkage (cold-start case study,
  RQ3_FINDINGS §4, n=1 caveat). General caution for ranking/reward-model
  deployments using steeper-than-necessary links without uncertainty
  correction.
- Two independently-derived effect ceilings (RQ3 analytic KL ≤0.23×MPD;
  RQ2b additivity residuals) converging on "no discriminable link-shape
  effect at real gap scales" — state the convergence explicitly.
- Deployment finding: coverage (25–75% unscoreable) and nonstationary
  drift (reliability slope 1.455 for every model; tie share 13.1→20.45%)
  bind next-month predictability; link choice does not.
- RQ2a descope reasoning (SCOPE_REFRAME.md): entry-based DiD is the wrong
  instrument for the one unexplained signal (excess anti-correlated with
  entry activity).

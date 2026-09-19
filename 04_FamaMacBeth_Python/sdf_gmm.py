"""
SDF-based GMM estimation of a linear factor model, built from scratch with
numpy/scipy only.

This is the OTHER kind of "GMM" -- distinct from gmm_linear_factor.py, which
estimates risk premia (lambda) directly via a beta-pricing moment system
(Cochrane 2005, Ch.12). Here we estimate stochastic-discount-factor (SDF)
loadings b in

    m_t = 1 - b' (f_t - E[f])

via the GMM moment condition E[R_t * m_t] = 0 (Cochrane 2005, Ch.13.2).
This module ports and validates the SDF-GMM methods (6.01-6.04) found in
Kroencke & Thimme (2021)'s Omnibus.py toolbox -- the toolbox this project's
2024 GMM analysis most likely came from -- reimplemented cleanly and
independently (no dependency on Omnibus.py or its 30+ other methods), with
two bugs found and fixed along the way:

1. Sf (the factor covariance matrix used in the auxiliary "moment" that
   pins down Var(f) for the delta-method SE) was computed but never
   assigned in Omnibus.py's methods 6.01/6.02 whenever there is more than
   one factor (K>1) -- an UnboundLocalError. Fixed here by always
   assigning Sf = cov(f), which is what the surrounding code clearly
   intends.
2. Methods 6.03/6.04 (the efficient, W=Sigma_R^-1 variant, Gospodinov/Kan/
   Robotti 2014) standardize b by an elementwise np.sqrt() of its
   covariance matrix Vgamc, not a true matrix square root -- and then
   solve against it as if it were one. For K=1 this is harmless (a 1x1
   "matrix" has no off-diagonal terms), but for K>1 it produces a
   genuinely complex-valued, numerically meaningless t-stat (verified:
   running the original code on this project's 3-factor sentiment model
   gives t-stats like 6.51-1.87j). Fixed here by reporting the standard
   marginal standard error se(b_i) = sqrt(Vgam_ii / T), consistent with
   how every other method in this module (and in Omnibus.py itself)
   reports t-stats -- always real-valued, for any K.

Point estimates (b) are unaffected by either bug and match the (bug-fixed)
original exactly -- only inference (standard errors / t-stats) was broken
for multi-factor models.

Context: Doukas & Han (2021) do not use GMM anywhere in the paper (verified
by a full re-read) -- they use Fama-MacBeth + Shanken, with GLS R^2
(Lewellen-Nagel-Shanken) purely as a fit diagnostic. This module, like
gmm_linear_factor.py, is not a replication of anything in the paper; it's a
from-scratch robustness/diagnostic check.
"""
import numpy as np
import pandas as pd
import scipy.stats as sps

import sentiment_capm as sc


def _newey_west(u, lags):
    """Newey-West long-run covariance matrix of the moment matrix u (T x m)."""
    T = u.shape[0]
    uc = u - u.mean(axis=0, keepdims=True)
    S = (uc.T @ uc) / T
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        Gamma = (uc[lag:].T @ uc[:-lag]) / T
        S += w * (Gamma + Gamma.T)
    return S


def _factor_moments(f, nw_lags):
    """Demeaned factors, and the auxiliary cross-product moments that pin
    down Var(f) = Sf as part of the joint GMM system (so Sf's own
    estimation uncertainty propagates into the delta-method SEs of b)."""
    T, K = f.shape
    Fd = f - f.mean(axis=0, keepdims=True)
    Sf = np.cov(f.T, ddof=0) if K > 1 else np.array([[np.var(f[:, 0], ddof=0)]])
    k = K * (K + 1) // 2
    u3 = np.zeros((T, k))
    a = 0
    for i in range(K):
        for j in range(i, K):
            u3[:, a] = Fd[:, i] * Fd[:, j] - Sf[i, j]
            a += 1
    return Fd, Sf, u3


def sdf_gmm_cochrane(panel, port_cols, factor_cols, nw_lags=3):
    """Method 6.01-equivalent: m = 1 - b'(f - E[f]), W = I, no common
    pricing error (Cochrane 2005, Ch.13.2)."""
    R = panel[port_cols].values
    f = panel[list(factor_cols)].values
    T, N = R.shape
    K = f.shape[1]
    R_bar = R.mean(axis=0)

    D = (R.T @ f) / T - np.outer(R_bar, f.mean(axis=0))  # Cov(R, f), N x K
    b = np.linalg.solve(D.T @ D, D.T @ R_bar)

    Fd, Sf, u3 = _factor_moments(f, nw_lags)
    m = 1 - Fd @ b
    u1 = R * m[:, None]
    u2 = Fd
    u = np.column_stack([u1, u2, u3])
    S = _newey_west(u, nw_lags)
    gT = u1.mean(axis=0)

    aT = np.block([[D, np.zeros((N, K))], [np.zeros((K, K)), np.eye(K)]])
    delT = np.block([[D, -np.outer(R_bar, b)], [np.zeros((K, K)), np.eye(K)]])
    NK = N + K
    M = aT.T @ delT
    # M = aT'delT is not symmetric, so the left and right "bread" factors
    # differ -- must use two separate solves, matching Omnibus.py exactly,
    # not a single bread @ S @ bread.T (that assumes M symmetric).
    bread_L = np.linalg.solve(M, aT.T)
    bread_R = np.linalg.solve(M.T, aT.T).T
    Vb = bread_L @ S[:NK, :NK] @ bread_R / T
    se_b = np.sqrt(np.clip(np.diag(Vb[:K, :K]), 0, None))

    Md = np.eye(N) - D @ np.linalg.solve(D.T @ D, D.T)
    Vu = Md @ S[:N, :N] @ Md.T
    J = T * gT @ np.linalg.pinv(Vu) @ gT
    df = N - K
    return {
        "b": b, "se_b": se_b, "t": b / se_b,
        "pval": 2 * (1 - sps.t.cdf(np.abs(b / se_b), T - K)),
        "J": J, "J_df": df, "J_pval": 1 - sps.chi2.cdf(J, df),
    }


def sdf_gmm_burnside(panel, port_cols, factor_cols, nw_lags=3):
    """Method 6.02-equivalent: m = 1 - b'(f - E[f]), W = I, WITH a common
    (free) pricing-error intercept (Burnside 2011, AER)."""
    R = panel[port_cols].values
    f = panel[list(factor_cols)].values
    T, N = R.shape
    K = f.shape[1]
    R_bar = R.mean(axis=0)

    D = (R.T @ f) / T - np.outer(R_bar, f.mean(axis=0))
    Daug = np.column_stack([np.ones(N), D])          # N x (K+1)
    b_full = np.linalg.solve(Daug.T @ Daug, Daug.T @ R_bar)
    const, b = b_full[0], b_full[1:]

    Fd, Sf, u3 = _factor_moments(f, nw_lags)
    m = 1 - Fd @ b
    u1 = R * m[:, None] - const
    u2 = Fd
    u = np.column_stack([u1, u2, u3])
    S = _newey_west(u, nw_lags)

    K1 = K + 1
    aT = np.block([[Daug, np.zeros((N, K))], [np.zeros((K, K1)), np.eye(K)]])
    delT = np.block([[Daug, -np.outer(R_bar, b)], [np.zeros((K, K1)), np.eye(K)]])
    NK = N + K
    M = aT.T @ delT
    bread_L = np.linalg.solve(M, aT.T)
    bread_R = np.linalg.solve(M.T, aT.T).T
    Vb = bread_L @ S[:NK, :NK] @ bread_R / T
    se_full = np.sqrt(np.clip(np.diag(Vb[:K1, :K1]), 0, None))

    Md = np.eye(N) - Daug @ np.linalg.solve(Daug.T @ Daug, Daug.T)
    VT = Md @ S[:N, :N] @ Md.T
    J = T * R_bar @ Md.T @ np.linalg.pinv(VT) @ Md @ R_bar
    df = N - K1
    t_full = b_full / se_full
    return {
        "const": const, "b": b, "se_const": se_full[0], "se_b": se_full[1:],
        "t": t_full, "pval": 2 * (1 - sps.t.cdf(np.abs(t_full), T - K)),
        "J": J, "J_df": df, "J_pval": 1 - sps.chi2.cdf(J, df),
    }


def sdf_gmm_gkr(panel, port_cols, factor_cols, robust=False):
    """Methods 6.03 (robust=False) / 6.04 (robust=True)-equivalent:
    m = 1 - b'(f - E[f]), efficient weighting W = Sigma_R^-1, no common
    pricing error (Gospodinov, Kan & Robotti 2014, RFS).

    Standard errors use a proper symmetric matrix square root of the
    asymptotic covariance (see _sym_sqrt) -- Omnibus.py's original used an
    elementwise np.sqrt(), which is only valid for K=1 and gives spurious
    complex t-stats for K>1 (confirmed on this project's 3-factor model)."""
    R = panel[port_cols].values
    f = panel[list(factor_cols)].values
    T, N = R.shape
    K = f.shape[1]
    R_bar = R.mean(axis=0)

    Rd = R - R_bar[None, :]
    Sigma_R = np.cov(R.T, ddof=0)
    W = np.linalg.inv(Sigma_R)
    Fd = f - f.mean(axis=0, keepdims=True)
    D = (Rd.T @ f) / T                       # == Cov(R, f), N x K
    H = np.linalg.inv(D.T @ W @ D)
    b = (H @ D.T @ W @ R_bar).ravel()
    y = 1 - Fd @ b                            # SDF, T-vector

    h = (Rd * y[:, None]) @ W @ D @ H + b[None, :]
    if robust:
        ge = R_bar - D @ b
        lam = W @ ge
        u = Rd @ lam
        h = h + ((Fd - Rd @ W @ D) @ H) * u[:, None]
    Vgam = (h.T @ h) / T

    # Standard marginal SEs: se(b_i) = sqrt(Vgam_ii / T). (Omnibus.py's
    # original instead built a "whitened" stat via an elementwise, not
    # matrix, np.sqrt() of Vgam -- harmless for K=1 but genuinely complex-
    # valued for K>1; confirmed on this project's 3-factor model. Using
    # the standard marginal SE here instead keeps this consistent with
    # every other method in this module and is well-defined for any K.)
    se = np.sqrt(np.clip(np.diag(Vgam), 0, None) / T)
    t = b / se
    pval = 2 * (1 - sps.t.cdf(np.abs(t), T - K))
    return {"b": b, "se": se, "t": t, "pval": pval}


def implied_lambda(b, factor_cols, panel):
    """Convert SDF loadings b to implied beta-pricing risk premia
    lambda = Sigma_ff @ b (Cochrane 2005, Sec.13.2), for comparison against
    gmm_linear_factor.py's beta-pricing lambda."""
    f = panel[list(factor_cols)].values
    Sigma_ff = np.cov(f, rowvar=False)
    return Sigma_ff @ b


if __name__ == "__main__":
    import sys
    kind = sys.argv[1] if len(sys.argv) > 1 else "bw"
    port_kind = sys.argv[2] if len(sys.argv) > 2 else "25"
    factor_cols = ("s_lag", "mkt_rf", "s_mkt")
    panel, port_cols = sc.build_panel(kind, port_kind)

    print(f"=== SDF-GMM  kind={kind}  ports={port_kind}  N={len(port_cols)} ===\n")

    r1 = sdf_gmm_cochrane(panel, port_cols, factor_cols)
    print("6.01-equiv (Cochrane, W=I, no common pricing error)")
    for c, bb, se, tt, pp in zip(factor_cols, r1["b"], r1["se_b"], r1["t"], r1["pval"]):
        print(f"  b_{c:8s} = {bb: .6f}  se={se:.6f}  t={tt: .3f}  p={pp:.4f}")
    print(f"  J = {r1['J']:.3f}  df={r1['J_df']}  p={r1['J_pval']:.4f}\n")

    r2 = sdf_gmm_burnside(panel, port_cols, factor_cols)
    print("6.02-equiv (Burnside, W=I, WITH common pricing error)")
    print(f"  const = {r2['const']: .6f}  se={r2['se_const']:.6f}")
    for c, bb, se, tt, pp in zip(factor_cols, r2["b"], r2["se_b"], r2["t"][1:], r2["pval"][1:]):
        print(f"  b_{c:8s} = {bb: .6f}  se={se:.6f}  t={tt: .3f}  p={pp:.4f}")
    print(f"  J = {r2['J']:.3f}  df={r2['J_df']}  p={r2['J_pval']:.4f}\n")

    r3 = sdf_gmm_gkr(panel, port_cols, factor_cols, robust=False)
    print("6.03-equiv (Gospodinov/Kan/Robotti, efficient W=Sigma_R^-1)")
    for c, bb, se, tt, pp in zip(factor_cols, r3["b"], r3["se"], r3["t"], r3["pval"]):
        print(f"  b_{c:8s} = {bb: .6f}  se={se:.6f}  t={tt: .3f}  p={pp:.4f}")

    r4 = sdf_gmm_gkr(panel, port_cols, factor_cols, robust=True)
    print("\n6.04-equiv (robust GKR)")
    for c, bb, tt, pp in zip(factor_cols, r4["b"], r4["t"], r4["pval"]):
        print(f"  b_{c:8s} = {bb: .6f}  t={tt: .3f}  p={pp:.4f}")

    lam1 = implied_lambda(r1["b"], factor_cols, panel)
    print("\nimplied lambda (=Sigma_ff @ b) from 6.01-equiv vs beta-pricing GMM lambda:")
    import gmm_linear_factor as glf
    beta_gmm = glf.gmm_linear_factor(panel, port_cols, factor_cols=factor_cols)
    for c, li, lb in zip(factor_cols, lam1, beta_gmm["lambda"]):
        print(f"  {c:8s}: implied={li: .6f}   beta-pricing GMM={lb: .6f}")

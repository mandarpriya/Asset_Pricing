"""
GMM estimation of a linear beta-pricing factor model -- built from scratch
with numpy/scipy only (no Kevin Sheppard's `linearmodels.asset_pricing`
GMM classes). Every step -- moment conditions, the Jacobian, the spectral
density (weighting) matrix, the sandwich standard errors, and Hansen's
J-test -- is written out explicitly below, following Cochrane (2005),
"Asset Pricing", Chapter 12 ("GMM and Two-Pass Cross-Sectional Regression").

CONTEXT: Doukas & Han (2021) do NOT use GMM anywhere (confirmed by a full
re-read of all 36 pages, tables, footnotes and appendix) -- they use
Fama-MacBeth two-pass regressions with the Shanken (1992) correction, and
report the Lewellen-Nagel-Shanken GLS R^2 as a fit DIAGNOSTIC, not an
estimator. This module is not a replication of anything in the paper; it's
a from-scratch, pedagogical GMM estimator built to double-check whether a
prior GMM analysis (done in 2024 with Kevin Sheppard's linearmodels
library) was actually doing what GMM is supposed to do.

THE MODEL (beta-pricing representation of a K-factor linear model)
--------------------------------------------------------------------
    R_t^e   = a + B f_t + e_t                    (time series, per asset i)
    E[R_i^e] = lambda_0 + beta_i' lambda           (cross section)

  R_t^e : N x 1 excess returns of N test-asset portfolios at time t
  f_t   : K x 1 factor realizations at time t
  a     : N x 1 asset-specific time-series intercepts
  B     : N x K matrix of factor loadings (betas), row i = beta_i
  lambda_0 : scalar cross-sectional intercept ("zero-beta" premium)
  lambda   : K x 1 factor risk premia -- the object of interest

STEP 1 -- moment conditions
------------------------------
All parameters theta = (a, vec(B), lambda_0, lambda) are pinned down by ONE
stacked GMM system g_T(theta) = 0:

  (a) Time-series moments (first stage) -- N*(K+1) of them, ONE set per
      asset i, using [1, f_t'] as instruments for the OLS residual:
        g1_i(theta) = (1/T) sum_t (R_i,t^e - a_i - beta_i'f_t) * [1, f_t]' = 0
      This EXACTLY identifies (a_i, beta_i) for each i -- N*(K+1) equations
      for N*(K+1) unknowns, so this step is just per-asset OLS time-series
      regression (see first_stage_betas() in sentiment_capm.py -- same
      thing, just not phrased as a moment condition there).

  (b) Cross-sectional moments (second stage) -- N of them, one per asset,
      requiring the AVERAGE pricing error of each asset to be zero once we
      pick the right (lambda_0, lambda):
        g2_i(theta) = (1/T) sum_t (R_i,t^e - lambda_0 - beta_i'lambda) = 0
      This is N equations for only (K+1) unknowns (lambda_0, lambda) --
      OVERIDENTIFIED by N-K-1 degrees of freedom. That overidentification
      is exactly what the J-test below examines: are the N-K-1 "extra"
      average pricing errors jointly indistinguishable from zero?

STEP 2 -- point estimates
----------------------------
 The time-series block is exactly identified, so (a_i, beta_i) = OLS,
independent of any weighting matrix. Plugging those betas into the
cross-sectional block and minimizing g2'Wg2 with identity weighting W=I
gives the SAME lambda as the paper's Fama-MacBeth procedure (average of
T monthly cross-sectional OLS regressions) -- a known equivalence, since
averaging a linear-in-Y estimator over t equals the estimator applied to
the time-averaged Y, when the regressors (beta) don't vary over t. We
verify this numerically below as a sanity check.

STEP 3 -- correct standard errors (the sandwich formula)
-------------------------------------------------------------
 Treating beta as *known* and running a single cross-sectional OLS of
average returns on beta (ignoring that beta itself was estimated from the
same data) UNDERSTATES lambda's standard errors. GMM fixes this
automatically via the delta-method "sandwich" formula -- the fully general
version of what Shanken (1992) approximates in closed form under
IID-normal assumptions:

    Avar(theta_hat) = (1/T) * (D'WD)^-1 D'WSWD (D'WD)^-1

  D = d gbar(theta)/d theta'   (Jacobian of the moments at theta_hat)
  S = long-run covariance of the moment conditions (Newey-West HAC, since
      monthly-return-based moments can be weakly autocorrelated)
  W = weighting matrix used to estimate theta (identity here -- Cochrane's
      own recommendation for cross-sectional asset pricing GMM, since
      "efficient" re-weighting is known to behave poorly in finite samples
      in this literature; we still compute the efficient version separately
      for the J-test, which REQUIRES it)

STEP 4 -- overidentification (Hansen's J) test
--------------------------------------------------
Re-estimate theta a second time with the EFFICIENT weighting matrix
W = S^-1 ("two-step GMM"). Then
    J = T * gbar(theta_2step)' @ S^-1 @ gbar(theta_2step)
is asymptotically chi-square with (N-K-1) degrees of freedom under the null
that the model is correctly specified. This plays the same role as
chi2_joint_test() in sentiment_capm.py, derived here from first principles
instead of the alpha-covariance formula used there.
"""
import numpy as np
import pandas as pd
import sentiment_capm as sc


def gmm_linear_factor(panel, port_cols, factor_cols=("s_lag", "mkt_rf", "s_mkt"), nw_lags=3):
    R = panel[port_cols].values                      # T x N
    F = panel[list(factor_cols)].values               # T x K
    T, N = R.shape
    K = F.shape[1]
    Fc = np.column_stack([np.ones(T), F])              # T x (K+1) instruments for TS block

    # ---- Step 1: exactly-identified first-stage OLS (time series) ----
    FcTFc = Fc.T @ Fc / T                               # (K+1) x (K+1)
    FcTFc_inv = np.linalg.inv(FcTFc)
    AB = (FcTFc_inv @ (Fc.T @ R) / T).T                 # N x (K+1): col0=a_i, cols1..K=beta_i
    a = AB[:, 0]
    B = AB[:, 1:]                                       # N x K

    # ---- Step 2: cross-sectional GMM, identity weight (= OLS on avg returns) ----
    Xc = np.column_stack([np.ones(N), B])               # N x (K+1)
    Rbar = R.mean(axis=0)
    lam = np.linalg.lstsq(Xc, Rbar, rcond=None)[0]       # (K+1,)
    lambda0, lam_k = lam[0], lam[1:]

    # ---- Step 3: stack theta and build the T x nmom moment matrix G ----
    ndim_ts = N * (K + 1)
    ndim_xs = K + 1
    ntheta = ndim_ts + ndim_xs
    nmom = ndim_ts + N

    resid_ts = R - Fc @ AB.T                             # T x N
    ts_moments = (resid_ts[:, :, None] * Fc[:, None, :]).reshape(T, N * (K + 1))
    fitted_xs = lambda0 + B @ lam_k                       # N,
    xs_moments = R - fitted_xs[None, :]                   # T x N
    G = np.column_stack([ts_moments, xs_moments])          # T x nmom

    # ---- Jacobian D = d gbar/d theta ----
    D = np.zeros((nmom, ntheta))
    for i in range(N):
        r0 = i * (K + 1)
        D[r0:r0 + (K + 1), r0:r0 + (K + 1)] = -FcTFc      # TS block: block-diagonal, N copies of -E[Fc Fc']
    xs_row0 = ndim_ts
    for i in range(N):
        col_beta0 = i * (K + 1) + 1
        D[xs_row0 + i, col_beta0:col_beta0 + K] = -lam_k   # d g2_i / d beta_i
        D[xs_row0 + i, ndim_ts] = -1.0                      # d g2_i / d lambda_0
        D[xs_row0 + i, ndim_ts + 1:ndim_ts + 1 + K] = -B[i, :]  # d g2_i / d lambda

    # ---- long-run covariance S (Newey-West HAC) ----
    Gc = G - G.mean(axis=0, keepdims=True)
    S = (Gc.T @ Gc) / T
    for lag in range(1, nw_lags + 1):
        w = 1 - lag / (nw_lags + 1)
        Gamma = (Gc[lag:].T @ Gc[:-lag]) / T
        S += w * (Gamma + Gamma.T)

    # ---- Step 1 (identity-weighted) sandwich covariance ----
    W1 = np.eye(nmom)
    bread1 = np.linalg.pinv(D.T @ W1 @ D)
    Avar1 = bread1 @ (D.T @ W1 @ S @ W1 @ D) @ bread1 / T
    se_theta1 = np.sqrt(np.clip(np.diag(Avar1), 0, None))
    se_lambda0_1 = se_theta1[ndim_ts]
    se_lam_1 = se_theta1[ndim_ts + 1:]

    # ---- Step 2 (efficient, two-step) GMM for the J-test ----
    # Re-minimize gbar'@S^-1@gbar. TS block stays OLS (exactly identified,
    # doesn't depend on W); re-solve only the XS block with W = S^-1's
    # relevant sub-block via a GLS-type cross-sectional regression.
    S_xs = S[ndim_ts:, ndim_ts:]                          # N x N block for the XS moments
    S_xs_inv = np.linalg.pinv(S_xs)
    XtSX = Xc.T @ S_xs_inv @ Xc
    XtSY = Xc.T @ S_xs_inv @ Rbar
    lam2 = np.linalg.solve(XtSX, XtSY)
    lambda0_2, lam_k_2 = lam2[0], lam2[1:]
    fitted_xs_2 = lambda0_2 + B @ lam_k_2
    gbar_xs_2 = (R - fitted_xs_2[None, :]).mean(axis=0)     # N,  (TS moments are ~0 by construction)

    J_stat = T * gbar_xs_2 @ S_xs_inv @ gbar_xs_2
    df_J = N - (K + 1)
    from scipy import stats as sstats
    J_pval = 1 - sstats.chi2.cdf(J_stat, df_J)

    return {
        "a": a, "B": B, "lambda0": lambda0, "lambda": lam_k,
        "se_lambda0": se_lambda0_1, "se_lambda": se_lam_1,
        "lambda0_2step": lambda0_2, "lambda_2step": lam_k_2,
        "J_stat": J_stat, "J_df": df_J, "J_pval": J_pval,
        "S": S, "D": D, "factor_cols": list(factor_cols),
    }


if __name__ == "__main__":
    import sys
    kind = sys.argv[1] if len(sys.argv) > 1 else "bw"
    port_kind = sys.argv[2] if len(sys.argv) > 2 else "25"
    panel, port_cols = sc.build_panel(kind, port_kind)
    out = gmm_linear_factor(panel, port_cols)

    factor_cols = out["factor_cols"]
    print(f"\n{'='*78}\nGMM LINEAR FACTOR MODEL  --  sentiment = {kind.upper()}  ({port_kind}-portfolio test assets)\n{'='*78}")
    print(f"T = {len(panel)} months, N = {len(port_cols)} portfolios, K = {len(factor_cols)} factors\n")

    print("-- Step 1 GMM (identity-weighted): point estimates + sandwich SE --")
    tbl = pd.DataFrame({
        "lambda": [out["lambda0"]] + list(out["lambda"]),
        "se": [out["se_lambda0"]] + list(out["se_lambda"]),
    }, index=["const"] + factor_cols)
    tbl["t"] = tbl["lambda"] / tbl["se"]
    print(tbl.round(4))

    print("\n-- Step 2 GMM (efficient, S^-1 weighted): point estimates --")
    tbl2 = pd.DataFrame({
        "lambda_2step": [out["lambda0_2step"]] + list(out["lambda_2step"]),
    }, index=["const"] + factor_cols)
    print(tbl2.round(4))

    print(f"\nHansen's J-test (overidentification): J = {out['J_stat']:.2f}  df = {out['J_df']}  p = {out['J_pval']:.4f}")

    # ---- validation: does GMM's step-1 lambda match the paper's Fama-MacBeth? ----
    fm = sc.run(kind, portfolio_kind=port_kind)
    print("\n-- Validation: GMM step-1 lambda vs. Fama-MacBeth (should match closely) --")
    fm_lam = fm["lambda_table"]["lambda"]
    compare = pd.DataFrame({
        "GMM_step1": [out["lambda0"]] + list(out["lambda"]),
        "FamaMacBeth": [fm_lam["const"], fm_lam["s_lag"], fm_lam["mkt_rf"], fm_lam["s_mkt"]],
    }, index=["const"] + factor_cols)
    compare["diff"] = compare["GMM_step1"] - compare["FamaMacBeth"]
    print(compare.round(6))

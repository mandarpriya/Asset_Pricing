"""
Sentiment-scaled CAPM (Doukas & Han, 2021, European Financial Management) --
independent re-implementation for personal replication.

Methodology followed (from the published paper, all 36 pages read):
  1st stage (full-sample time series, per test portfolio i):
      R_i,t - rf_t = a_i + b_s,i * s_{t-1} + b_m,i * MktRF_t
                        + b_sm,i * (s_{t-1} * MktRF_t) + e_i,t
  2nd stage (Fama-MacBeth, cross-sectional, one regression per month t):
      R_i,t - rf_t = c_t + lam_s,t * b_s,i + lam_m,t * b_m,i
                        + lam_sm,t * b_sm,i + alpha_i,t
  Risk premia = time-series average of the monthly lambda_t's.
  Classical Fama-MacBeth SE: sample std of lambda_t / sqrt(T) (paper's footnote 16).
  Shanken (1992) correction: SE_shanken = SE_FM * sqrt(1 + lam' Sigma_f^-1 lam),
      Sigma_f = covariance matrix of the realized [s_{t-1}, MktRF_t, s_{t-1}*MktRF_t].
  R^2: unadjusted / adjusted cross-sectional OLS R^2 on average returns.
  GLS R^2 (Lewellen, Nagel & Shanken 2010): 1 - (alpha'Sigma^-1 alpha) /
      ((Rbar - Rbar_gls*iota)'Sigma^-1(Rbar - Rbar_gls*iota)), Sigma = sample
      covariance matrix of the N portfolios' monthly returns.
  Chi-square joint pricing-error test: alpha' pinv(Cov(alpha)) alpha ~ chi2(N-K),
      Cov(alpha) = (1/T)(I-P)Sigma_eps(I-P)', P = projection onto [1,betas],
      Sigma_eps = covariance of first-stage residuals (paper's footnote 18).
  State beta: B_i = b_m,i + b_sm,i * mean(s_t | state), state = month where
      standardized sentiment is >=+1sd (good) or <=-1sd (bad) of its own mean
      (Lettau & Ludvigson 2001b convention, as the paper uses). "State beta" =
      B_i(bad) - B_i(good). Then regress realized average returns on beta,
      White (HC1) robust SE.

IMPORTANT CAVEATS -- read before trusting numbers against the published tables:
  - Different data than the paper: test assets/factors are Ken French's current
    25 Size-BM portfolios (refreshed through 2026), not the paper's 1965-2015
    vintage; CRSP value-weighted market return substituted with Ken French's
    Mkt-RF; sentiment source(s) differ (see below).
  - "MCSI" here is the University of Michigan Index of Consumer Sentiment
    pulled directly from sca.isr.umich.edu (their own official table), with
    the same quarterly->monthly linear interpolation the paper describes for
    the pre-1978 period.
  - "PMI" is the ISM Manufacturing PMI (investing.com export, Dec 1969-Aug
    2026) -- NOT one of the paper's four indices (BW, MCSI, CBCCI, AS). It's
    included only as an extra, informal robustness check the user asked for.
  - Baker-Wurgler (BW) and Conference Board (CBCCI) indices are NOT included
    (not sourced yet -- BW needs Wurgler's static file, CBCCI is paywalled
    beyond a short public history).
  - This is a faithful-to-the-paper *implementation*, not a byte-for-byte
    reproduction: exact table values will differ from Doukas & Han (2021).
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_DIR = "00_Data"


# ---------------------------------------------------------------- data I/O --
def load_portfolios(path=f"{DATA_DIR}/clean_25_Portfolios_5x5.csv"):
    df = pd.read_csv(path, parse_dates=["Date"])
    df["ym"] = df["Date"].dt.year * 100 + df["Date"].dt.month
    df = df.drop(columns=["Date"]).set_index("ym")
    return df  # 25 columns, monthly returns in percent


def load_portfolios_32(path=f"{DATA_DIR}/clean_32_Portfolios_OP_INV.csv"):
    """Ken French 32 portfolios formed on Size x Operating Profitability x
    Investment (2x4x4), value-weighted returns. Alternative test-asset set
    to the 25 Size-BM portfolios, for a robustness check (recovered from the
    user's own ff32_panel.csv, long-format VW/EW returns since Jul 1963)."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df["ym"] = df["Date"].dt.year * 100 + df["Date"].dt.month
    df = df.drop(columns=["Date"]).set_index("ym")
    return df  # 32 columns, monthly returns in percent


_PORT_LOADERS = {"25": load_portfolios, "32": load_portfolios_32}


def load_factors(path=f"{DATA_DIR}/clean_F-F_Research_Data_Factors.csv"):
    df = pd.read_csv(path, parse_dates=["Date"])
    df["ym"] = df["Date"].dt.year * 100 + df["Date"].dt.month
    return df.drop(columns=["Date"]).set_index("ym")  # Mkt-RF, SMB, HML, RF


def load_mcsi(path=f"{DATA_DIR}/mcsi_raw.csv"):
    """University of Michigan ICS: quarterly through 1977, monthly from 1978.
    Linearly interpolate to monthly, as the paper does (Section 3.1)."""
    raw = pd.read_csv(path)
    month_num = {m: i for i, m in enumerate(
        ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"], start=1)}
    raw["ym"] = raw["YYYY"] * 100 + raw["Month"].map(month_num)
    raw = raw.sort_values("ym").set_index("ym")["ICS_ALL"]
    full_index = range(raw.index.min(), raw.index.max() + 1)
    full_index = [ym for ym in range(raw.index.min() // 100 * 100 + 1, raw.index.max() + 1)
                  if 1 <= ym % 100 <= 12]
    s = raw.reindex(full_index)
    # interpolate on a true time axis (months), not the coded integer ym
    dt_index = pd.to_datetime(pd.Series(full_index).astype(str), format="%Y%m")
    s.index = dt_index
    s = s.interpolate(method="linear").bfill().ffill()
    s.index = full_index
    return s.rename("sentiment")


def load_pmi(path=f"{DATA_DIR}/pmi_ism_raw.csv"):
    df = pd.read_csv(path).sort_values("ym").set_index("ym")["pmi"]
    return df.rename("sentiment")


def load_bw(path=f"{DATA_DIR}/bw_raw.csv"):
    """Baker & Wurgler sentiment index, orthogonalized version (SENT_ORTH),
    from the officially maintained update (github.com/BWInvestorSentimentIndex).
    This is the exact series the paper uses -- monthly since July 1965,
    already orthogonalized to macro variables, no interpolation needed."""
    df = pd.read_csv(path).rename(columns={"yearmo": "ym"}).sort_values("ym").set_index("ym")
    return df["SENT_ORTH"].dropna().rename("sentiment")


def load_cbcci(path=f"{DATA_DIR}/cbcci_raw.csv"):
    """Conference Board Consumer Confidence Index (investing.com export,
    same [Date, Time, Actual, Forecast, Previous] layout as the PMI file)."""
    df = pd.read_csv(path).sort_values("ym").set_index("ym")["cbcci"]
    return df.rename("sentiment")


def load_cfnai(path=f"{DATA_DIR}/cfnai_raw.csv"):
    """Chicago Fed National Activity Index (chicagofed.org export, main
    CFNAI series). NOT one of the paper's four indices -- a business-cycle
    activity gauge, not an investor-sentiment survey. Included as an extra
    robustness check the user asked for."""
    df = pd.read_csv(path).sort_values("ym").set_index("ym")["cfnai"]
    return df.rename("sentiment")


_LOADERS = {"mcsi": load_mcsi, "pmi": load_pmi, "bw": load_bw,
            "cbcci": load_cbcci, "cfnai": load_cfnai}


# ------------------------------------------------------------- panel build --
def build_panel(sentiment_kind="mcsi", portfolio_kind="25"):
    ports = _PORT_LOADERS[portfolio_kind]()
    facs = load_factors()
    sent = _LOADERS[sentiment_kind]()

    panel = ports.join(facs, how="inner").join(sent, how="inner")
    panel = panel.sort_index()
    # standardize sentiment over the estimation sample (mean 0, sd 1), as in the paper
    panel["sentiment_z"] = (panel["sentiment"] - panel["sentiment"].mean()) / panel["sentiment"].std()
    panel["s_lag"] = panel["sentiment_z"].shift(1)
    panel["mkt_rf"] = panel["Mkt-RF"]
    panel["s_mkt"] = panel["s_lag"] * panel["mkt_rf"]
    panel = panel.dropna(subset=["s_lag", "s_mkt"])

    port_cols = list(ports.columns)
    for c in port_cols:
        panel[c] = panel[c] - panel["RF"]  # excess returns
    return panel, port_cols


# ----------------------------------------------------------- first stage ----
def first_stage_betas(panel, port_cols, factor_cols=("s_lag", "mkt_rf", "s_mkt")):
    betas = {}
    resid = {}
    for c in port_cols:
        X = sm.add_constant(panel[list(factor_cols)])
        fit = sm.OLS(panel[c], X).fit()
        betas[c] = fit.params[list(factor_cols)].values
        resid[c] = fit.resid.values
    beta_mat = pd.DataFrame(betas, index=list(factor_cols)).T  # N x K
    resid_mat = pd.DataFrame(resid, index=panel.index)          # T x N
    return beta_mat, resid_mat


# ---------------------------------------------------------- second stage ----
def fama_macbeth(panel, port_cols, beta_mat):
    X = sm.add_constant(beta_mat.values)  # N x (K+1), same every month
    lambdas = []
    for t, row in panel[port_cols].iterrows():
        y = row.values
        fit = sm.OLS(y, X).fit()
        lambdas.append(fit.params)
    lam_df = pd.DataFrame(lambdas, index=panel.index,
                           columns=["const"] + list(beta_mat.columns))
    lam_mean = lam_df.mean()
    T = len(lam_df)
    se_fm = lam_df.std(ddof=1) / np.sqrt(T)
    t_fm = lam_mean / se_fm
    return lam_df, lam_mean, se_fm, t_fm


def shanken_correction(lam_mean, panel, factor_cols=("s_lag", "mkt_rf", "s_mkt")):
    lam = lam_mean[list(factor_cols)].values
    Sigma_f = np.cov(panel[list(factor_cols)].values.T)
    mult = 1 + lam @ np.linalg.solve(Sigma_f, lam)
    return np.sqrt(mult)  # multiply classical FM SE (of the non-constant lambdas) by this


# ------------------------------------------------------------------ fit R^2 -
def cross_sectional_fit(panel, port_cols, beta_mat, lam_mean, factor_cols=("s_lag", "mkt_rf", "s_mkt")):
    avg_ret = panel[port_cols].mean()
    fitted = lam_mean["const"] + beta_mat[list(factor_cols)].values @ lam_mean[list(factor_cols)].values
    fitted = pd.Series(fitted, index=port_cols)
    alpha = avg_ret - fitted
    ss_res = (alpha ** 2).sum()
    ss_tot = ((avg_ret - avg_ret.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    N, K = len(port_cols), len(factor_cols)
    r2_adj = 1 - (1 - r2) * (N - 1) / (N - K - 1)
    return avg_ret, fitted, alpha, r2, r2_adj


def gls_r2(panel, port_cols, avg_ret, fitted):
    Sigma = panel[port_cols].cov().values
    Sigma_inv = np.linalg.pinv(Sigma)
    alpha = (avg_ret - fitted).values
    iota = np.ones(len(port_cols))
    Rbar = avg_ret.values
    Rbar_gls_mean = (iota @ Sigma_inv @ Rbar) / (iota @ Sigma_inv @ iota)
    num = alpha @ Sigma_inv @ alpha
    den = (Rbar - Rbar_gls_mean * iota) @ Sigma_inv @ (Rbar - Rbar_gls_mean * iota)
    return 1 - num / den


def chi2_joint_test(panel, port_cols, beta_mat, alpha, resid_mat, factor_cols=("s_lag", "mkt_rf", "s_mkt")):
    T = len(panel)
    N, K = len(port_cols), len(factor_cols)
    X = sm.add_constant(beta_mat[list(factor_cols)].values)  # N x (K+1)
    P = X @ np.linalg.pinv(X.T @ X) @ X.T
    I = np.eye(N)
    Sigma_eps = resid_mat.cov().values
    cov_alpha = (1 / T) * (I - P) @ Sigma_eps @ (I - P).T
    a = alpha.values
    stat = a @ np.linalg.pinv(cov_alpha) @ a
    from scipy import stats as sstats
    df = N - K
    pval = 1 - sstats.chi2.cdf(stat, df)
    return stat, df, pval


# -------------------------------------------------------------- state beta --
def state_beta_analysis(panel, port_cols, beta_mat, threshold="1sd"):
    s = panel["sentiment_z"]
    if threshold == "1sd":
        good = s >= 1.0
        bad = s <= -1.0
    else:  # mean split
        good = s >= 0
        bad = s < 0
    s_good_mean = panel.loc[good, "sentiment_z"].mean()
    s_bad_mean = panel.loc[bad, "sentiment_z"].mean()

    b_m = beta_mat["mkt_rf"]
    b_sm = beta_mat["s_mkt"]
    beta_good = b_m + b_sm * s_good_mean
    beta_bad = b_m + b_sm * s_bad_mean
    state_beta = beta_bad - beta_good  # paper's convention: bad - good

    ret_good = panel.loc[good, port_cols].mean()
    ret_bad = panel.loc[bad, port_cols].mean()
    ret_all = panel[port_cols].mean()
    ret_market_beta = beta_mat["mkt_rf"]  # static market beta proxy (b_m only)

    def slope_reg(x, y):
        X = sm.add_constant(x.values)
        fit = sm.OLS(y.values, X).fit(cov_type="HC1")
        return fit.params[1], fit.tvalues[1], fit.rsquared

    out = {}
    out["state_beta_vs_avg_return"] = slope_reg(state_beta, ret_all)
    out["bad_beta_vs_bad_return"] = slope_reg(beta_bad, ret_bad)
    out["good_beta_vs_good_return"] = slope_reg(beta_good, ret_good)
    out["static_beta_vs_avg_return"] = slope_reg(ret_market_beta, ret_all)
    table = pd.DataFrame({
        "market_beta": b_m, "state_beta": state_beta,
        "beta_good_state": beta_good, "beta_bad_state": beta_bad,
        "ret_all": ret_all, "ret_good": ret_good, "ret_bad": ret_bad,
    })
    return out, table


# ------------------------------------------------------------------- driver -
def run(sentiment_kind="mcsi", threshold="1sd", portfolio_kind="25"):
    panel, port_cols = build_panel(sentiment_kind, portfolio_kind)
    beta_mat, resid_mat = first_stage_betas(panel, port_cols)
    lam_df, lam_mean, se_fm, t_fm = fama_macbeth(panel, port_cols, beta_mat)
    shanken_mult = shanken_correction(lam_mean, panel)
    se_shanken = se_fm.copy()
    se_shanken[["s_lag", "mkt_rf", "s_mkt"]] *= shanken_mult
    t_shanken = lam_mean / se_shanken

    avg_ret, fitted, alpha, r2, r2_adj = cross_sectional_fit(panel, port_cols, beta_mat, lam_mean)
    r2gls = gls_r2(panel, port_cols, avg_ret, fitted)
    chi2, chi2_df, chi2_p = chi2_joint_test(panel, port_cols, beta_mat, alpha, resid_mat)
    state_out, state_table = state_beta_analysis(panel, port_cols, beta_mat, threshold)

    print(f"\n{'='*78}\nSENTIMENT-SCALED CAPM  --  sentiment = {sentiment_kind.upper()}"
          f"  ({threshold} good/bad split, {portfolio_kind}-portfolio test assets)\n{'='*78}")
    print(f"Sample: {panel.index.min()} - {panel.index.max()}  (T = {len(panel)} months, N = {len(port_cols)} portfolios)\n")

    print("-- Fama-MacBeth cross-sectional regression (Table 3 analogue) --")
    tbl = pd.DataFrame({
        "lambda": lam_mean, "t_FM": t_fm, "SE_FM": se_fm,
        "t_Shanken": t_shanken, "SE_Shanken": se_shanken,
    })
    print(tbl.round(4))
    print(f"\nR^2 (unadjusted): {r2:.4f}   R^2 (adjusted): {r2_adj:.4f}   R^2 (GLS): {r2gls:.4f}")
    print(f"Chi2 joint pricing-error test: {chi2:.2f}  (df={chi2_df}, p={chi2_p:.4f})")

    print("\n-- State-beta security market line (Table 5-7 analogue) --")
    for k, (b, t, r2_) in state_out.items():
        print(f"  {k:32s}  slope={b:8.4f}   t={t:7.2f}   R2={r2_:.3f}")

    return {
        "panel": panel, "port_cols": port_cols, "beta_mat": beta_mat,
        "lambda_table": tbl, "r2": r2, "r2_adj": r2_adj, "r2_gls": r2gls,
        "chi2": (chi2, chi2_df, chi2_p), "state_out": state_out,
        "state_table": state_table,
    }


if __name__ == "__main__":
    import sys
    kind = sys.argv[1] if len(sys.argv) > 1 else "mcsi"
    port_kind = sys.argv[2] if len(sys.argv) > 2 else "25"
    run(kind, portfolio_kind=port_kind)

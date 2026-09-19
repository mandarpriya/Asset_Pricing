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


def load_as():
    """Augmented Sentiment (AS) index -- the paper's fourth and headline
    index, AS_t = 0.318*BW_t + 0.443*MCI_t + 0.452*CB_t (their own PCA
    loading on the first principal component of BW, MCSI and CBCCI).
    Each component is standardized (z-scored) over the common overlap
    sample before combining, since the paper's loadings apply to
    standardized inputs. Overlap sample here is Dec 1969 - Dec 2025
    (limited by CBCCI's start date)."""
    bw = load_bw()
    mcsi = load_mcsi()
    cbcci = load_cbcci()
    df = pd.DataFrame({"bw": bw, "mcsi": mcsi, "cbcci": cbcci}).dropna()
    z = (df - df.mean()) / df.std()
    as_idx = 0.318 * z["bw"] + 0.443 * z["mcsi"] + 0.452 * z["cbcci"]
    return as_idx.rename("sentiment")


def load_pls(path=f"{DATA_DIR}/pls_sentiment_raw.csv"):
    """PLS-based sentiment index (Huang, Jiang, Tu & Zhou, 2015, RFS) --
    the orthogonalized version, from the officially maintained update
    (Fuwei Jiang's website, through Dec 2023). This is the paper's Table 13
    robustness sentiment measure -- an externally-sourced index (built via
    partial least squares instead of principal components), not something
    Doukas & Han construct themselves; they just re-use the published
    series (their footnote 26)."""
    df = pd.read_csv(path)
    s = df.set_index("yyyymm")["PLS_SENT_ORTH"]
    s.index.name = "ym"
    return s.rename("sentiment")


def load_goyal_controls(path=f"{DATA_DIR}/goyal_controls_raw.csv"):
    """Predictive-regression control variables from Amit Goyal's updated
    Welch & Goyal (2008) predictor dataset: real interest rate, term
    premium, default premium, inflation.

    Paper's exact definitions (Doukas & Han 2021, footnote 21): real rate
    = 30-day T-bill return minus inflation (matches here); default
    premium = BAA - AAA (matches here); term premium = 20-year T-bill
    MINUS 1-year T-bill (NOT matched here -- Goyal's file has no 1-year
    T-bill series, only tbl [~3-month] and lty [~20-year govt bond yield],
    so this uses the standard lty - tbl term spread as an approximation).
    CAY (Lettau-Ludvigson consumption-wealth ratio) is the paper's 5th
    control (their own equation notation says Sum_{i=1}^{4} despite the
    prose listing 5 -- an inconsistency in the paper itself); CAY isn't in
    Goyal's file and is omitted here."""
    df = pd.read_csv(path)
    df = df.set_index("yyyymm")
    df.index.name = "ym"
    return df  # real_rate, term_premium, default_premium, inflation


_LOADERS = {"mcsi": load_mcsi, "pmi": load_pmi, "bw": load_bw,
            "cbcci": load_cbcci, "cfnai": load_cfnai, "as": load_as, "pls": load_pls}


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


# --------------------------------------------------- scaled FF3 (Table 11) --
def build_panel_ff3(sentiment_kind="mcsi", portfolio_kind="25"):
    """Panel for the paper's Table 11 spec: sentiment-scaled FF3, i.e. the
    first-stage regressors are s_{t-1}*MktRF_t, s_{t-1}*SMB_t, s_{t-1}*HML_t
    -- NOT a plain s_{t-1} level term (unlike the base scaled-CAPM Eq.9)."""
    panel, port_cols = build_panel(sentiment_kind, portfolio_kind)
    panel["s_smb"] = panel["s_lag"] * panel["SMB"]
    panel["s_hml"] = panel["s_lag"] * panel["HML"]
    return panel, port_cols


def run_ff3(sentiment_kind="mcsi", portfolio_kind="25"):
    """Table 11 analogue: E_t(R_i,t+1) = rf + b^s_i,m*lam^s_m + b^s_i,smb*lam^s_smb
    + b^s_i,hml*lam^s_hml, on the 25 Size-BM portfolios (paper tests all four
    sentiment indices against this spec)."""
    factor_cols = ("s_mkt", "s_smb", "s_hml")
    panel, port_cols = build_panel_ff3(sentiment_kind, portfolio_kind)
    beta_mat, resid_mat = first_stage_betas(panel, port_cols, factor_cols)
    lam_df, lam_mean, se_fm, t_fm = fama_macbeth(panel, port_cols, beta_mat)
    shanken_mult = shanken_correction(lam_mean, panel, factor_cols)
    se_shanken = se_fm.copy()
    se_shanken[list(factor_cols)] *= shanken_mult
    t_shanken = lam_mean / se_shanken

    avg_ret, fitted, alpha, r2, r2_adj = cross_sectional_fit(
        panel, port_cols, beta_mat, lam_mean, factor_cols)
    r2gls = gls_r2(panel, port_cols, avg_ret, fitted)

    print(f"\n{'='*78}\nSENTIMENT-SCALED FF3 (Table 11 analogue)  --  sentiment = {sentiment_kind.upper()}"
          f"  ({portfolio_kind}-portfolio test assets)\n{'='*78}")
    print(f"Sample: {panel.index.min()} - {panel.index.max()}  (T = {len(panel)} months, N = {len(port_cols)} portfolios)\n")
    tbl = pd.DataFrame({
        "lambda": lam_mean, "t_FM": t_fm, "SE_FM": se_fm,
        "t_Shanken": t_shanken, "SE_Shanken": se_shanken,
    })
    print(tbl.round(4))
    print(f"\nR^2 (unadjusted): {r2:.4f}   R^2 (adjusted): {r2_adj:.4f}   R^2 (GLS): {r2gls:.4f}")

    return {
        "panel": panel, "port_cols": port_cols, "beta_mat": beta_mat,
        "lambda_table": tbl, "r2": r2, "r2_adj": r2_adj, "r2_gls": r2gls,
    }


# --------------------------------------------- anomaly portfolios (Tables 8/9) --
# PARTIAL replication only. The paper's Table 8/9 use 8 Stambaugh, Yu & Yuan
# (2012) anomalies: asset growth, net operating assets, net stock issues,
# total accruals, composite equity issuance, investment-to-assets, return on
# equity, and failure probability -- built from CRSP/Compustat via WRDS. No
# WRDS access here, and Stambaugh's own public data page only offers the
# composite MISP score / a 4-factor MGMT-PERF model, not per-anomaly decile
# returns. Substituted with the 4 closest analogues publicly available from
# Ken French's data library: net share issues (NI), investment (INV),
# accruals (AC), and operating profitability (OP) as a stand-in for ROE --
# NOT the paper's exact variable definitions, and only 4 of the paper's 8
# anomalies. Long/short leg directions follow the standard convention in
# this literature (the side theory predicts to earn the higher return is
# "long"): low NI/INV/AC = long, high = short; for OP it's the reverse
# (high profitability = long) since profitability is positively priced.

_ANOMALY_CODES = ["NI", "INV", "AC", "OP"]
_ANOMALY_LONG_SHORT = {
    "NI": ("Dec1", "Dec10"), "INV": ("Dec1", "Dec10"),
    "AC": ("Dec1", "Dec10"), "OP": ("Dec10", "Dec1"),
}


def load_anomaly_deciles(code):
    path = f"{DATA_DIR}/clean_{code}_deciles.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    df["ym"] = df["Date"].dt.year * 100 + df["Date"].dt.month
    df = df.drop(columns=["Date"]).set_index("ym")
    df.columns = [f"{code}_{c}" for c in df.columns]
    return df  # 10 columns (Dec1..Dec10), monthly returns in percent


def load_portfolios_anomaly_pool():
    """Pooled deciles across the 4 available anomalies (40 portfolios total
    -- the paper pools 8 anomalies x 10 deciles = 80)."""
    parts = [load_anomaly_deciles(c) for c in _ANOMALY_CODES]
    df = parts[0]
    for p in parts[1:]:
        df = df.join(p, how="inner")
    return df


_PORT_LOADERS["anomaly_pool"] = load_portfolios_anomaly_pool


def _build_single_anomaly_panel(code, sentiment_kind="as"):
    ports = load_anomaly_deciles(code)
    port_cols = list(ports.columns)
    facs = load_factors()
    sent = _LOADERS[sentiment_kind]()
    panel = ports.join(facs, how="inner").join(sent, how="inner").sort_index()
    panel["sentiment_z"] = (panel["sentiment"] - panel["sentiment"].mean()) / panel["sentiment"].std()
    panel["s_lag"] = panel["sentiment_z"].shift(1)
    panel["mkt_rf"] = panel["Mkt-RF"]
    panel["s_mkt"] = panel["s_lag"] * panel["mkt_rf"]
    panel = panel.dropna(subset=["s_lag", "s_mkt"])
    for c in port_cols:
        panel[c] = panel[c] - panel["RF"]
    return panel, port_cols


def table8_state_beta(sentiment_kind="as", threshold="1sd"):
    """Table 8 Panel B analogue: state-beta regression (Tables 6/7
    methodology) run separately on each anomaly's own 10 deciles."""
    rows = {}
    for code in _ANOMALY_CODES:
        panel, port_cols = _build_single_anomaly_panel(code, sentiment_kind)
        beta_mat, _ = first_stage_betas(panel, port_cols)
        state_out, _ = state_beta_analysis(panel, port_cols, beta_mat, threshold)
        b, t, r2 = state_out["state_beta_vs_avg_return"]
        rows[code] = {"b": b, "t": t, "R2": r2}
    return pd.DataFrame(rows).T


def table9_long_short(sentiment_kind="as", threshold="1sd"):
    """Table 9 analogue: returns and conditional betas of the long and
    short legs of each anomaly, split by good/bad sentiment state."""
    rows = []
    for code in _ANOMALY_CODES:
        panel, port_cols = _build_single_anomaly_panel(code, sentiment_kind)
        long_suffix, short_suffix = _ANOMALY_LONG_SHORT[code]
        long_col, short_col = f"{code}_{long_suffix}", f"{code}_{short_suffix}"

        beta_mat, _ = first_stage_betas(panel, port_cols)
        s = panel["sentiment_z"]
        if threshold == "1sd":
            good, bad = s >= 1.0, s <= -1.0
        else:
            good, bad = s >= 0, s < 0
        s_good_mean = panel.loc[good, "sentiment_z"].mean()
        s_bad_mean = panel.loc[bad, "sentiment_z"].mean()

        def cond_beta(col):
            b_m, b_sm = beta_mat.loc[col, "mkt_rf"], beta_mat.loc[col, "s_mkt"]
            return b_m + b_sm * s_good_mean, b_m + b_sm * s_bad_mean  # (good, bad)

        long_beta_good, long_beta_bad = cond_beta(long_col)
        short_beta_good, short_beta_bad = cond_beta(short_col)

        long_ret_good, long_ret_bad = panel.loc[good, long_col].mean(), panel.loc[bad, long_col].mean()
        short_ret_good, short_ret_bad = panel.loc[good, short_col].mean(), panel.loc[bad, short_col].mean()
        ls_ret_good = long_ret_good - short_ret_good
        ls_ret_bad = long_ret_bad - short_ret_bad

        rows.append({
            "anomaly": code,
            "long_ret_good": long_ret_good, "long_ret_bad": long_ret_bad,
            "short_ret_good": short_ret_good, "short_ret_bad": short_ret_bad,
            "long_short_good": ls_ret_good, "long_short_bad": ls_ret_bad,
            "long_beta_good": long_beta_good, "long_beta_bad": long_beta_bad,
            "short_beta_good": short_beta_good, "short_beta_bad": short_beta_bad,
        })
    return pd.DataFrame(rows).set_index("anomaly")


def run_anomaly_tables(sentiment_kind="as", threshold="1sd"):
    """Driver for the partial Table 8/9 replication (see module-level note
    above on what's substituted and why)."""
    print(f"\n{'='*78}\nANOMALY PORTFOLIOS -- PARTIAL Table 8/9 analogue  --  sentiment = {sentiment_kind.upper()}"
          f"\n(4 of the paper's 8 anomalies: NI, INV, AC, OP-as-ROE -- see module docstring)\n{'='*78}")

    print("\n-- Table 8 Panel B analogue: per-anomaly state-beta regression --")
    t8 = table8_state_beta(sentiment_kind, threshold)
    print(t8.round(4))

    print("\n-- Table 8 Panel C analogue: pooled 40-portfolio FMB cross-section --")
    pooled = run(sentiment_kind, threshold=threshold, portfolio_kind="anomaly_pool")

    print("\n-- Table 9 analogue: long/short leg returns & conditional betas by state --")
    t9 = table9_long_short(sentiment_kind, threshold)
    print(t9.round(4))

    return {"table8_state_beta": t8, "pooled_fmb": pooled, "table9": t9}


# ------------------------------------------------- predictive regression (Table 2 / 13 Panel A) --
def predictive_regression(sentiment_kind="mcsi", with_controls=True, horizon=1):
    """Table 2 / Table 13 Panel A analogue:
    MktRF_{t+1} = a + b*Sentiment_t + sum(alpha_i * Controls_t) + e_t
    (HAC/Newey-West SEs, 3 lags). NOTE: the paper's control set is real
    interest rate, inflation, term premium, default premium, AND CAY
    (Lettau-Ludvigson consumption-wealth ratio) -- CAY isn't in Goyal's
    predictor file and is omitted here (real_rate/term_premium/
    default_premium/inflation only)."""
    facs = load_factors()
    sent = _LOADERS[sentiment_kind]()
    panel = facs.join(sent, how="inner").sort_index()
    control_cols = []
    if with_controls:
        controls = load_goyal_controls()
        panel = panel.join(controls, how="inner")
        control_cols = ["real_rate", "term_premium", "default_premium", "inflation"]
    panel["sentiment_z"] = (panel["sentiment"] - panel["sentiment"].mean()) / panel["sentiment"].std()
    panel["mkt_rf_fwd"] = panel["Mkt-RF"].shift(-horizon)

    cols = ["sentiment_z"] + control_cols
    reg_df = panel.dropna(subset=["mkt_rf_fwd"] + cols)
    X = sm.add_constant(reg_df[cols])
    y = reg_df["mkt_rf_fwd"]
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    return fit, reg_df


def run_table13(sentiment_kind="pls", threshold="1sd", portfolio_kind="25"):
    """Table 13 analogue: PLS-sentiment robustness check.
    Panel A: predictive regression of next-month market excess return on
    sentiment + macro controls (CAY omitted -- see predictive_regression
    docstring). Panel B: conditional CAPM scaled by sentiment (identical
    spec to Eq.9/Table 3 -- just reuses run())."""
    print(f"\n{'='*78}\nTABLE 13 analogue  --  sentiment = {sentiment_kind.upper()}\n{'='*78}")
    print("\n-- Panel A analogue: predictive regression (CAY control omitted -- not in Goyal's data) --")
    fit, reg_df = predictive_regression(sentiment_kind, with_controls=True)
    coef = fit.params["sentiment_z"]
    tstat = fit.tvalues["sentiment_z"]
    print(f"  beta_sentiment = {coef:.4f}   t (HAC) = {tstat:.3f}   (T = {len(reg_df)})")
    print(fit.summary().tables[1])

    print("\n-- Panel B analogue: conditional CAPM scaled by sentiment (Eq.9/Table 3 spec) --")
    panelB = run(sentiment_kind, threshold=threshold, portfolio_kind=portfolio_kind)

    return {"panelA_fit": fit, "panelB": panelB}


def run_table2():
    """Table 2 analogue: predictive regression of next-month market excess
    return on lagged sentiment, Panel A (univariate, Eq.11) and Panel B
    (with controls -- real rate/term premium/default premium/inflation;
    CAY omitted, see load_goyal_controls docstring), for all four indices."""
    print(f"\n{'='*78}\nTABLE 2 analogue -- sentiment predicts next-month market excess return\n{'='*78}")
    rows_a, rows_b = {}, {}
    for kind in ["bw", "mcsi", "cbcci", "as"]:
        fit_a, df_a = predictive_regression(kind, with_controls=False)
        rows_a[kind.upper()] = {
            "beta": fit_a.params["sentiment_z"], "t": fit_a.tvalues["sentiment_z"],
            "R2_pct": 100 * fit_a.rsquared, "T": len(df_a),
        }
        fit_b, df_b = predictive_regression(kind, with_controls=True)
        rows_b[kind.upper()] = {
            "beta": fit_b.params["sentiment_z"], "t": fit_b.tvalues["sentiment_z"],
            "R2_pct": 100 * fit_b.rsquared, "T": len(df_b),
        }
    panelA = pd.DataFrame(rows_a).T
    panelB = pd.DataFrame(rows_b).T
    print("\n-- Panel A: univariate (Eq.11), HAC/Newey-West(3) SEs --")
    print(panelA.round(4))
    print("\n-- Panel B: with controls (real rate, term premium[approx], default premium, inflation) --")
    print(panelB.round(4))
    return {"panelA": panelA, "panelB": panelB}


if __name__ == "__main__":
    import sys
    kind = sys.argv[1] if len(sys.argv) > 1 else "mcsi"
    port_kind = sys.argv[2] if len(sys.argv) > 2 else "25"
    run(kind, portfolio_kind=port_kind)

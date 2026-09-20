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

SENTIMENT INDICES AVAILABLE (_LOADERS keys):
  bw     Baker-Wurgler orthogonalized index (SENT_ORTH) -- the paper's primary
         measure, and the only one that produces a well-specified model here.
  mcsi   University of Michigan consumer sentiment. Quarterly pre-1978, linearly
         interpolated to monthly exactly as the paper describes. NOTE the paper
         itself is inconsistent: its text says "MCSI" (p.215) but its Table 3
         labels the same row "MSCI".
  cbcci  Conference Board consumer confidence (investing.com export, repaired --
         see below).
  as     Augmented Sentiment: first principal component of BW/MCSI/CBCCI,
         re-estimated by PCA on THIS sample rather than reusing the paper's
         published 0.318/0.443/0.452 loadings, which were fit on theirs.
  pls    Huang-Jiang-Tu-Zhou partial-least-squares index (the Table 13 measure).
  pmi    ISM Manufacturing PMI -- NOT one of the paper's indices; extra check.
  cfnai  Chicago Fed National Activity Index -- a business-cycle activity gauge,
         not investor sentiment; included as a deliberate contrast case.

TEST ASSETS (_PORT_LOADERS keys): "25" (the paper's own), "industry48",
  "32", "anomaly_pool". The choice matters more than anything else here --
  see the note on Lewellen-Nagel-Shanken under gls_r2().

DEVIATIONS FROM THE PAPER -- read before comparing to the published tables:
  - Sample is 1969:12-2025:12 (COMMON_WINDOW) vs the paper's 1965:07-2015:09.
  - Market return is Ken French's Mkt-RF, not CRSP's value-weighted excess
    return. (Mkt-RF is the closer match of the two available: Goyal's
    CRSP_SPvw is the S&P 500, not the full CRSP universe the paper uses.)
  - CBCCI and PMI came from investing.com exports that were BROKEN: a date
    format change made ~80 months look missing, and one CBCCI row had lost
    its month annotation. Both repaired; one interpolated month each remains
    (CBCCI 2008:01, PMI 2008:02) at the format seam.
  - Term premium is lty - tbl (~20y minus ~3m); the paper wants 20y minus 1y,
    and Goyal's file has no 1-year series.
  - CAY is available via load_cay() but NOT a default control: the series ends
    in 2003Q1 here (2019Q3 even in Lettau's newest), which would truncate the
    sample severely.
  - Anomaly tables use 4 public Ken French proxies, not the paper's 8
    WRDS-built Stambaugh-Yu-Yuan measures.

This is a faithful-to-the-paper *implementation*, not a byte-for-byte
reproduction: exact table values will differ from Doukas & Han (2021). On the
25 portfolios it tracks them closely -- signs match throughout, CBCCI's lambda_m
comes out at -0.746 against their -0.74, MCSI's intercept at 1.08 vs 1.04.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA

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


def load_portfolios_industry48(path=f"{DATA_DIR}/clean_48_Industry_Portfolios.csv"):
    """Ken French 48 Industry Portfolios, value-weighted monthly returns
    (percent), Jul 1926 - present. Missing-data codes (-99.99/-999) are
    already converted to NaN in the cleaned file (some industries, e.g.
    semiconductors, didn't exist yet in the early decades -- all 48 are
    fully populated from Jul 1969 onward). Alternative test-asset set to
    the 25 Size-BM / 32 Size-OP-INV portfolios."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df["ym"] = df["Date"].dt.year * 100 + df["Date"].dt.month
    df = df.drop(columns=["Date"]).set_index("ym")
    return df  # 48 columns, monthly returns in percent (NaN where no firms)


def load_portfolios_industry12(path=f"{DATA_DIR}/clean_12_Industry_Portfolios.csv"):
    """Ken French 12 Industry Portfolios -- the closest thing in this library
    to a GICS-style sector cut: NoDur, Durbl, Manuf, Enrgy, Chems, BusEq
    (tech), Telcm, Utils, Shops, Hlth, Money (financials), Other.

    WHY THIS RATHER THAN A SCREENER. These are built from CRSP, which keeps
    delisted firms and their delisting returns, so there is no survivorship
    bias. A stock screener returns only CURRENTLY listed companies, which
    silently deletes every firm that went bankrupt, merged or delisted --
    and for sector work that is devastating, because the deletions cluster
    exactly where the interesting variation is (tech through 2000-02,
    financials through 2008). Coverage also runs from 1926 rather than
    whenever the screener's survivors happened to list.

    Fully populated -- no missing months anywhere, unlike the 48-industry
    file whose narrow buckets are empty in the early decades."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df["ym"] = df["Date"].dt.year * 100 + df["Date"].dt.month
    return df.drop(columns=["Date"]).set_index("ym")


def load_portfolios_industry10(path=f"{DATA_DIR}/clean_10_Industry_Portfolios.csv"):
    """Ken French 10 Industry Portfolios: NoDur, Durbl, Manuf, Enrgy, HiTec,
    Telcm, Shops, Hlth, Utils, Other.

    NOT simply the 12-industry set with two buckets merged -- the 10-cut
    drops Chems and Money as standalone groups and folds BusEq into a
    broader HiTec. So 10 and 12 are alternative partitions of the same
    universe, not a strict nesting."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df["ym"] = df["Date"].dt.year * 100 + df["Date"].dt.month
    return df.drop(columns=["Date"]).set_index("ym")


_PORT_LOADERS = {"25": load_portfolios, "32": load_portfolios_32,
                  "industry48": load_portfolios_industry48,
                  "industry12": load_portfolios_industry12,
                  "industry10": load_portfolios_industry10}


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


def load_as(verbose=True):
    """Augmented Sentiment (AS) index -- first principal component of
    BW, MCSI and CBCCI, via sklearn's PCA on OUR OWN standardized sample
    (NOT the paper's fixed 0.318/0.443/0.452 weights, which were fit on
    their own different sample period, so reusing them verbatim would
    be wrong)."""
    bw = load_bw()
    mcsi = load_mcsi()
    cbcci = load_cbcci()
    df = pd.DataFrame({"bw": bw, "mcsi": mcsi, "cbcci": cbcci}).dropna()
    z = (df - df.mean()) / df.std()

    pca = PCA(n_components=3)
    scores = pca.fit_transform(z.values)  # T x 3, uncorrelated components
    loadings_all = pd.DataFrame(
        pca.components_.T, index=["bw", "mcsi", "cbcci"],
        columns=["PC1", "PC2", "PC3"]
    )

    pc1_loadings = loadings_all["PC1"].copy()
    if pc1_loadings.sum() < 0:  # PCA sign is arbitrary -- flip so loadings are positive
        pc1_loadings *= -1
        scores[:, 0] *= -1

    if verbose:
        print("AS index -- PCA loadings (all 3 components):")
        print(loadings_all.round(4))
        print("Explained variance ratio:", pca.explained_variance_ratio_.round(4))
        print(f"PC1 loadings used for AS:\n{pc1_loadings.round(4)}")
        print("(paper's reported loadings, for comparison: BW=0.318, MCSI=0.443, CBCCI=0.452)")

    as_idx = pd.Series(scores[:, 0], index=z.index, name="sentiment")
    return as_idx


def load_aaii(path=f"{DATA_DIR}/aaii_raw.csv", measure="spread"):
    """AAII Sentiment Survey -- the American Association of Individual
    Investors' weekly poll asking members whether they are bullish, neutral
    or bearish on the stock market over the next six months.

    NOT one of the paper's indices, but arguably the most on-point measure
    available here: it asks investors directly about the STOCK MARKET,
    whereas MCSI and CBCCI ask households about the ECONOMY. Brown & Cliff
    (2004, 2005) are the standard references for using it this way.

    The survey is WEEKLY (Thursdays); this loader averages the weeks within
    each calendar month, so a month is the mean of its 2-5 survey readings.

    measure= picks the series:
      "spread"  (default) bullish - bearish, the standard "bull-bear
                spread" used in the literature
      "bullish" / "bearish" / "neutral" -- the raw shares

    COVERAGE WARNING: the survey begins July 1987, so using it costs 211 of
    the 673 months in COMMON_WINDOW. Any AAII result is a shorter-sample
    check, not directly comparable to the other indices' full-window runs."""
    df = pd.read_csv(path).set_index("ym").sort_index()
    if measure not in ("spread", "bullish", "bearish", "neutral"):
        raise ValueError(f"measure must be spread/bullish/bearish/neutral; got {measure!r}")
    return df[measure].rename("sentiment")


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
    = 30-day T-bill return minus inflation; default premium = BAA - AAA
    (matches here); term premium = 20-year T-bill MINUS 1-year T-bill
    (NOT matched here -- Goyal's file has no 1-year T-bill series, only
    tbl [~3-month, ANNUALIZED] and lty [~20-year govt bond yield], so this
    uses the standard lty - tbl term spread as an approximation).

    real_rate is built as Rfree - infl, both MONTHLY rates. An earlier
    version of this file used tbl - infl, which silently mixed an
    ANNUALISED 3-month rate with a MONTHLY inflation rate -- the result was
    ~100x too large and essentially just tracked the nominal rate level.
    Rfree is Goyal's 30-day T-bill return, which is what footnote 21 asks
    for. (Fixing this moved the Table 2 sentiment coefficients only
    slightly and changed no conclusions, but the old variable was not a
    real interest rate.)

    CAY (Lettau-Ludvigson consumption-wealth ratio) is the paper's 5th
    control -- their own equation notation says Sum_{i=1}^{4} despite the
    prose listing 5, an inconsistency in the paper itself. It is NOT
    included here by default because it is quarterly and the available
    series ends long before this sample does; see load_cay() and the
    controls= argument of predictive_regression()."""
    df = pd.read_csv(path)
    df = df.set_index("yyyymm")
    df.index.name = "ym"
    return df  # real_rate, term_premium, default_premium, inflation


# The 14 standard Welch-Goyal predictors, derived below in load_goyal_full()
GOYAL_FULL_COLS = ["dp", "dy", "ep", "de", "svar", "bm", "ntis",
                   "tbl", "lty", "ltr", "tms", "dfy", "dfr", "infl"]

# Two of those 14 are EXACT linear combinations of the others, by construction:
#     de  = dp - ep          (both are log(X) - log(Index), so the Index cancels)
#     tms = lty - tbl        (the term spread IS that difference)
# Verified numerically: the 14-column matrix has rank 12. Putting all 14 in one
# regression therefore leaves the coefficients non-identified (statsmodels emits
# SingularMatrixWarning and silently pseudo-inverts). So any regression on the
# "full" set uses these 12 instead. PCA is unaffected -- it handles collinear
# inputs gracefully -- so the "pca" mode still consumes all 14.
GOYAL_INDEP_COLS = [c for c in GOYAL_FULL_COLS if c not in ("de", "tms")]


def load_goyal_full(path=f"{DATA_DIR}/goyal_predictors_raw.csv"):
    """All 14 standard Welch & Goyal (2008) return predictors, derived from
    the raw PredictorData file.

    This is an EXTENSION beyond what Doukas & Han do -- their control set is
    the four in load_goyal_controls() plus CAY. Use it to test their
    robustness claim harder than they did, not to reproduce their Table 2.

    Definitions follow Welch & Goyal:
      dp   = log(D12) - log(Index)              dividend-price ratio
      dy   = log(D12) - log(Index_{t-1})        dividend yield
      ep   = log(E12) - log(Index)              earnings-price ratio
      de   = log(D12) - log(E12)                dividend payout ratio
      svar = sum of squared daily S&P returns   stock variance
      bm   = book-to-market (DJIA)
      ntis = net equity expansion
      tbl  = 3-month T-bill rate (annualised)
      lty  = long-term (~20y) govt bond yield
      ltr  = long-term govt bond return
      tms  = lty - tbl                          term spread
      dfy  = BAA - AAA                          default yield spread
      dfr  = corpr - ltr                        default return spread
      infl = CPI inflation (monthly)

    NOTE on units: tbl/lty/tms are ANNUALISED rates while infl/ltr/dfr/svar
    are monthly. OLS is scale-invariant so this does not bias anything, but
    it does mean the fitted coefficients are not comparable across
    predictors in magnitude.

    NOTE on inflation: Welch & Goyal lag infl by one month in their own
    predictive regressions, because CPI is released with a delay. That lag
    is NOT applied here (the paper doesn't mention it either); pass
    lag_infl=True to apply it.

    'csp' (cross-sectional premium) is deliberately excluded -- Welch &
    Goyal discontinued it and it has no data over most of this sample."""
    raw = pd.read_csv(path).set_index("yyyymm").sort_index()
    # Index/D12/E12 can arrive as strings with thousands separators
    for c in ["Index", "D12", "E12"]:
        raw[c] = pd.to_numeric(raw[c].astype(str).str.replace(",", ""), errors="coerce")

    d = pd.DataFrame(index=raw.index)
    d["dp"] = np.log(raw["D12"]) - np.log(raw["Index"])
    d["dy"] = np.log(raw["D12"]) - np.log(raw["Index"].shift(1))
    d["ep"] = np.log(raw["E12"]) - np.log(raw["Index"])
    d["de"] = np.log(raw["D12"]) - np.log(raw["E12"])
    d["svar"] = raw["svar"]
    d["bm"] = raw["b/m"]
    d["ntis"] = raw["ntis"]
    d["tbl"] = raw["tbl"]
    d["lty"] = raw["lty"]
    d["ltr"] = raw["ltr"]
    d["tms"] = raw["lty"] - raw["tbl"]
    d["dfy"] = raw["BAA"] - raw["AAA"]
    d["dfr"] = raw["corpr"] - raw["ltr"]
    d["infl"] = raw["infl"]
    d.index.name = "ym"
    return d[GOYAL_FULL_COLS]


def load_cay(path=f"{DATA_DIR}/cay_raw.csv"):
    """Lettau & Ludvigson consumption-wealth ratio (cay), expanded from
    quarterly to monthly.

    cay is the cointegrating residual from the long-run relationship
    between log consumption (c), log asset wealth (a) and log labour
    income (y). A high cay means consumption is high relative to wealth and
    income, which -- under consumption smoothing -- signals that investors
    expect high future returns. It is the conditioning variable Lettau &
    Ludvigson (2001b) used to scale the CAPM, i.e. the direct methodological
    ancestor of what Doukas & Han do with sentiment.

    TWO IMPORTANT LIMITATIONS:

    1. COVERAGE. The series runs 1952Q1-2019Q3 (Lettau's current public
       vintage; the formula in its header is cay = c - 0.218a - 0.801y +
       0.441). It ends six years before this project's sample does, so
       including it as a control TRUNCATES the estimation sample to ~598 of
       the 673 months in COMMON_WINDOW. That is why it is opt-in rather
       than part of the default control set.

    2. FREQUENCY. cay is quarterly by construction (NIPA consumption and
       labour income are quarterly). Each quarter's value is assigned to all
       three of its months -- a step function, NOT linear interpolation.
       Interpolating linearly would blend in the *next* quarter's value,
       which is look-ahead bias in a predictive regression. (Contrast
       load_mcsi(), where linear interpolation is fine because MCSI is a
       contemporaneous survey level, not a predictor being used to forecast
       the following month.)

    Even the step expansion is mildly optimistic about timing: quarterly
    NIPA data is released with roughly a one-month lag, so cay for Q4 is not
    truly known until late January."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    q = df.set_index("date")["cay"].sort_index()

    # date is coded YYYYQQ (e.g. 195104 = 1951Q4) -> expand to the 3 months
    rows = {}
    for code, val in q.items():
        year, quarter = int(code) // 100, int(code) % 100
        for m in range(3 * (quarter - 1) + 1, 3 * (quarter - 1) + 4):
            rows[year * 100 + m] = val
    s = pd.Series(rows, name="cay").sort_index()
    s.index.name = "ym"
    return s


MACRO_COLS = ["unrate_chg", "houst_gr", "cape_yield"]


def load_macro(unrate_path=f"{DATA_DIR}/unrate_raw.csv",
               houst_path=f"{DATA_DIR}/houst_raw.csv",
               shiller_path=f"{DATA_DIR}/shiller_raw.csv"):
    """Real-activity and valuation controls, for asking whether sentiment's
    predictive power survives once the real economy is held constant.

    This is an EXTENSION beyond Doukas & Han, who never test this.

    Three variables, each already transformed to be usable as a regressor:

      unrate_chg  12-month CHANGE in the unemployment rate (percentage
                  points). BLS via FRED (UNRATE), monthly from 1948.
      houst_gr    12-month LOG GROWTH in housing starts. Census/HUD via FRED
                  (HOUST), monthly from 1959, seasonally adjusted annual
                  rate. Housing LEADS the cycle -- breaking ground is a bet
                  on demand 9-18 months out, and it is the most
                  interest-rate-sensitive sector there is -- which makes it
                  a real competitor to sentiment rather than a straw man.
                  Contrast CFNAI and the Philadelphia Fed's USPHCI, which
                  are COINCIDENT by construction and so are the textbook
                  "already in the price" case.
      cape_yield  1 / CAPE, i.e. the cyclically-adjusted earnings yield.
                  Robert Shiller's data. CAPE smooths earnings over 10 years
                  and is the best-known long-horizon valuation predictor;
                  Goyal's 'ep' is the 1-year version, so this is additive
                  rather than redundant.

    WHY CHANGES AND YIELDS RATHER THAN LEVELS. All three underlying series
    are extremely persistent -- the unemployment rate has monthly
    autocorrelation near 0.99, housing starts swing cyclically, and CAPE
    trends. Regressing returns on a near-unit-root regressor over a period
    when markets rose manufactures significance out of nothing (Stambaugh
    bias). Note the paper defends ITSELF on exactly this point (p.220,
    citing Stambaugh et al. 2014 simulating 200 million equally persistent
    regressors); these transformations are the same defence applied here.

    REVISIONS CAVEAT: FRED serves the CURRENT revised vintage, not what was
    known in real time. Housing starts in particular get revised
    meaningfully, so the regression sees slightly better information than
    investors had. A footnote for the write-up, not a reason to avoid it.

    Coverage is bound by housing starts: 1959:01 plus 12 months for the
    growth rate means these are usable from 1960:01, comfortably before
    COMMON_WINDOW begins.

    THE 2025:10 GAP IS REAL AND IS LEFT AS NaN. The US federal funding lapse
    meant the BLS household survey was not conducted that month, so no
    unemployment rate exists for October 2025 -- it was never measured, as
    opposed to measured-and-lost. Interpolating would fabricate an
    observation, so this deliberately does not: unrate_chg is NaN for
    2025:10, and any regression using it drops exactly that one month out of
    673. Say so in a footnote rather than papering over it."""
    u = pd.read_csv(unrate_path).set_index("ym").sort_index()["unrate"]
    h = pd.read_csv(houst_path).set_index("ym").sort_index()["houst"]
    s = pd.read_csv(shiller_path).set_index("ym").sort_index()["cape"]

    out = pd.DataFrame(index=sorted(set(u.index) | set(h.index) | set(s.index)))
    out.index.name = "ym"
    out["unrate_chg"] = u.reindex(out.index) - u.reindex(out.index).shift(12)
    out["houst_gr"] = np.log(h.reindex(out.index)) - np.log(h.reindex(out.index).shift(12))
    out["cape_yield"] = 1.0 / s.reindex(out.index)
    return out[MACRO_COLS]


_LOADERS = {"mcsi": load_mcsi, "pmi": load_pmi, "bw": load_bw,
            "cbcci": load_cbcci, "cfnai": load_cfnai, "as": load_as,
            "pls": load_pls, "aaii": load_aaii}


# Common sample window across all 5 individual sentiment indices (BW, MCSI,
# CBCCI, PMI, CFNAI), the Goyal macro controls, and the 48-industry test
# assets -- determined by checking each series' actual coverage: CBCCI/PMI
# start Dec 1969 (binding on the start side), BW's officially maintained
# update ends Dec 2025 (binding on the end side). Pass this to any function
# below via date_range=COMMON_WINDOW to force every run onto the identical
# window, so results across indices/models are directly comparable rather
# than each one silently using its own maximal-available sample.
COMMON_WINDOW = (196912, 202512)


def _apply_date_range(panel, date_range):
    if date_range is None:
        return panel
    start_ym, end_ym = date_range
    return panel.loc[(panel.index >= start_ym) & (panel.index <= end_ym)]


# ------------------------------------------------------------- panel build --
def build_panel(sentiment_kind="mcsi", portfolio_kind="25", date_range=None):
    ports = _PORT_LOADERS[portfolio_kind]()
    facs = load_factors()
    sent = _LOADERS[sentiment_kind]()

    panel = ports.join(facs, how="inner").join(sent, how="inner")
    panel = panel.sort_index()
    panel = _apply_date_range(panel, date_range)
    port_cols = list(ports.columns)
    # drop any month with a missing test-asset return (e.g. an industry
    # portfolio with no firms yet in the early decades) BEFORE standardizing
    # sentiment, so the z-score isn't computed over a period we'll drop anyway
    panel = panel.dropna(subset=port_cols)
    # standardize sentiment over the estimation sample (mean 0, sd 1), as in the paper
    # -- when date_range is set, this mean/std is computed WITHIN the common
    # window, so different indices' z-scores are standardized over the same
    # period rather than each index's own idiosyncratic full history
    panel["sentiment_z"] = (panel["sentiment"] - panel["sentiment"].mean()) / panel["sentiment"].std()
    panel["s_lag"] = panel["sentiment_z"].shift(1)
    # Doukas & Han's Eq.9 timing (confirmed against the paper): the
    # regressors are LAGGED sentiment (s_lag), the CONTEMPORANEOUS market
    # return (mkt_rf, same period t as the portfolio return), and their
    # product (lagged sentiment x contemporaneous market return). Only
    # sentiment is lagged -- Mkt-RF/SMB/HML are not. This is what makes it
    # a genuine conditional CAPM: the asset's exposure to THIS period's
    # market move, scaled by sentiment known BEFORE the period started.
    panel["mkt_rf"] = panel["Mkt-RF"]
    panel["s_mkt"] = panel["s_lag"] * panel["mkt_rf"]
    panel = panel.dropna(subset=["s_lag", "s_mkt"])

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


def gls_r2(panel, port_cols, beta_mat, factor_cols=("s_lag", "mkt_rf", "s_mkt")):
    """Lewellen, Nagel & Shanken (2010) GLS cross-sectional R^2:

        1 - (a' Sigma^-1 a) / ((Rbar - Rbar_gls*iota)' Sigma^-1 (Rbar - Rbar_gls*iota))

    CRITICAL: the lambdas used to form the pricing errors 'a' are re-estimated
    here by GLS cross-sectional regression -- they are NOT the OLS/Fama-MacBeth
    lambdas reported in the lambda table. This matters because the GLS estimator
    is by construction the one that MINIMISES a'Sigma^-1 a. Feeding this formula
    alphas built from OLS lambdas leaves the numerator un-minimised, so the ratio
    can exceed 1 and the statistic can come out NEGATIVE -- which a genuine GLS
    R^2 can never be, since the model nests the constant-only benchmark that
    defines the denominator. (An earlier version of this function did exactly
    that and produced GLS R^2 of -0.20 to -0.44 on the 25 portfolios.)

    Consequence worth remembering when reading the output: the GLS R^2 describes
    the fit of the GLS estimator, while the lambda table reports FM/OLS estimates.
    That is the standard convention (it is what LNS do, and what Doukas & Han
    report in their Table 3), but the two columns do refer to different
    estimators of the same model."""
    Sigma = panel[port_cols].cov().values
    Sigma_inv = np.linalg.pinv(Sigma)
    Rbar = panel[port_cols].mean().values
    N = len(port_cols)
    iota = np.ones(N)

    X = np.column_stack([iota, beta_mat[list(factor_cols)].values])  # N x (K+1)
    lam_gls = np.linalg.solve(X.T @ Sigma_inv @ X, X.T @ Sigma_inv @ Rbar)
    alpha = Rbar - X @ lam_gls

    Rbar_gls_mean = (iota @ Sigma_inv @ Rbar) / (iota @ Sigma_inv @ iota)
    dev = Rbar - Rbar_gls_mean * iota
    return 1 - (alpha @ Sigma_inv @ alpha) / (dev @ Sigma_inv @ dev)


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
def run(sentiment_kind="mcsi", threshold="1sd", portfolio_kind="25", date_range=None):
    panel, port_cols = build_panel(sentiment_kind, portfolio_kind, date_range)
    beta_mat, resid_mat = first_stage_betas(panel, port_cols)
    lam_df, lam_mean, se_fm, t_fm = fama_macbeth(panel, port_cols, beta_mat)
    shanken_mult = shanken_correction(lam_mean, panel)
    se_shanken = se_fm.copy()
    se_shanken[["s_lag", "mkt_rf", "s_mkt"]] *= shanken_mult
    t_shanken = lam_mean / se_shanken

    avg_ret, fitted, alpha, r2, r2_adj = cross_sectional_fit(panel, port_cols, beta_mat, lam_mean)
    r2gls = gls_r2(panel, port_cols, beta_mat)
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
def build_panel_ff3(sentiment_kind="mcsi", portfolio_kind="25", date_range=None):
    """Panel for the paper's Table 11 spec: sentiment-scaled FF3, i.e. the
    first-stage regressors are s_{t-1}*MktRF_t, s_{t-1}*SMB_t, s_{t-1}*HML_t
    -- NOT a plain s_{t-1} level term (unlike the base scaled-CAPM Eq.9).
    Only sentiment is lagged; SMB/HML are contemporaneous, same timing as
    MktRF in build_panel() (see that function's docstring)."""
    panel, port_cols = build_panel(sentiment_kind, portfolio_kind, date_range)
    panel["s_smb"] = panel["s_lag"] * panel["SMB"]
    panel["s_hml"] = panel["s_lag"] * panel["HML"]
    return panel, port_cols


def run_ff3(sentiment_kind="mcsi", portfolio_kind="25", date_range=None):
    """Table 11 analogue: E_t(R_i,t+1) = rf + b^s_i,m*lam^s_m + b^s_i,smb*lam^s_smb
    + b^s_i,hml*lam^s_hml, on the 25 Size-BM portfolios (paper tests all four
    sentiment indices against this spec)."""
    factor_cols = ("s_mkt", "s_smb", "s_hml")
    panel, port_cols = build_panel_ff3(sentiment_kind, portfolio_kind, date_range)
    beta_mat, resid_mat = first_stage_betas(panel, port_cols, factor_cols)
    lam_df, lam_mean, se_fm, t_fm = fama_macbeth(panel, port_cols, beta_mat)
    shanken_mult = shanken_correction(lam_mean, panel, factor_cols)
    se_shanken = se_fm.copy()
    se_shanken[list(factor_cols)] *= shanken_mult
    t_shanken = lam_mean / se_shanken

    avg_ret, fitted, alpha, r2, r2_adj = cross_sectional_fit(
        panel, port_cols, beta_mat, lam_mean, factor_cols)
    r2gls = gls_r2(panel, port_cols, beta_mat, factor_cols)

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


# ------------------------------------------------- plain (unscaled) FF3 baseline --
def run_ff3_plain(sentiment_kind="mcsi", portfolio_kind="25", date_range=None):
    """The 'usual' Fama-French 3-factor Fama-MacBeth test -- NOT scaled by
    sentiment (contrast with run_ff3(), which is the sentiment-SCALED FF3,
    Table 11 analogue). This is a baseline: plain Mkt-RF/SMB/HML betas
    priced via standard two-pass Fama-MacBeth, sentiment plays no role in
    the regression itself. sentiment_kind only restricts the sample to the
    same overlap window used by the sentiment-scaled tests, so R^2/chi2
    are directly comparable across models on an apples-to-apples sample."""
    factor_cols = ("Mkt-RF", "SMB", "HML")
    panel, port_cols = build_panel(sentiment_kind, portfolio_kind, date_range)
    beta_mat, resid_mat = first_stage_betas(panel, port_cols, factor_cols)
    lam_df, lam_mean, se_fm, t_fm = fama_macbeth(panel, port_cols, beta_mat)
    shanken_mult = shanken_correction(lam_mean, panel, factor_cols)
    se_shanken = se_fm.copy()
    se_shanken[list(factor_cols)] *= shanken_mult
    t_shanken = lam_mean / se_shanken

    avg_ret, fitted, alpha, r2, r2_adj = cross_sectional_fit(
        panel, port_cols, beta_mat, lam_mean, factor_cols)
    r2gls = gls_r2(panel, port_cols, beta_mat, factor_cols)
    chi2, chi2_df, chi2_p = chi2_joint_test(panel, port_cols, beta_mat, alpha, resid_mat, factor_cols)

    print(f"\n{'='*78}\nPLAIN (unscaled) FAMA-FRENCH 3-FACTOR MODEL -- baseline"
          f"  ({portfolio_kind}-portfolio test assets, sample matched to {sentiment_kind.upper()})\n{'='*78}")
    print(f"Sample: {panel.index.min()} - {panel.index.max()}  (T = {len(panel)} months, N = {len(port_cols)} portfolios)\n")
    tbl = pd.DataFrame({
        "lambda": lam_mean, "t_FM": t_fm, "SE_FM": se_fm,
        "t_Shanken": t_shanken, "SE_Shanken": se_shanken,
    })
    print(tbl.round(4))
    print(f"\nR^2 (unadjusted): {r2:.4f}   R^2 (adjusted): {r2_adj:.4f}   R^2 (GLS): {r2gls:.4f}")
    print(f"Chi2 joint pricing-error test: {chi2:.2f}  (df={chi2_df}, p={chi2_p:.4f})")

    return {
        "panel": panel, "port_cols": port_cols, "beta_mat": beta_mat,
        "lambda_table": tbl, "r2": r2, "r2_adj": r2_adj, "r2_gls": r2gls,
        "chi2": (chi2, chi2_df, chi2_p),
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
def predictive_regression(sentiment_kind="mcsi", with_controls=True, horizon=1,
                          date_range=None, controls="paper", n_pc=3,
                          add_cay=False, add_macro=False):
    """Table 2 / Table 13 Panel A analogue:
    MktRF_{t+1} = a + b*Sentiment_t + sum(alpha_i * Controls_t) + e_t
    (HAC/Newey-West SEs, 3 lags).

    controls= selects WHICH control set (ignored when with_controls=False):

      "paper" (default) -- the four in load_goyal_controls(): real_rate,
          term_premium, default_premium, inflation. This is what reproduces
          Doukas & Han's Table 2 Panel B, so it stays the default.

      "full"  -- the Welch-Goyal predictors, as the 12 linearly independent
          ones (GOYAL_INDEP_COLS: 'de' and 'tms' are dropped because they
          are EXACT combinations of others -- see the note there). An
          EXTENSION beyond the paper: a much more demanding test of their
          claim that sentiment's predictive power isn't macro information in
          disguise. Caveat: even these 12 are heavily collinear (dp/dy/ep
          especially), so individual coefficients are unstable and hard to
          read, even though the sentiment coefficient and R^2 remain
          meaningful. Welch & Goyal themselves evaluate their predictors
          ONE AT A TIME rather than jointly, for exactly this reason.

      "pca"   -- the first n_pc principal components of those 14, z-scored
          first. The standard remedy for the collinearity above: it keeps
          most of the macro information while leaving the regression well
          conditioned. n_pc=3 by default.

    add_macro=True appends the real-activity/valuation block from
    load_macro(): the 12-month change in unemployment, 12-month growth in
    housing starts, and the CAPE earnings yield. This is the test of whether
    sentiment survives controlling for the real economy -- a question the
    paper never asks. Costs nothing in sample length (the block is available
    from 1960:01), except the single 2025:10 month that has no unemployment
    reading; see load_macro().

    add_cay=True appends the Lettau-Ludvigson consumption-wealth ratio.
    WARNING: this TRUNCATES the sample hard -- cay ends in 2003Q1 in the
    bundled file (2019Q3 even in Lettau's newest public series), against a
    sample that otherwise runs to 2025:12. Treat any cay specification as a
    short-sample robustness check, never as the headline result.

    date_range=(start_ym, end_ym) restricts the sample -- controls included
    -- to a common window (e.g. COMMON_WINDOW) before standardizing
    sentiment, so different indices are compared on an identical sample."""
    facs = load_factors()
    sent = _LOADERS[sentiment_kind]()
    panel = facs.join(sent, how="inner").sort_index()
    control_cols = []

    if with_controls:
        if controls == "paper":
            panel = panel.join(load_goyal_controls(), how="inner")
            control_cols = ["real_rate", "term_premium", "default_premium", "inflation"]
        elif controls in ("full", "pca"):
            panel = panel.join(load_goyal_full(), how="inner")
            # "full" regresses on the 12 independent predictors (de/tms are exact
            # combinations of others); "pca" keeps all 14 since PCA is untroubled
            # by collinearity and the components are built from the full set.
            control_cols = list(GOYAL_FULL_COLS) if controls == "pca" else list(GOYAL_INDEP_COLS)
        else:
            raise ValueError(f"controls must be 'paper', 'full' or 'pca'; got {controls!r}")

    if add_macro:
        panel = panel.join(load_macro(), how="inner")
        control_cols = control_cols + list(MACRO_COLS)

    if add_cay:
        panel = panel.join(load_cay(), how="inner")
        control_cols = control_cols + ["cay"]

    panel = _apply_date_range(panel, date_range)
    panel["sentiment_z"] = (panel["sentiment"] - panel["sentiment"].mean()) / panel["sentiment"].std()
    panel["mkt_rf_fwd"] = panel["Mkt-RF"].shift(-horizon)

    cols = ["sentiment_z"] + control_cols
    reg_df = panel.dropna(subset=["mkt_rf_fwd"] + cols).copy()

    if with_controls and controls == "pca":
        # collapse the 14 collinear predictors into n_pc orthogonal components.
        # standardize first, else the annualised-rate columns dominate purely
        # through their units (see the units note in load_goyal_full).
        macro = reg_df[list(GOYAL_FULL_COLS)]
        z = (macro - macro.mean()) / macro.std()
        pcs = PCA(n_components=n_pc).fit_transform(z.values)
        pc_cols = [f"macro_pc{i+1}" for i in range(n_pc)]
        for i, c in enumerate(pc_cols):
            reg_df[c] = pcs[:, i]
        extra = (list(MACRO_COLS) if add_macro else []) + (["cay"] if add_cay else [])
        cols = ["sentiment_z"] + pc_cols + extra

    X = sm.add_constant(reg_df[cols])
    y = reg_df["mkt_rf_fwd"]
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    return fit, reg_df


def run_table13(sentiment_kind="pls", threshold="1sd", portfolio_kind="25", date_range=None,
                controls="paper", add_cay=False, add_macro=False):
    """Table 13 analogue: PLS-sentiment robustness check.
    Panel A: predictive regression of next-month market excess return on
    sentiment + macro controls. Panel B: conditional CAPM scaled by
    sentiment (identical spec to Eq.9/Table 3 -- just reuses run()).
    controls=/add_cay= are passed through to predictive_regression()."""
    print(f"\n{'='*78}\nTABLE 13 analogue  --  sentiment = {sentiment_kind.upper()}\n{'='*78}")
    print(f"\n-- Panel A analogue: predictive regression (controls={controls}, "
          f"cay={add_cay}, macro={add_macro}) --")
    fit, reg_df = predictive_regression(sentiment_kind, with_controls=True, date_range=date_range,
                                        controls=controls, add_cay=add_cay, add_macro=add_macro)
    coef = fit.params["sentiment_z"]
    tstat = fit.tvalues["sentiment_z"]
    print(f"  beta_sentiment = {coef:.4f}   t (HAC) = {tstat:.3f}   (T = {len(reg_df)})")
    print(fit.summary().tables[1])

    print("\n-- Panel B analogue: conditional CAPM scaled by sentiment (Eq.9/Table 3 spec) --")
    panelB = run(sentiment_kind, threshold=threshold, portfolio_kind=portfolio_kind, date_range=date_range)

    return {"panelA_fit": fit, "panelB": panelB}


def run_table2(date_range=None, controls="paper", add_cay=False, add_macro=False, n_pc=3):
    """Table 2 analogue: predictive regression of next-month market excess
    return on lagged sentiment, Panel A (univariate, Eq.11) and Panel B
    (with controls), for the paper's four indices.

    controls= / add_cay= / n_pc= pass through to predictive_regression().
    Defaults reproduce the paper; controls="full" or "pca" run the extended
    macro control sets, and add_cay=True adds the consumption-wealth ratio
    at the cost of a much shorter sample. See that function's docstring."""
    print(f"\n{'='*78}\nTABLE 2 analogue -- sentiment predicts next-month market excess return\n{'='*78}")
    rows_a, rows_b = {}, {}
    for kind in ["bw", "mcsi", "cbcci", "as"]:
        fit_a, df_a = predictive_regression(kind, with_controls=False, date_range=date_range,
                                            add_cay=False)
        rows_a[kind.upper()] = {
            "beta": fit_a.params["sentiment_z"], "t": fit_a.tvalues["sentiment_z"],
            "R2_pct": 100 * fit_a.rsquared, "T": len(df_a),
        }
        fit_b, df_b = predictive_regression(kind, with_controls=True, date_range=date_range,
                                            controls=controls, n_pc=n_pc,
                                            add_cay=add_cay, add_macro=add_macro)
        rows_b[kind.upper()] = {
            "beta": fit_b.params["sentiment_z"], "t": fit_b.tvalues["sentiment_z"],
            "R2_pct": 100 * fit_b.rsquared, "T": len(df_b),
        }
    panelA = pd.DataFrame(rows_a).T
    panelB = pd.DataFrame(rows_b).T
    print("\n-- Panel A: univariate (Eq.11), HAC/Newey-West(3) SEs --")
    print(panelA.round(4))
    label = {"paper": "real rate, term premium[approx], default premium, inflation",
             "full": f"{len(GOYAL_INDEP_COLS)} independent Welch-Goyal predictors",
             "pca": f"first {n_pc} PCs of the {len(GOYAL_FULL_COLS)} Welch-Goyal predictors"}[controls]
    if add_macro:
        label += " + macro (unemployment, housing, CAPE yield)"
    if add_cay:
        label += " + cay (SHORT SAMPLE)"
    print(f"\n-- Panel B: with controls ({label}) --")
    print(panelB.round(4))
    return {"panelA": panelA, "panelB": panelB}


# --------------------------------------------- 5 individual sentiment-scaled CAPMs --
def run_all_sentiments(portfolio_kind="industry48", threshold="1sd", date_range=COMMON_WINDOW):
    """Convenience driver: runs the sentiment-scaled CAPM (Eq.9 spec, via
    run()) individually for the 5 sentiment indices -- MCSI, CBCCI, PMI, BW,
    CFNAI -- all on the SAME test-asset set and the SAME common date window
    (default COMMON_WINDOW = Dec 1969-Dec 2025), so the 5 results are
    directly comparable. Deliberately excludes the composite AS index this
    round (it's built FROM BW/MCSI/CBCCI, so including it alongside those
    three would be double-counting the same information)."""
    kinds = ["mcsi", "cbcci", "pmi", "bw", "cfnai"]
    results = {}
    rows = []
    for kind in kinds:
        res = run(kind, threshold=threshold, portfolio_kind=portfolio_kind, date_range=date_range)
        results[kind] = res
        lam = res["lambda_table"]
        rows.append({
            "sentiment": kind.upper(),
            "lam_s_lag": lam.loc["s_lag", "lambda"], "t_s_lag": lam.loc["s_lag", "t_Shanken"],
            "lam_mkt_rf": lam.loc["mkt_rf", "lambda"], "t_mkt_rf": lam.loc["mkt_rf", "t_Shanken"],
            "lam_s_mkt": lam.loc["s_mkt", "lambda"], "t_s_mkt": lam.loc["s_mkt", "t_Shanken"],
            "R2": res["r2"], "R2_gls": res["r2_gls"],
            "chi2_p": res["chi2"][2],
        })
    summary = pd.DataFrame(rows).set_index("sentiment")
    print(f"\n{'='*78}\nSUMMARY -- 5 sentiment-scaled CAPMs, common window, {portfolio_kind} portfolios\n{'='*78}")
    print(summary.round(4))
    return {"results": results, "summary": summary}


if __name__ == "__main__":
    import sys
    kind = sys.argv[1] if len(sys.argv) > 1 else "mcsi"
    port_kind = sys.argv[2] if len(sys.argv) > 2 else "25"
    run(kind, portfolio_kind=port_kind)

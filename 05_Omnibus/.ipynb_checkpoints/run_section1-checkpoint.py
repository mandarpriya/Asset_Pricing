"""
run_section1.py -- Section 0 + the six preliminary tests (1.01-1.06) only.

Section 0 is the data processing every omnibus method shares: first-pass
betas, second-pass lambdas, pricing errors, four R-squareds.
Methods 1.01-1.06 are the PRELIMINARY tests, run BEFORE any price-of-risk
test is worth reading:

    1.01 / 1.02 / 1.03   H0: all betas on factor k are zero
    1.04 / 1.05 / 1.06   H0: all betas on factor k are equal to each other

    .01 / .04  assume iid residuals
    .02 / .05  Newey-West covariance
    .03 / .06  VARHAC covariance

Each returns one chi-square statistic and p-value PER FACTOR.

TRAP -- Section 0 is NOT the same for every method.  Line 172 of Omnibus.py
branches on the INTEGER part of the method number:

    floor(method) in [1,2,4]  ->  cross-sectional regression WITHOUT intercept
    floor(method) in [3,5]    ->  cross-sectional regression WITH intercept

So running a 1.xx method gives you the no-intercept second pass.  `const` and
`R2i` come back as NaN by construction (they do not exist in that branch), and
`R2` is computed from NO-INTERCEPT lambdas.  On the BW / 25 panel:

    method 1.01   Lambda [ 0.7989  0.6160 -2.5069]   R2 0.8167   R2i  nan
    method 3.03   Lambda [ 0.7551  0.5353 -2.4930]   R2 0.5862   R2i  0.8177

Same data, same key names, different numbers.  The betas are identical across
all methods -- only the second pass changes.  Do not compare an `R2` from a
1.xx/2.xx run against an `R2` from a 3.xx/5.xx run.

Why they come first: a cross-sectional regression of mean returns on betas
is only meaningful if the betas (a) differ from zero and (b) differ from
each other. If 1.04-1.06 fail to reject, the betas are a flat line across
test assets, and lambda in the second pass is fitting noise -- whatever
standard error the later methods attach to it.
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "04_FamaMacBeth_Python"))
sys.path.insert(0, str(ROOT / "05_Omnibus"))

import sentiment_capm as sc
from omnibus_section1 import omnibus_section1

LABELS = {
    1.01: ("betas all zero",  "iid"),
    1.02: ("betas all zero",  "Newey-West"),
    1.03: ("betas all zero",  "VARHAC"),
    1.04: ("betas all equal", "iid"),
    1.05: ("betas all equal", "Newey-West"),
    1.06: ("betas all equal", "VARHAC"),
}


def run(sentiment_kind="bw", portfolio_kind="25",
        factor_cols=("s_lag", "mkt_rf", "s_mkt"), date_range=sc.COMMON_WINDOW):

    panel, ports = sc.build_panel(sentiment_kind, portfolio_kind, date_range)
    R = panel[ports].values
    f = panel[list(factor_cols)].values
    T, K = f.shape
    N = R.shape[1]

    print(f"{sentiment_kind.upper()} / {portfolio_kind} portfolios   "
          f"T={T}  N={N}  K={K}   {panel.index[0]}-{panel.index[-1]}")

    ans = omnibus_section1(R, f, 1.01)          # Section 0 comes free
    print("\nSECTION 0 -- shared by every method")
    print(f"  beta spread (max-min) : "
          + "  ".join(f"{c}={v:.3f}" for c, v in
                      zip(factor_cols, ans['beta'].max(0) - ans['beta'].min(0))))
    print(f"  lambda OLS            : "
          + "  ".join(f"{c}={v:+.4f}" for c, v in zip(factor_cols, ans['Lambda'])))
    print("  (second pass WITHOUT intercept -- method 1.xx; see TRAP in docstring)")
    print(f"  R2 / R2_gls           : {float(ans['R2']):.4f} / {float(ans['R2_gls']):.4f}")
    print("  const, R2i, R2i_gls   : n/a in this branch -- use a 3.xx/5.xx method")

    print(f"\nSECTION 1 -- preliminary tests   (chi2, p-value) per factor")
    print(f"  {'method':7s} {'H0':16s} {'cov':11s} " +
          "".join(f"{c:>22s}" for c in factor_cols))
    for m in (1.01, 1.02, 1.03, 1.04, 1.05, 1.06):
        a = omnibus_section1(R, f, m)
        cells = "".join(f"{t:12.1f} ({p:6.4f})"
                        for t, p in zip(np.ravel(a['Test']), np.ravel(a['pval'])))
        h0, cov = LABELS[m]
        print(f"  {m:<7.2f} {h0:16s} {cov:11s} " + cells)


if __name__ == "__main__":
    run(*sys.argv[1:])

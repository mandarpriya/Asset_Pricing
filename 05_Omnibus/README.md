# 05_Omnibus — third-party code (not written by this project)

The Python modules in this folder are **not mine**. They are redistributed here,
unmodified, so that the replication in `04_FamaMacBeth_Python/` is reproducible
without a separate download.

## Source and citation

> Kroencke, Tim A. and Thimme, Julian (2021),
> *"A Skeptical Appraisal of Robust Asset Pricing Tests"*, Working Paper.

The authors' own header states:

> "Please cite our paper whenever using this code or the omnibus.m function
> for your own research. Thank you!"

**If you use anything in this folder, cite that paper.** Any results in this
repository produced via `omnibus_tests()` or `intercept_restriction()` in
`04_FamaMacBeth_Python/sentiment_capm.py` depend on it.

The authors note that large parts of their code are in turn adapted from the
supplementary materials of:

- Burnside, A. Craig (2007), "The forward premium is still a puzzle," NBER WP 13129.
- Bryzgalova, Svetlana, Jiantao Huang and Christian Julliard (2020), "Bayesian
  solutions for the factor zoo: We just ran two quadrillion models."
- Giglio, Stefano and Dacheng Xiu (2021), "Asset pricing with omitted factors,"
  *Journal of Political Economy*.
- Gospodinov, Nikolay, Raymond Kan and Cesare Robotti (2014),
  "Misspecification-robust inference in linear asset-pricing models with
  irrelevant risk factors," *Review of Financial Studies* 27(7), 2139–2170.
- Kan, Raymond, Cesare Robotti and Jay Shanken (2013), "Pricing model
  performance and the two-pass cross-sectional regression methodology,"
  *Journal of Finance* 68(6), 2617–2649.
- Kleibergen, Frank, Lingwei Kong and Zhaoguo Zhan, "Identification robust
  inference..."

## What is here, and what is not

Only the modules `omnibus()` needs at import time:

| file | role |
|---|---|
| `Omnibus.py` | the dispatcher — ~70 cross-sectional tests, selected by a numeric `method` argument |
| `nw.py` | Newey-West HAC covariance |
| `hac_var.py` | VARHAC covariance |
| `linchi2.py` | linear chi-square routines |
| `cdfchic.py` | chi-square CDF helper |
| `block_bootstrap.py` | block bootstrap |
| `FMB_coefficients.py` | Fama-MacBeth coefficients |

**Deliberately not copied** from the original distribution: the replication
data (`Date_3Factors_25Returns_*.csv`, `RX_HKS_*.csv`, ~5 MB), the figure and
table scripts (`replication.py`, `Boot_*.py`, `MC_*.py`, `Table_horse_race.py`),
`Manipulate_returns.py`, and `results/`. Those are not needed to call
`omnibus()` and would bloat this repository. Get them from the authors if you
want to reproduce their own paper.

## How this project uses it

`sentiment_capm.py` puts this folder on `sys.path` and imports `Omnibus`
(its helpers are imported as top-level modules, so importing `Omnibus.py`
by file path alone does not work). Methods used:

| method | what it gives | why it matters here |
|---|---|---|
| 3.03 | Shanken standard errors | cross-check against `run()`'s own implementation |
| 3.07 | Kan/Robotti/Shanken **misspecification-robust** SEs | ordinary FM and Shanken SEs assume the model is *correctly specified*; these do not |
| 5.14 | KRS standard error of the cross-sectional R² | every other R² in this project is reported without one |

## Two findings worth recording

**It independently validates this project's code.** For BW on the 25 size-BM
portfolios over 1970:01–2025:12, `omnibus()`'s `R2i` (0.8177) and `R2i_gls`
(0.5643) match `run()` to four decimals, as do all three factor lambdas. That
is confirmation from a separate implementation — and in particular it confirms
the GLS R² correction described in `sentiment_capm.py`'s `gls_r2()` docstring.

**Key-name trap.** `omnibus()` returns both `R2` (no intercept) and `R2i`
(with intercept), and they differ substantially — 0.5862 versus 0.8177 in the
run above. Method 5.14's standard error belongs to `R2i`. Pairing it with `R2`
is an easy and silent mistake.

## Known issues

Method 5.13 (test of H₀: R² = 0) raises `RuntimeError: Unable to parse
arguments` on this data. This is inside the authors' own code, not a problem
with how it is called here — their file header flags known numerical issues in
several methods (2.09/3.09 and 6.03/6.04 are noted explicitly). Method 5.14
still returns the standard error, which is the part this project relies on.

`omnibus()` also hardcodes its options internally: `lags = 3`, `excess_ret = 0`,
`traded_f = 1`. The `traded_f` setting only selects a default null value for
`Lambda0`, and since this project leaves `Lambda0` at its default of `0`, the
setting never binds — every test run here is against H₀: parameter = 0.

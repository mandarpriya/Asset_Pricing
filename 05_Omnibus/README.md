# 05_Omnibus — third-party code (not written by this project)

The Python modules in this folder are **not mine**. They are redistributed here
so that the replication in `04_FamaMacBeth_Python/` is reproducible without a
separate download.

They are redistributed **with four small fixes**, each recorded in
[Local fixes](#local-fixes) below and marked `# BUGFIX:` in the source. The
unmodified originals sit beside them as `*_ORIGINAL.py`. Every fix is a type or
shape correction required by current NumPy/SciPy, or a plainly dropped
assignment; none changes any formula, and each was verified to leave the
numerical output bit-identical where the original still ran.

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

`omnibus()` hardcodes its options internally: `lags = 3`, `excess_ret = 0`,
`traded_f = 1`. The `traded_f` setting only selects a default null value for
`Lambda0`, and since this project leaves `Lambda0` at its default of `0`, the
setting never binds — every test run here is against H₀: parameter = 0.

`Section 0` is **not** the same for every method. Line 172 branches on the
integer part of the method number: `floor(method)` in `[1,2,4]` runs the
second-pass cross-sectional regression **without** an intercept, `[3,5]` **with**
one. So `const`, `R2i` and `R2i_gls` come back `NaN` from a 1.xx/2.xx/4.xx call,
and `R2` from a 2.xx call is not comparable with `R2` from a 3.xx call. On the
FF3 / 25 size-BM panel, 1964:01–2026:07:

| method | `const` | `R2` | `R2i` |
|---|---|---|---|
| 2.01 (no intercept) | `NaN` | **0.4812** | `NaN` |
| 3.03 (with intercept) | 0.0116 | **−42.6979** | **0.6631** |

Same data, same betas — only the second pass differs. The −42.70 is not a bug:
in the 3.xx branch `R2` is built from *with-intercept* lambdas but then drops the
constant from the pricing errors, so it measures a model nothing constrained it
to fit. `R2i` (0.6631) is the meaningful figure there, and `R2` (0.4812) is the
meaningful one in the 2.xx branch. Quoting the wrong one of the four is the
easiest silent mistake to make with this code.

## Local fixes

All four are in the authors' code, not in how this project calls it. They are
latent in the published version and surface only on a current Python stack, or
with more than one factor. Kroencke and Thimme's own paper runs `K = 1` (a
tangency portfolio) on 2019 data with 2022-era libraries, which is very likely
why none of them was ever hit.

| file | what was wrong | fix |
|---|---|---|
| `hac_var.py` (~l.86) | `bic = float(np.log(vv.T @ vv) + ...)`. `vv` is `(T,1)`, so the argument is a `(1,1)` array. Converting an `ndim > 0` array to a Python scalar was deprecated in NumPy 1.25 and is a `TypeError` from NumPy 2.3. Breaks methods **1.03** and **1.06** (the VARHAC ones). | wrap in `np.squeeze` |
| `Omnibus.py` (6.01, 6.02) | `if f.shape[1] == 1: Sf = ... else: np.cov(f.T, ddof=0)` — the `Sf =` is missing from the `else`, so the value is computed and discarded and the next line raises `UnboundLocalError`. Breaks **6.01** and **6.02** for any `K > 1`. The authors' own 3.02/3.04 have the same if/else *with* the assignment. | `Sf = np.cov(f.T, ddof = 0)` |
| `Omnibus.py` (3.06) | `Gammahat[0, phat] = Gammatilde[0]`. `Gammatilde` is `(K+1,1)`, so `Gammatilde[0]` has shape `(1,)`; NumPy ≥ 2.3 refuses to assign it into a scalar slot. Breaks **3.06** (Giglio/Xiu). | `Gammatilde[0, 0]` |
| `linchi2.py` (~l.91) | `mar = np.array(10 ** (-9), ndmin=2)` — a `(1,1)` array, a MATLAB-translation artifact. It is used four lines later as a bound for `spo.brentq`, which requires scalar bounds; current SciPy reports this as `RuntimeError: Unable to parse arguments`. `d0` arriving as a column causes the same problem via `d[0]`. Breaks **5.11**, **5.12**, **5.13**. | `mar = 1e-9`, plus coercing `c` to a float and `d0` to 1-D at the top of `linchi2_func` |

After these, **all 57 methods in sections 2–6 run**, plus 1.01–1.06.

### Library compatibility

`legacy_compat.py` (mine, not the authors') restores names that NumPy 2.0 and
pandas 2.0 removed rather than deprecated: `np.NAN`, `np.NaN`, `np.float`,
`np.int`, `np.alltrue`, and `DataFrame.set_axis(..., inplace=True)`. It patches
at runtime and is a no-op on older versions, so the authors' files stay as they
are on disk. Import it before calling anything here.

### Line endings

The authors' files use CRLF. The patched files keep CRLF, so
`diff *_ORIGINAL.py *.py` shows only the real changes.

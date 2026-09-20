# =============================================================================
# omnibus_sections_1_3.R
#
# Sections 0-3 of Kroencke & Thimme's omnibus() -- rewritten in R, step by step.
#
# The point of this file is that nothing is hidden. Every matrix is named, every
# formula is on one line, and you can stop after any block and look at what you
# have. It reproduces the Python output to ~1e-12.
#
# COVERED
#   Section 0   first-pass betas, second-pass lambdas, pricing errors, R^2s
#   Section 1   1.01 1.02 1.04 1.05   preliminary tests on the betas
#   Section 2   2.01 2.02 2.03        lambda, cross-section WITHOUT intercept
#   Section 3   3.01 3.02 3.03        lambda, cross-section WITH intercept
#
# NOT COVERED (deliberately)
#   *.03/*.06 VARHAC   -- a VAR-based HAC estimator; ~80 lines on its own
#   2.04/2.05 3.04/3.05 GMM standard errors
#   3.06 Giglio/Xiu, 3.07 Kan/Robotti/Shanken, Sections 4-6
#
# THE ONE THING THAT TRIPS PEOPLE UP
#   Sections 2 and 3 are the SAME test. Section 2 runs the second-pass
#   regression without an intercept, Section 3 with one. They give different
#   lambdas, different R^2s and often different conclusions. Section 2's `R2`
#   and Section 3's `R2i` are the two numbers worth quoting; Section 3's `R2`
#   is a constrained fit of an unconstrained estimate and can be wildly
#   negative. See the README in 05_Omnibus.
#
# Source: Kroencke, T. A. and Thimme, J. (2021), "A Skeptical Appraisal of
# Robust Asset Pricing Tests". Cite that paper if you use this.
# =============================================================================


# ---------------------------------------------------------------- helpers ----

#' Newey-West HAC covariance. Port of the authors' nw.py for fixed `lag`.
#' h is T x r. Returns r x r.
nw_cov <- function(h, lag = 3) {
  T <- nrow(h)
  h <- sweep(h, 2, colMeans(h))              # demean, as nw.py does
  V <- crossprod(h) / T
  if (lag > 0) for (i in seq_len(lag)) {
    V1 <- crossprod(h[(i + 1):T, , drop = FALSE], h[1:(T - i), , drop = FALSE]) / T
    V  <- V + (1 - i / (lag + 1)) * (V1 + t(V1))
  }
  V
}

#' Upper tail of chi-square: the authors' cdfchic.py.
chi2_sf <- function(x, df) pchisq(x, df, lower.tail = FALSE)


# --------------------------------------------------------------- SECTION 0 ---

#' Everything every method shares.
#'
#' @param Rmat  T x N matrix of test-asset EXCESS returns
#' @param fmat  T x K matrix of factors
#' @param intercept  FALSE = second pass without intercept (methods 1/2/4)
#'                   TRUE  = second pass with intercept    (methods 3/5)
omnibus_sec0 <- function(Rmat, fmat, intercept = FALSE) {

  T <- nrow(fmat); K <- ncol(fmat); N <- ncol(Rmat)
  stopifnot(nrow(Rmat) == T)

  ## --- FIRST PASS: time-series regression of every asset on [1, f] ----------
  ## One solve, not a loop: all N assets share the same regressor matrix.
  Fm   <- cbind(1, fmat)                        # T x (K+1)
  FFi  <- solve(crossprod(Fm))                  # (F'F)^-1
  B    <- FFi %*% crossprod(Fm, Rmat)           # (K+1) x N   row 1 = alphas
  alpha <- B[1, ]
  beta  <- t(B[-1, , drop = FALSE])             # N x K
  e     <- Rmat - Fm %*% B                      # T x N residuals
  S     <- crossprod(e) / T                     # N x N, residual cov (iid)

  ## After this point the time dimension is gone. Everything below is a
  ## regression with N observations.
  Rbar  <- colMeans(Rmat)
  Sigma <- cov(Rmat)                            # N x N (ddof = 1, as numpy)

  ## --- SECOND PASS: cross-sectional regression of mean returns on betas -----
  if (!intercept) {
    A      <- solve(crossprod(beta), t(beta))              # K x N
    Lambda <- as.vector(A %*% Rbar)
    Theta  <- Lambda
    const  <- NA_real_
    Sb     <- solve(Sigma, beta)
    Lgls   <- as.vector(solve(crossprod(Sb, beta), crossprod(Sb, Rbar)))
    cgls   <- NA_real_
  } else {
    X      <- cbind(1, beta)                               # N x (K+1)
    A      <- solve(crossprod(X), t(X))                    # (K+1) x N
    Theta  <- as.vector(A %*% Rbar)
    Lambda <- Theta[-1]
    const  <- Theta[1]
    Sx     <- solve(Sigma, X)
    Tg     <- as.vector(solve(crossprod(Sx, X), crossprod(Sx, Rbar)))
    Lgls   <- Tg[-1]
    cgls   <- Tg[1]
  }

  ## --- pricing errors and the four R-squareds -------------------------------
  PE  <- as.vector(Rbar - beta %*% Lambda)      # intercept NOT removed
  PEi <- if (intercept) PE - const else rep(NA_real_, N)
  den <- sum((Rbar - mean(Rbar))^2)             # cross-sectional variance
  R2  <- 1 - sum(PE^2) / den
  R2i <- if (intercept) 1 - sum(PEi^2) / den else NA_real_

  Si    <- solve(Sigma)
  mu    <- as.numeric(crossprod(rep(1, N), Si %*% Rbar) /
                      crossprod(rep(1, N), Si %*% rep(1, N)))
  dev   <- Rbar - mu
  dgls  <- as.numeric(crossprod(dev, Si %*% dev))
  PEg   <- as.vector(Rbar - beta %*% Lgls)
  PEig  <- if (intercept) PEg - cgls else rep(NA_real_, N)
  R2g   <- 1 - as.numeric(crossprod(PEg,  Si %*% PEg))  / dgls
  R2ig  <- if (intercept) 1 - as.numeric(crossprod(PEig, Si %*% PEig)) / dgls else NA_real_

  list(T = T, K = K, N = N, F = Fm, FFi = FFi, B = B,
       alpha = alpha, beta = beta, e = e, S = S,
       Rbar = Rbar, Sigma = Sigma, A = A,
       Lambda = Lambda, Theta = Theta, const = const,
       Lambda_gls = Lgls, const_gls = cgls,
       PE = PE, PEi = PEi, R2 = R2, R2i = R2i, R2_gls = R2g, R2i_gls = R2ig)
}


# --------------------------------------------------------------- SECTION 1 ---

#' Preliminary tests: do the betas vary enough for a second pass to mean
#' anything? One chi-square per factor.
#'
#' 1.01 / 1.02  H0: all N betas on factor k are ZERO        (df = N)
#' 1.04 / 1.05  H0: all N betas on factor k are EQUAL       (df = N-1)
#' .01/.04 assume iid residuals; .02/.05 use Newey-West.
omnibus_sec1 <- function(Rmat, fmat, lag = 3, fnames = NULL) {

  s <- omnibus_sec0(Rmat, fmat, intercept = FALSE)
  N <- s$N; K <- s$K; T <- s$T
  if (is.null(fnames)) fnames <- colnames(fmat)
  if (is.null(fnames)) fnames <- paste0("f", seq_len(K))

  ## B is (K+1) x N and R is column-major, so as.vector(B) already stacks
  ## asset by asset: (alpha_1, b_11..b_K1, alpha_2, ...). That is exactly the
  ## ordering the covariance matrices below assume.
  b_vec <- as.vector(s$B)

  ## iid: AV = S (x) (F'F)^-1
  AV_iid <- kronecker(s$S, s$FFi)

  ## Newey-West: stack the score e_i * F asset by asset, HAC it, sandwich it
  g      <- do.call(cbind, lapply(seq_len(N), function(i) s$e[, i] * s$F))
  AV_nw  <- kronecker(diag(N), s$FFi) %*% nw_cov(g, lag) %*%
            kronecker(diag(N), s$FFi) * T

  ## RR turns "all betas equal" into N-1 contrasts b_1 - b_j
  RR <- cbind(rep(1, N - 1), -diag(N - 1))

  out <- list()
  for (nm in c("1.01", "1.02", "1.04", "1.05")) {
    AV    <- if (nm %in% c("1.01", "1.04")) AV_iid else AV_nw
    equal <- nm %in% c("1.04", "1.05")
    stat  <- pv <- numeric(K)
    for (k in seq_len(K)) {
      idx <- seq(k + 1, N * (K + 1), by = K + 1)   # the k-th beta of each asset
      b   <- b_vec[idx]
      V   <- AV[idx, idx]
      if (equal) { b <- RR %*% b; V <- RR %*% V %*% t(RR) }
      stat[k] <- as.numeric(crossprod(b, solve(V, b)))
      pv[k]   <- chi2_sf(stat[k], if (equal) N - 1 else N)
    }
    out[[nm]] <- data.frame(method = nm,
                            H0  = if (equal) "betas all equal" else "betas all zero",
                            cov = if (nm %in% c("1.01","1.04")) "iid" else "Newey-West",
                            factor = fnames, chi2 = stat, pval = pv,
                            row.names = NULL)
  }
  do.call(rbind, out)
}


# ------------------------------------------------------------ SECTIONS 2/3 ---

#' Price-of-risk tests. `intercept = FALSE` gives Section 2 (methods 2.01-2.03),
#' `intercept = TRUE` gives Section 3 (methods 3.01-3.03).
#'
#'  x.01  Fama-MacBeth: sd of the period-by-period lambdas
#'  x.02  OLS/analytic, NO Shanken correction
#'  x.03  Shanken (1992) errors-in-variables correction
#'
#' Shanken multiplies the sampling error by (1 + lambda' Sigma_f^-1 lambda).
#' It NEVER changes the coefficients, only the standard errors.
omnibus_sec23 <- function(Rmat, fmat, intercept = FALSE, fnames = NULL) {

  s <- omnibus_sec0(Rmat, fmat, intercept = intercept)
  T <- s$T; K <- s$K
  est  <- if (intercept) s$Theta else s$Lambda
  labs <- if (is.null(fnames)) colnames(fmat) else fnames
  if (is.null(labs)) labs <- paste0("f", seq_len(K))
  if (intercept) labs <- c("(const)", labs)

  Sf_small <- cov(fmat) * (T - 1) / T                 # numpy ddof = 0
  Sf <- if (intercept) {
    M <- matrix(0, K + 1, K + 1); M[-1, -1] <- Sf_small; M
  } else Sf_small

  ASA    <- s$A %*% s$S %*% t(s$A)
  shank  <- 1 + as.numeric(crossprod(s$Lambda, solve(Sf_small, s$Lambda)))

  ## x.01 -- Fama-MacBeth: refit the cross-section every period, take the sd
  lam_t  <- Rmat %*% t(s$A)                           # T x (K or K+1)
  se_fm  <- sqrt(colMeans(sweep(lam_t, 2, colMeans(lam_t))^2) / T)

  se_ols <- sqrt(diag((ASA + Sf) / T))
  se_sh  <- sqrt(diag((ASA * shank + Sf) / T))

  res <- lapply(list(c("01", "Fama-MacBeth"), c("02", "OLS, no Shanken"),
                     c("03", "Shanken-corrected")), function(m) {
    se <- switch(m[1], "01" = se_fm, "02" = se_ols, "03" = se_sh)
    tv <- est / se
    data.frame(method = paste0(if (intercept) "3." else "2.", m[1]),
               se_type = m[2], param = labs, estimate = est, se = se,
               t = tv, pval = 1 - pt(abs(tv), T - K), row.names = NULL)
  })
  do.call(rbind, res)
}


# ------------------------------------------------------------------ report ---

omnibus_report <- function(Rmat, fmat, label = "", fnames = NULL, lag = 3) {
  s0 <- omnibus_sec0(Rmat, fmat, intercept = FALSE)
  s3 <- omnibus_sec0(Rmat, fmat, intercept = TRUE)
  cat(sprintf("\n%s\nT = %d   N = %d   K = %d\n", label, s0$T, s0$N, s0$K))

  cat("\n-- SECTION 0 --------------------------------------------------\n")
  cat(sprintf("  beta spread (max-min) : %s\n",
              paste(sprintf("%s=%.3f", if (is.null(fnames)) colnames(fmat) else fnames,
                            apply(s0$beta, 2, function(z) diff(range(z)))), collapse = "  ")))
  cat(sprintf("  no intercept : lambda %s | R2 %8.4f  R2_gls %8.4f\n",
              paste(sprintf("%+.4f", s0$Lambda), collapse = " "), s0$R2, s0$R2_gls))
  cat(sprintf("  with interc. : const %+.4f  lambda %s | R2i %7.4f  R2i_gls %7.4f\n",
              s3$const, paste(sprintf("%+.4f", s3$Lambda), collapse = " "),
              s3$R2i, s3$R2i_gls))

  cat("\n-- SECTION 1: are the betas usable? ---------------------------\n")
  print(omnibus_sec1(Rmat, fmat, lag, fnames), digits = 4, row.names = FALSE)

  cat("\n-- SECTION 2: lambda, NO intercept ----------------------------\n")
  print(omnibus_sec23(Rmat, fmat, FALSE, fnames), digits = 4, row.names = FALSE)

  cat("\n-- SECTION 3: lambda, WITH intercept --------------------------\n")
  print(omnibus_sec23(Rmat, fmat, TRUE, fnames), digits = 4, row.names = FALSE)
  invisible(list(sec0_noint = s0, sec0_int = s3))
}

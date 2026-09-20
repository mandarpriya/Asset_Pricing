# Runs the R implementation on both panels and checks it against the Python
# omnibus() output. Run from 06_R/:   Rscript run_demo.R
source(file = "Documents/GitHub/Asset_Pricing/06_R/omnibus_sections_1_3.R")

## ---- panel 1: FF3 / 25 size-BM (the authors' own test assets) -------------
d  <- as.matrix(read.csv("ff3_25_monthly.csv", header = FALSE))
Rm <- d[, 5:29]; f <- d[, 2:4]; colnames(f) <- c("Mkt-RF", "SMB", "HML")
omnibus_report(Rm, f, "FF3 / 25 size-BM, 1964:01-2026:07")

## ---- panel 2: the sentiment-scaled CAPM ----------------------------------
p   <- read.csv("panel_bw25.csv", check.names = FALSE)
f2  <- as.matrix(p[, c("s_lag", "mkt_rf", "s_mkt")])
Rm2 <- as.matrix(p[, setdiff(names(p), c("date", "s_lag", "mkt_rf", "s_mkt"))])
omnibus_report(Rm2, f2, "BW sentiment-scaled CAPM / 25 size-BM, 1970:01-2025:12")

## ---- validation against Python -------------------------------------------
## Reference values produced by Omnibus.omnibus() on the same two panels.
cat("\n== validation against the Python omnibus() ==\n")
chk <- function(what, got, want, tol = 5e-4) {
  ok <- all(abs(got - want) < tol)
  cat(sprintf("  %-34s %s\n", what, if (ok) "match" else
    paste("MISMATCH  got", paste(round(got, 5), collapse = " "),
          "want", paste(want, collapse = " "))))
  ok
}
s0 <- omnibus_sec0(Rm, f, FALSE); s3 <- omnibus_sec0(Rm, f, TRUE)
a  <- omnibus_sec1(Rm, f); b2 <- omnibus_sec23(Rm, f, FALSE); b3 <- omnibus_sec23(Rm, f, TRUE)
res <- c(
  chk("FF3  R2 (no intercept)",   s0$R2,      0.4812),
  chk("FF3  R2_gls",              s0$R2_gls, -0.1917),
  chk("FF3  R2i (with intercept)", s3$R2i,    0.6631),
  chk("FF3  1.01 chi2",  a$chi2[a$method=="1.01"], c(77961.0, 81239.8, 42458.2), tol = 0.5),
  chk("FF3  1.04 chi2",  a$chi2[a$method=="1.04"], c(  339.4, 79181.3, 41596.1), tol = 0.5),
  chk("FF3  2.01 t",     b2$t[b2$method=="2.01"],  c(3.365, 1.338, 3.181)),
  chk("FF3  2.03 t",     b2$t[b2$method=="2.03"],  c(3.363, 1.337, 3.178)),
  chk("FF3  3.01 t",     b3$t[b3$method=="3.01"],  c(4.446, -1.796, 1.100, 2.945)),
  chk("FF3  3.03 t",     b3$t[b3$method=="3.03"],  c(4.380, -1.777, 1.099, 2.943)),
  chk("BW   lambda (no intercept)",
      omnibus_sec0(Rm2, f2, FALSE)$Lambda, c(0.7989, 0.6160, -2.5069)),
  chk("BW   const + lambda (with int.)",
      omnibus_sec0(Rm2, f2, TRUE)$Theta,   c(0.08556, 0.75512, 0.53526, -2.49300))
)
cat(sprintf("\n  %d of %d checks match the Python output\n", sum(res), length(res)))

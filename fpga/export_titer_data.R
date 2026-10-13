# Export the training sets of a BART g-formula pipeline's six titer models in the form the FPGA kernel and the host
# programs read (cutpoint ranks, centred y and weights, priors), plus a reference BART::wbart fit of each for comparison.
# Usage: Rscript export_titer_data.R <estimator script, e.g. BART_gformula_week6_mean.R> <folder with simulated_observed_week6.csv> <output folder>
# Runs the estimator script up to (not including) its titer fits, so the data preparation is exactly the pipeline's own.
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 3)
est <- readLines(args[1])
stop_at <- grep("^titer_fit <- list\\(\\)", est)[1] - 1
Sys.setenv(SIM_OUT_DIR = normalizePath(args[2]))
if (Sys.getenv("NDPOST") == "") Sys.setenv(NDPOST = "100")
if (Sys.getenv("NSKIP") == "") Sys.setenv(NSKIP = "1000")
eval(parse(text = est[seq_len(stop_at)]))
OUT <- args[3]; dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
for (w in 1:6) {
  pc <- pred_cols_titer(w); tgt <- paste0("titer_", w)
  ix <- which(stats::complete.cases(d[, c(pc, tgt)]))
  om <- 1 / S_obs[ix, w]; om <- pmin(om, quantile(om, IPCW_CAP_PCTL, na.rm = TRUE)); om <- om / mean(om); wt <- 1 / sqrt(om)
  x <- as.matrix(d[ix, pc, drop = FALSE]); y <- d[ix, tgt]
  set.seed(100 + w)
  fit <- wbart(x.train = x, y.train = y, w = wt, ndpost = ndpost, nskip = nskip, printevery = 100000)
  bm <- bartModelMatrix(x, 100L, usequants = FALSE, rm.const = TRUE)
  X <- bm$X; xi <- fit$treedraws$cutpoints; stopifnot(length(xi) == ncol(X))
  q <- sapply(seq_along(xi), function(v) findInterval(X[, v], xi[[v]]))   # #cutpoints <= x:  x < xi[c] (0-based) <=> q <= c
  yc <- y - mean(y)
  sigest <- summary(lm(yc ~ ., data.frame(X, yc)))$sigma
  lambda <- sigest^2 * qchisq(0.1, 3) / 3
  tau <- (max(yc) - min(yc)) / (2 * 2 * sqrt(200))
  pre <- file.path(OUT, sprintf("titer%d", w))
  write.table(q, paste0(pre, "_q.txt"), row.names = FALSE, col.names = FALSE)
  write.table(cbind(yc, wt), paste0(pre, "_yw.txt"), row.names = FALSE, col.names = FALSE)
  writeLines(c(sprintf("n %d", nrow(X)), sprintf("p %d", ncol(X)), sprintf("numcut %s", paste(lengths(xi), collapse = " ")),
               sprintf("fmean %.17g", mean(y)), sprintf("tau %.17g", tau), sprintf("lambda %.17g", lambda),
               sprintf("sigest %.17g", sigest), "nu 3", sprintf("ndpost %d", ndpost), sprintf("nskip %d", nskip)), paste0(pre, "_meta.txt"))
  write.table(cbind(fit$yhat.train.mean - mean(y)), paste0(pre, "_ref_yhat.txt"), row.names = FALSE, col.names = FALSE)
  write.table(cbind(fit$sigma), paste0(pre, "_ref_sigma.txt"), row.names = FALSE, col.names = FALSE)
  cat(sprintf("titer_%d: n=%d p=%d\n", w, nrow(X), ncol(X)))
}

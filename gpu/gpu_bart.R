# Titer fits on the GPU: gpu_export() writes a fit's input exactly as fastbart::fwbart() would prepare it (same model
# matrix, cutpoints, sigma prior and tau); bart_gpu fits all exported models in one batch; gpu_import() reads a fit back
# as a "wbart" object (sigma draws, tree draws in BART's text format, mu, rm.const), which compile_trees() / predict_draw()
# and the estimator use as before.
gpu_export <- function(x.train, y.train, w, prefix, ndpost, nskip, k = 2, sigdf = 3, sigquant = .90, ntree = 200L, numcut = 100L) {
  temp <- BART::bartModelMatrix(x.train, numcut, usequants = FALSE, cont = FALSE, xinfo = matrix(0.0, 0, 0), rm.const = TRUE)
  X <- temp$X; n <- length(y.train); p <- ncol(X)
  xi <- lapply(seq_len(p), function(v) temp$xinfo[v, seq_len(temp$numcut[v])]); names(xi) <- colnames(X)
  q <- vapply(seq_len(p), function(v) findInterval(X[, v], xi[[v]]), integer(n))
  if (!is.matrix(q)) q <- matrix(q, nrow = n)
  fmean <- mean(y.train); yc <- y.train - fmean
  sigest <- if (p < n) summary(lm(yc ~ ., data.frame(X, yc)))$sigma else sd(yc)
  lambda <- (sigest * sigest * qchisq(1.0 - sigquant, sigdf)) / sigdf
  tau <- (max(yc) - min(yc)) / (2 * k * sqrt(ntree))
  data.table_write <- function(m, f) write.table(m, f, row.names = FALSE, col.names = FALSE, quote = FALSE)
  data.table_write(q, paste0(prefix, "_q.txt"))
  data.table_write(cbind(sprintf("%.17g", yc), sprintf("%.17g", w)), paste0(prefix, "_yw.txt"))
  writeLines(c(sprintf("n %d", n), sprintf("p %d", p), sprintf("numcut %s", paste(lengths(xi), collapse = " ")),
               sprintf("fmean %.17g", fmean), sprintf("tau %.17g", tau), sprintf("lambda %.17g", lambda),
               sprintf("sigest %.17g", sigest), sprintf("nu %d", sigdf), sprintf("ndpost %d", ndpost), sprintf("nskip %d", nskip)),
             paste0(prefix, "_meta.txt"))
  list(xi = xi, rm.const = temp$rm.const, fmean = fmean)
}
gpu_import <- function(prefix, info) {
  fit <- list(sigma = scan(paste0(prefix, "_sigma.txt"), quiet = TRUE),
              treedraws = list(cutpoints = info$xi, trees = readChar(paste0(prefix, "_trees.txt"), file.size(paste0(prefix, "_trees.txt")), useBytes = TRUE)),
              mu = info$fmean, rm.const = info$rm.const)
  class(fit) <- "wbart"
  fit
}

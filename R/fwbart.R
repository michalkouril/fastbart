# Drop-in replacement for BART::wbart (continuous outcome, optional weights w). Same preprocessing and priors as wbart; the
# MCMC runs in the restructured C++ sampler. Unsupported settings (sparse/DART, x.test, n > 65536, p > 32, ntree > 200) fall
# back to BART::wbart.
fwbart <- function(x.train, y.train, x.test = matrix(0.0, 0, 0), sparse = FALSE, xinfo = matrix(0.0, 0, 0), usequants = FALSE,
                   cont = FALSE, rm.const = TRUE, sigest = NA, sigdf = 3, sigquant = .90, k = 2.0, power = 2.0, base = .95,
                   sigmaf = NA, lambda = NA, fmean = mean(y.train), w = rep(1, length(y.train)), ntree = 200L, numcut = 100L,
                   ndpost = 1000L, nskip = 100L, printevery = 100L, ...) {
  temp <- BART::bartModelMatrix(x.train, numcut, usequants = usequants, cont = cont, xinfo = xinfo, rm.const = rm.const)
  X <- temp$X
  n <- length(y.train); p <- ncol(X)
  if (sparse || length(x.test) > 0 || n > 65536 || p > 32 || ntree > 200 || any(temp$numcut > 255) || length(list(...)) > 0) {
    warning("fwbart: unsupported settings, falling back to BART::wbart")
    return(BART::wbart(x.train, y.train, x.test = x.test, sparse = sparse, xinfo = xinfo, usequants = usequants, cont = cont,
                       rm.const = rm.const, sigest = sigest, sigdf = sigdf, sigquant = sigquant, k = k, power = power, base = base,
                       sigmaf = sigmaf, lambda = lambda, fmean = fmean, w = w, ntree = ntree, numcut = numcut, ndpost = ndpost,
                       nskip = nskip, printevery = printevery, ...))
  }
  xi <- lapply(seq_len(p), function(v) temp$xinfo[v, seq_len(temp$numcut[v])])
  q <- vapply(seq_len(p), function(v) findInterval(X[, v], xi[[v]]), integer(n))     # x < xi[c] (0-based)  <=>  q <= c
  if (!is.matrix(q)) q <- matrix(q, nrow = n)
  yc <- y.train - fmean
  nu <- sigdf
  if (is.na(lambda)) {
    if (is.na(sigest)) sigest <- if (p < n) summary(lm(yc ~ ., data.frame(X, yc)))$sigma else sd(yc)
    lambda <- (sigest * sigest * qchisq(1.0 - sigquant, nu)) / nu
  } else sigest <- sqrt(lambda)
  tau <- if (is.na(sigmaf)) (max(yc) - min(yc)) / (2 * k * sqrt(ntree)) else sigmaf / sqrt(ntree)
  rng <- fastbart_rng()
  seed <- if (rng == "internal") sample.int(.Machine$integer.max, 1) else 0L
  ptm <- proc.time()
  res <- fbart_cpp(q, yc, 1 / w^2, integer(0), as.integer(temp$numcut), base / (1 + 0:63)^power, as.integer(ntree),
                   as.integer(nskip), as.integer(ndpost), tau, lambda, nu, sigest, seed, 0L, 0, as.integer(rng == "R"))
  res$proc.time <- proc.time() - ptm
  names(xi) <- colnames(X)
  res$treedraws <- list(cutpoints = xi, trees = res$trees); res$trees <- NULL
  res$mu <- fmean
  res$yhat.train.mean <- res$yhat.train.mean + fmean
  res$yhat.train <- res$yhat.train + fmean
  colnames(res$varcount) <- colnames(X)
  res$varprob <- matrix(1 / p, ndpost, p, dimnames = list(NULL, colnames(X)))
  res$varcount.mean <- colMeans(res$varcount); res$varprob.mean <- colMeans(res$varprob)
  res$rm.const <- temp$rm.const
  class(res) <- "wbart"
  res
}

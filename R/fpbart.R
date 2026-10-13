# Drop-in replacement for BART::pbart (binary outcome, probit link). Same preprocessing, priors and latent-variable sampler as
# pbart; the MCMC runs in the restructured C++ sampler. Unsupported settings (sparse/DART, x.test, n > 65536, p > 32,
# ntree > 200) fall back to BART::pbart.
fpbart <- function(x.train, y.train, x.test = matrix(0.0, 0, 0), sparse = FALSE, xinfo = matrix(0.0, 0, 0), usequants = FALSE,
                   cont = FALSE, rm.const = TRUE, k = 2.0, power = 2.0, base = .95, binaryOffset = NULL, ntree = 50L,
                   numcut = 100L, ndpost = 1000L, nskip = 100L, printevery = 100L, ...) {
  if (length(binaryOffset) == 0) binaryOffset <- qnorm(mean(y.train))
  temp <- BART::bartModelMatrix(x.train, numcut, usequants = usequants, cont = cont, xinfo = xinfo, rm.const = rm.const)
  X <- temp$X
  n <- length(y.train); p <- ncol(X)
  if (sparse || length(x.test) > 0 || n > 65536 || p > 32 || ntree > 200 || any(temp$numcut > 255) || length(list(...)) > 0) {
    warning("fpbart: unsupported settings, falling back to BART::pbart")
    return(BART::pbart(x.train, y.train, x.test = x.test, sparse = sparse, xinfo = xinfo, usequants = usequants, cont = cont,
                       rm.const = rm.const, k = k, power = power, base = base, binaryOffset = binaryOffset, ntree = ntree,
                       numcut = numcut, ndpost = ndpost, nskip = nskip, printevery = printevery, ...))
  }
  xi <- lapply(seq_len(p), function(v) temp$xinfo[v, seq_len(temp$numcut[v])])
  q <- vapply(seq_len(p), function(v) findInterval(X[, v], xi[[v]]), integer(n))     # x < xi[c] (0-based)  <=>  q <= c
  if (!is.matrix(q)) q <- matrix(q, nrow = n)
  tau <- 3 / (k * sqrt(ntree))
  rng <- fastbart_rng()
  seed <- if (rng == "internal") sample.int(.Machine$integer.max, 1) else 0L
  ptm <- proc.time()
  res <- fbart_cpp(q, numeric(n), rep(1, n), as.integer(y.train), as.integer(temp$numcut), base / (1 + 0:63)^power,
                   as.integer(ntree), as.integer(nskip), as.integer(ndpost), tau, 1, 3, 1, seed, 1L, binaryOffset,
                   as.integer(rng == "R"))
  res$proc.time <- proc.time() - ptm
  res$sigma <- NULL; res$yhat.train.mean <- NULL
  names(xi) <- colnames(X)
  res$treedraws <- list(cutpoints = xi, trees = res$trees); res$trees <- NULL
  res$yhat.train <- res$yhat.train + binaryOffset
  res$prob.train <- pnorm(res$yhat.train)
  res$prob.train.mean <- apply(res$prob.train, 2, mean)
  colnames(res$varcount) <- colnames(X)
  res$varprob <- matrix(1 / p, ndpost, p, dimnames = list(NULL, colnames(X)))
  res$varcount.mean <- colMeans(res$varcount); res$varprob.mean <- colMeans(res$varprob)
  res$rm.const <- temp$rm.const
  res$binaryOffset <- binaryOffset
  class(res) <- "pbart"
  res
}

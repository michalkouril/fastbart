# Parse a fit's tree draws once; then predict_draw() evaluates a single posterior draw quickly (same values as
# pwbart(newdata, fit$treedraws, mu)[m, ]). Works for wbart, pbart, fwbart and fpbart fits.
compile_trees <- function(fit) {
  ct <- compile_trees_cpp(fit$treedraws$trees, unname(lapply(fit$treedraws$cutpoints, as.numeric)))
  ct$mu <- if (!is.null(fit$binaryOffset)) fit$binaryOffset else if (length(fit$mu)) fit$mu else fit$offset
  ct$rm.const <- fit$rm.const
  ct$probit <- inherits(fit, "pbart")
  class(ct) <- "fastbart_trees"
  ct
}
# newdata: data.frame or matrix with the training columns (constant columns are dropped as in the fit); m: draw (1-based).
# Returns f (+ mu); for a probit fit use pnorm() of the result for probabilities.
predict_draw <- function(ct, newdata, m) {
  if (!is.null(ct$rm.const) && is.numeric(ct$rm.const) && length(ct$rm.const) < NCOL(newdata))
    newdata <- newdata[, ct$rm.const, drop = FALSE]
  cols <- if (is.data.frame(newdata)) lapply(newdata, as.numeric) else lapply(seq_len(ncol(newdata)), function(j) as.numeric(newdata[, j]))
  predict_draw_cpp(ct, cols, as.integer(m), ct$mu)
}

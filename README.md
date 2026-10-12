# fastbart

Drop-in replacement for `BART::wbart()` (continuous outcome, optional weights `w`), using the same sampler
restructured for speed.

```r
# remotes::install_github("michalkouril/fastbart")
library(fastbart)
fit <- fwbart(x.train, y.train, w = wt, ndpost = 100, nskip = 1000)   # same arguments as BART::wbart
predict(fit, newdata)                                                # returns a "wbart" object: BART's predict()/pwbart() work
```

## What is the same as `BART::wbart`

- Preprocessing in R is identical: `bartModelMatrix()` (cutpoints, constant-column removal), the `lm()`-based `sigest`,
  `lambda` from `sigdf`/`sigquant`, `tau` from `k`.
- Sampler: birth/death proposals with BART's proposal probabilities, tree prior `base/(1+depth)^power`, conjugate leaf
  (mu) draws with per-observation weights, inverse-chi-square sigma draw.
- Returned object: `sigma` (all `nskip + ndpost` draws), `yhat.train`, `yhat.train.mean`, `treedraws` (cutpoints + trees in
  BART's text format), `varcount`, `varprob`, `mu`, `rm.const`, `proc.time`; class `"wbart"`.

## What is different

- Data layout: each observation stores its current leaf in every tree, so one pass over the data per tree gives every
  sufficient statistic (birth split, death children, all leaves for the mu draw) and applies the previous tree's update.
  `BART::wbart` makes about four tree-traversal passes per tree.
- Its own random-number generator (xoshiro128\*\*, Box-Muller, Marsaglia-Tsang), so draws are not bit-identical to
  `BART::wbart` for the same `set.seed()`; R's RNG only seeds it. Results agree statistically (see below).
- Not supported, so the call falls back to `BART::wbart` with a warning: `sparse = TRUE` (DART), `x.test`, `n > 8192`,
  more than 32 predictors, `ntree > 200`, more than 255 cutpoints, or extra arguments. Trees are capped at 64 nodes
  (counted in `fit$moves[3]`; never reached in testing).

## Benchmark

Six weighted titer models (n = 3,091-4,573, p = 15-19) from a BART g-formula pipeline: ntree = 200, nskip = 1000,
ndpost = 100. Single thread, both packages built with R's default flags (`-O2`), called alternately from the same R
session and pinned to the same core, 2 rounds.

| | 12 fits, total wall time |
|---|---|
| `BART::wbart` (2.9.10) | 152.4 s |
| `fastbart::fwbart` | 57.3 s |
| speed-up | **2.66x** |

Agreement with `BART::wbart` (posterior mean of f): correlation 0.992-0.998, the same range as two `BART::wbart` runs with
different seeds (0.993-0.997). Posterior mean sigma is within 1.2%.
`predict()` on an `fwbart` fit reproduces its `yhat.train` to 1e-9.

## Source

`src/bart_kernel.cpp` is shared with an FPGA (Vitis HLS) version of the same sampler; in this package it is compiled
with `-DFAST_CPU`, so the HLS pragmas are ignored. License: GPL (>= 2), as BART.

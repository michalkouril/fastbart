# fastbart

Drop-in replacements for `BART::wbart()` (continuous outcome, optional weights `w`) and `BART::pbart()` (binary outcome,
probit), using the same samplers restructured for speed.

```r
# remotes::install_github("michalkouril/fastbart")
library(fastbart)
fit <- fwbart(x.train, y.train, w = wt, ndpost = 100, nskip = 1000)   # same arguments as BART::wbart
predict(fit, newdata)                                                # returns a "wbart" object: BART's predict()/pwbart() work

pfit <- fpbart(x.train, y01, ndpost = 100, nskip = 1000)            # same arguments as BART::pbart
predict(pfit, newdata)$prob.test                                     # returns a "pbart" object

ct <- compile_trees(fit)                                             # any wbart/pbart/fwbart/fpbart fit
f7 <- predict_draw(ct, newdata, 7)                                   # = pwbart(newdata, fit$treedraws, fit$mu)[7, ]
```

## Fast single-draw prediction

Simulations that step through posterior draws one at a time (e.g. a g-formula forward simulation) often call
`pwbart()` on a single draw thousands of times; each call re-parses the tree text and rebuilds the model matrix.
`compile_trees(fit)` parses a fit's trees once into flat arrays; `predict_draw(ct, newdata, m)` then evaluates draw `m` in
C++ directly on the columns of a data frame or matrix (constant columns dropped as in the fit; for a probit fit, take
`pnorm()` of the result).

The results are identical to `pwbart()[m, ]`: the same comparison (`x < cutpoint` goes left) and the same summation
order over trees. Each node's two children are stored next to each other, so a step is branch-free, and trees are
evaluated tree-outer and row-inner so the CPU overlaps many rows: 2.16x faster than a row-outer loop on the same layout
(600 predictions, n = 5,237, 200 trees).

In the g-formula pipeline below, using these for the per-draw predictions (plus replacing an `apply(..., cumprod)` with
a column loop) made the forward simulation of one replicate 3.4x faster (36.1 s to 10.5 s, both run side by side) with
byte-identical output.

## What is the same as `BART::wbart` / `BART::pbart`

- Preprocessing in R is identical: `bartModelMatrix()` (cutpoints, constant-column removal), the `lm()`-based `sigest`,
  `lambda` from `sigdf`/`sigquant`, `tau` from `k`.
- Sampler: birth/death proposals with BART's proposal probabilities, tree prior `base/(1+depth)^power`, conjugate leaf
  (mu) draws with per-observation weights, inverse-chi-square sigma draw.
- Returned object: `sigma` (all `nskip + ndpost` draws), `yhat.train`, `yhat.train.mean`, `treedraws` (cutpoints + trees in
  BART's text format), `varcount`, `varprob`, `mu`, `rm.const`, `proc.time`; class `"wbart"`.
- `fpbart`: pbart's latent-variable sampler (z redrawn every iteration from BART's truncated-normal sampler, sigma fixed at 1),
  pbart's defaults (`ntree = 50`, `tau = 3/(k*sqrt(ntree))`, `binaryOffset = qnorm(mean(y))`). Returned object: `yhat.train`
  (with offset), `prob.train`, `prob.train.mean`, `treedraws`, `varcount`, `varprob`, `binaryOffset`, `rm.const`,
  `proc.time`; class `"pbart"`.

## What is different

- Data layout: each observation stores its current leaf in every tree, so one pass over the data per tree gives every
  sufficient statistic (birth split, death children, all leaves for the mu draw) and applies the previous tree's update.
  `BART::wbart` makes about four tree-traversal passes per tree.
- Its own random-number generator (xoshiro128\*\*, Box-Muller, Marsaglia-Tsang), so draws are not bit-identical to
  `BART::wbart` for the same `set.seed()`; R's RNG only seeds it. Results agree statistically (see below).
- Not supported, so the call falls back to `BART::wbart` with a warning: `sparse = TRUE` (DART), `x.test`, `n > 65536`,
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

`fpbart`: censoring-hazard model from the same pipeline (n = 24,139 person-weeks, p = 3, ntree = 50, nskip = 1000,
ndpost = 100), 2 rounds alternating, same core:

| | per fit |
|---|---|
| `BART::pbart` | 21.4 s |
| `fastbart::fpbart` | 5.4 s |
| speed-up | **3.95x** |

Agreement (posterior mean probability per row): correlation with `BART::pbart` 0.9873, versus 0.9864 between two
`BART::pbart` runs with different seeds; mean predicted probability 0.0883 vs 0.0880 (observed 0.0889). `predict()` on an
`fpbart` fit reproduces its `prob.train` to 1e-10.

### Full pipeline: patient bootstrap of a BART g-formula

The six titer models above are refit on every bootstrap resample of a BART g-formula pipeline (also: a CBPS model, a
`pbart` censoring-hazard model and a forward simulation in R, all unchanged). Swapping `wbart` for `fastbart::fwbart`
in the titer fits (a one-line change), B = 100 resamples + the original data (101 replicates), 10 replicates in parallel,
nskip = 1000, ndpost = 100, on an i9-14900K (32 threads):

| | wall time, 101 replicates |
|---|---|
| `BART::wbart` | 24.2 min |
| `fastbart::fwbart` | 16.0 min |
| speed-up | **1.51x** |

The whole-pipeline gain is smaller than the per-model 2.66x because the titer fits are about 65% of a replicate; the
`pbart` hazard model (about 17%) and the R simulation (about 14%) are untouched. Pipeline results agree: point estimates
within 0.015 across all 18 reported differences, bootstrap SDs within about 7%, and identical bootstrap-interval coverage
of the true values (17 of 18 in both). The fwbart run had extra background load (FPGA builds, load average about 4); a
back-to-back rerun of the wbart baseline under the same conditions is pending, as is a full run with all three changes
(`fwbart`, `fpbart`, fast prediction).

## Source

`src/bart_kernel.cpp` is shared with an FPGA (Vitis HLS) version of the same sampler; in this package it is compiled
with `-DFAST_CPU`, so the HLS pragmas are ignored. License: GPL (>= 2), as BART.

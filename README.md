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

## Random numbers and reproducibility

By default (since 0.2.0) `fwbart` and `fpbart` draw from R's own generator (`unif_rand`, `norm_rand`, `exp_rand`,
`rchisq`), like BART: results follow `set.seed()` and are **bit-identical across platforms and R versions**. Tested:
three bootstrap replicates of the g-formula pipeline below gave identical values (162 of 162) on Windows 10 / R 4.5.2,
Linux / R 4.5.2 and Linux / R 4.1.2. They are still not bit-identical to `BART::wbart` / `pbart` (the sampler consumes the
random numbers in a different order); see `docs/bit-exact.md`.

The built-in xoshiro128** generator of 0.1.0 is still available with the environment variable `FASTBART_RNG=internal`
(`fastbart_rng()` reports the current setting). It is a little faster (about 3% on the hazard model, up to about 15% on the
titer models), but its normal draws use the platform's single-precision math library, so results then differ between
platforms by Monte Carlo error (e.g. Windows vs Linux: estimates within 0.036, the same size as fast vs original on one
machine). The benchmarks below say which generator they used.

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

**Clean back-to-back run (idle machine).** Original pipeline, then the full fast stack (`fwbart` + `fpbart` +
`compile_trees`/`predict_draw`), same command (B = 100 + the original data = 101 replicates, 10 in parallel,
nskip = 1000, ndpost = 100, seed 1), on an otherwise idle i7-8700B (6 cores / 12 threads), R 4.5.2 in Docker
(`rocker/r-ver:4.5.2`), BART 2.9.10, CBPS 0.24, fastbart 0.1.0 (`98663d8`, internal generator), single-threaded BLAS
(`OPENBLAS_NUM_THREADS=1`):

| pipeline | wall time | speed-up |
|---|---|---|
| original | 69.8 min | 1.00x |
| `fwbart` + `fpbart` + `compile_trees`/`predict_draw` | 26.1 min | **2.67x** |

The machine load stayed at the 10 benchmark workers throughout (192 samples, maximum 10.3); no other process used
more than 50% CPU (the one logged sighting was the load sampler's own `cut` command). Results agree as on the other
machine: all 101 replicates succeed in both runs, point estimates within 0.027, bootstrap SDs 0.99-1.09x, identical
coverage (17 of 18). The original pipeline on this machine (R 4.5.2) matches the one on the i9-14900K (R 4.1.2) to
1.1e-7 across all 1,818 replicate rows (not bit-identical: different BLAS libraries in `lm()`/CBPS).

**Back-to-back run on a nearly idle i9-14900K** (32 threads), R 4.1.2, BART 2.9.10, CBPS 0.24, fastbart 0.2.0 with the
default generator (R's), same command:

| pipeline | wall time | speed-up |
|---|---|---|
| original | 24.5 min | 1.00x |
| `fwbart` + `fpbart` + `compile_trees`/`predict_draw` (R's generator) | 7.5 min | **3.27x** |

Nearly idle: one unrelated process used about 1.6 of the 32 threads throughout both runs (same load for both); the
machine was otherwise quiet. Results agree as above: all replicates succeed in both runs, point estimates within 0.022,
bootstrap SDs 0.93-1.16x, identical coverage (17 of 18).

**Back-to-back runs on an idle Core Ultra 7 268V laptop** (Lunar Lake, 8 cores, on AC power, performance profile),
R 4.5.2 (Ubuntu package), BART 2.9.10, CBPS 0.24, fastbart 0.2.0 with the default generator (R's), single-threaded BLAS,
same command. The four runs alternate (original, fast, fast, original) so that warm-up and heat affect both pipelines
equally:

| run | pipeline | wall time |
|---|---|---|
| 1 | original | 49.7 min |
| 2 | `fwbart` + `fpbart` + `compile_trees`/`predict_draw` | 13.9 min |
| 3 | `fwbart` + `fpbart` + `compile_trees`/`predict_draw` | 14.0 min |
| 4 | original | 49.5 min |
| | **speed-up (49.6 / 13.95 min)** | **3.56x** |

The CPU averaged 3.2 GHz at 77-81 C in every run (no sustained throttling); the machine load stayed at the 10 benchmark
workers. One sample in run 1 caught the desktop's file indexer at full CPU (it was then stopped). Repeated runs give
identical results, and the fast runs are bit-identical to the fastbart 0.2.0 run on the i7-8700B above (R 4.5.2 in
Docker), as the default generator makes them. Results agree with the original as on the other machines (estimates within
0.022, bootstrap SDs 0.93-1.16x, coverage 17 of 18 in both).

**Runs on an i9-14900K that was not idle.**

The six titer models above are refit on every bootstrap resample of a BART g-formula pipeline, together with a CBPS model,
a `pbart` censoring-hazard model and a forward simulation in R. B = 100 resamples + the original data (101 replicates),
10 replicates in parallel, nskip = 1000, ndpost = 100, seed 1, on an i9-14900K (32 threads, R 4.1.2), fastbart 0.1.0
(internal generator).

| pipeline | wall time | speed-up |
|---|---|---|
| original (`BART::wbart`, `BART::pbart`, `pwbart` per draw) | 24.2 min | 1.00x |
| titer fits on `fwbart` | 16.0 min | 1.51x |
| `fwbart` + `fpbart` + `compile_trees`/`predict_draw` | 8.0 min | ~3.0x |
| original, rerun | 28.2 min | |

Other jobs were running on this machine during these runs, and not equally during each (the original pipeline took
24.2 min in one run and 28.2 min in another), so these speed-ups are indicative; the clean comparison is the one above.
The changes to the pipeline script are the two model calls (`wbart(` to `fastbart::fwbart(`, `pbart(` to
`fastbart::fpbart(`), the per-draw prediction helpers (`compile_trees`/`predict_draw` instead of re-parsing trees for
`pwbart`), and an `apply(..., cumprod)` replaced by a column loop.

Pipeline results agree with the original: all 101 replicates succeed; point estimates within 0.027 across all 18 reported
differences (0.015 with only `fwbart`); bootstrap SDs 0.99-1.09x the original's; identical bootstrap-interval coverage of
the true values (17 of 18 in both). The remaining differences are Monte Carlo: the fast samplers use their own
random-number streams. With the original model fits, the fast prediction path alone gives byte-identical output.

## FPGA proof of concept (AMD/Xilinx Alveo U200)

The same sampler was also built as an FPGA kernel, to gauge whether an FPGA could speed up the BART fits further.
Result: it works and is **2.4x faster than one CPU core running `fwbart`** (6.5x faster than `BART::wbart`), but one
kernel on one card is **about 3-4x slower than 10 CPU cores** running `fwbart` in parallel, so for the bootstrap the CPU
version is the better choice.

**Design.** `src/bart_kernel.cpp` compiled with Vitis HLS 2023.2 for the U200 (`xilinx_u200_gen3x16_xdma_2_202110_1`,
XRT 2.17). Same algorithm as `fwbart` (birth/death moves, one data pass per tree using per-row leaf ids), with the data
pass in fixed point (exact sums), 8 rows per clock cycle, the tree-control arithmetic in single precision and an on-chip
xoshiro128** generator. FPGA build settings: 32 node slots per tree and 8 lanes (`-DNODES=32 -DKL=8`; see `fpga/`); in the FPGA build the
node masks are one bit per slot and are scanned one node per cycle. Limits:
up to 8,192 rows, 32 predictors, 200 trees.

**Hardware result: six weighted titer models** (ntree = 200, nskip = 1000, ndpost = 100), one kernel at 100 MHz:

| model | n | U200 | `BART::wbart`, 1 core | correlation with `wbart` | sigma (U200 / `wbart`) |
|---|---|---|---|---|---|
| titer 1 | 4,573 | 2.24 s | 15.2 s | 0.9950 | 0.2010 / 0.2008 |
| titer 2 | 4,078 | 2.10 s | 13.4 s | 0.9941 | 0.1952 / 0.1951 |
| titer 3 | 3,666 | 1.98 s | 12.6 s | 0.9972 | 0.1932 / 0.1951 |
| titer 4 | 3,377 | 1.90 s | 11.6 s | 0.9968 | 0.1909 / 0.1909 |
| titer 5 | 3,208 | 1.85 s | 12.2 s | 0.9971 | 0.1906 / 0.1929 |
| titer 6 | 3,091 | 1.82 s | 11.6 s | 0.9973 | 0.1875 / 0.1892 |
| **all 6** | | **11.9 s** | **76.7 s** | | |

`fastbart::fwbart` on one core takes 28.7 s for all six (about 4.8 s per fit; 0.1.0, R's default compiler flags), so the
card is 2.4x faster than `fwbart` and 6.5x faster than `BART::wbart`. CPU times on an i9-14900K, one core each. Data transfers to and from the
card add about 3 ms per fit. The card's output is **bit-identical** to the bit-accurate C++ simulation of the kernel
(same Metropolis-Hastings decisions, sigma draws and fitted values for the same seed) for all six models; agreement with
`BART::wbart` is within seed-to-seed MCMC variation, as for `fwbart`.

**Throughput.** The card runs one fit at a time, so it is worth about 2.4 CPU cores of `fwbart`. The bootstrap's 606
titer fits would take about 20 min on the card versus about 5-6 min on 10 cores (`fwbart`, including the slowdown of
cores sharing a machine). The host CPU is nearly idle while the card works, so the card could run alongside CPU
workers.

**Getting there (each build: HLS minutes, place-and-route 1.5-4 h):**

| build | outcome |
|---|---|
| 16 lanes, 64 node slots, 300 MHz | did not fit (placement: 26,219 CLBs needed, 24,246 free) |
| 16 lanes, 32 node slots, 300 MHz | unroutable congestion (stopped after 3 h of rip-up; worst path 6.4 ns over) |
| 16 lanes, 32 node slots, 200 MHz | routing failed (conflicting nets at the per-tree leaf mask) |
| 8 lanes, 32-bit masks, sequential mask scan, 200 MHz | placed and routed, but 2.6-4 ns over budget; stopped |
| same, **100 MHz** | **met timing (up to 102-105 MHz); runs on the card** |

The node-slot count was reduced from 64 to 32 because trees here use at most 13 nodes (16 slots hit the cap during
burn-in; 32 never did). The 100 MHz kernel uses about 257k LUTs, 195k flip-flops, 400 URAM, 312 BRAM and 383 DSP blocks
(HLS estimate) and is placed across two of the U200's three SLRs; the SLR crossings are the main reason 200 MHz did not
close. On one occasion loading an xclbin crashed the XRT driver (NULL dereference in `ert_ctrl_unset_xgq`), which needed
a reboot; after the reboot the same design loaded and ran normally.

**What could make it faster (estimates, not built):** keeping the kernel within one SLR to reach 200-250 MHz (2-2.5x);
overlapping each tree's control steps with its data pass (about 1.5x); packing the per-row leaf ids more tightly (8-bit
ids in 72-bit URAM words today) so that two or three kernels fit on the card (2-3x throughput). Together roughly
0.15-0.3 s per fit for the card, about the throughput of 10-20 CPU cores of `fwbart`.

## Source

`src/bart_kernel.cpp` is shared with an FPGA (Vitis HLS) version of the same sampler; in this package it is compiled
with `-DFAST_CPU`, so the HLS pragmas are ignored. The FPGA build (host programs, data export, Makefile) is in `fpga/`. License: GPL (>= 2), as BART.

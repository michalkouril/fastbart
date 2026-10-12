# Making fwbart / fpbart bit-identical to BART::wbart / BART::pbart

Status: design note only, not implemented. Written against BART 2.9.10 (`cwbart.cpp`, `heterbart.cpp`, `heterbd.cpp`,
`heterbartfuns.cpp`, `cpbart.cpp`, `bart.cpp`, `bd.cpp`, `bartfuns.cpp`, `tree.cpp`, `rn.h`, `rtnorm.cpp`) and fastbart
`98663d8` (`src/bart_kernel.cpp`, `src/glue.cpp`, `R/fwbart.R`, `R/fpbart.R`).

## Goal and scope

Given the same `set.seed()` and inputs, `fwbart()` returns the same `sigma`, `yhat.train`, `yhat.train.mean`,
`treedraws$trees` text and `varcount` as `wbart()`, bit for bit; likewise `fpbart()` vs `pbart()`. A pipeline that uses
them then produces byte-identical output, because R's random stream is left in the same state for the code that follows the
fits.

"Same" means the same machine, the same R build and the same compiler flags. BART is not bit-reproducible across
platforms (libm `log`/`exp`/`pow`, FMA contraction, compilers), so matching a run made on another machine (for example
the original Windows R 4.5.2 runs) is out of scope even for BART itself.

The CPU build only (`-DFAST_CPU`). The FPGA kernel cannot be bit-exact (see the last section).

`compile_trees` / `predict_draw` are already bit-identical to `pwbart()[m, ]`; nothing to do there.

## What currently differs

Two kinds of difference, both needed:

1. **Random numbers.** fastbart uses its own generator (xoshiro128**, Box-Muller, Marsaglia-Tsang) and does not consume
   R's stream. BART draws from R's stream (`rn.h`, Rcpp branch: `arn` holds an `Rcpp::RNGScope`; `normal()` =
   `norm_rand()`, `uniform()` = `unif_rand()`, `chi_square(df)` = `rchisq(df)`, `exp()` = `exp_rand()`,
   `discrete()` = `rmultinom(1, wts, p, ...)` with weights normalized in `set_wts`).
2. **Floating-point order.** Sums and update formulas are evaluated in a different order, or in float instead of double.
   One flipped accept/reject decision changes the rest of the chain.

## 1. Random numbers: same generator, same calls, same order

Use R's API (`unif_rand`, `norm_rand`, `exp_rand`, `rchisq`, `rmultinom`) through an `rn`-like class, inside an
`Rcpp::RNGScope` (Rcpp attributes already add one to `fbart_cpp`). Remove the `sample.int()` seed draw from
`fwbart()` / `fpbart()`: it consumes R's stream, which BART does not do.

Every call BART makes has to be made, in the same order, including draws whose result is discarded.

### wbart, per MCMC iteration (`cwbart.cpp` loop, `heterbart::draw`)

For each tree j = 0..m-1 (`heterbd`, then `heterdrmu`):

1. `uniform()`: birth if `< PBx`.
2. Birth (`bprop`):
   - `uniform()`: index into `goodbots` (`floor(u * goodbots.size())`).
   - `discrete()` with weights `pv` (uniform 1/p with DART off): the variable v, over **all** p variables.
   - If v is a good variable for nx: `uniform()` for the cutpoint (`L + floor(u * (U - L + 1))`). If v is bad:
     no draw; `c = getbadcut(v)` (cutpoint of the nearest ancestor that splits on v).
   - Then `uniform()` (`uu`) is drawn **always**, even when `nl < 5 || nr < 5` makes `alpha = 0`, and even for a bad
     variable (which always ends up rejected: one child is empty).
   - If accepted: `normal()` for `mul`, then `normal()` for `mur` (in `heterdrawnodemu`). These values are overwritten by
     `heterdrmu` straight after, but the draws must be made.
3. Death (`dprop`):
   - `uniform()`: index into the nog nodes (`floor(u * nognds.size())`).
   - `uniform()` for the Metropolis test (`log(u) < lalpha`).
   - If accepted: `normal()` for `mu` (also overwritten by `heterdrmu`).
4. `heterdrmu`: one `normal()` per bottom node, **in `getbots` order** (left-first depth-first), whatever its storage slot.

After all trees: `chi_square(n + nu)` for sigma. (DART is off, so `draw_s` / `draw_theta0` make no draws.)

**Node orders must be BART's:** `goodbots` is `getbots` order filtered by `cansplit`; nog nodes are in `getnogs` order
(depth-first, left first; a node with grandchildren recurses into the children that have children). fastbart's kernel
currently uses storage-slot order for `select_kth` and for the leaf draws. Keep slot storage, but enumerate in
depth-first order when choosing and when drawing.

### pbart (`cpbart.cpp`, `bart::draw`, `bd`, `drmu`)

- Before the loop: for each row k in order, the initial latent value: `y == 0 ? -rtnorm(0, offset, 1) :
  rtnorm(0, -offset, 1)`.
- Per iteration: the same per-tree sequence as above with the homoscedastic functions (`bd`, `getsuff`, `lh`,
  `drawnodemu`, `drmu`; sigma fixed at 1), then for each row k in order:
  `y == 0 ? -rtnorm(-f_k, offset, 1) : rtnorm(f_k, -offset, 1)`.
- `rtnorm` (`rtnorm.cpp`) must be copied exactly: for `tau <= 0`, repeat `normal()` until `>= tau`; else repeat
  `z = exp()/lambda + tau` with the test `uniform() > exp(-0.5 * pow(z - lambda, 2.))`. Note `pow(., 2.)`, not a
  multiplication, and `exp()` = `exp_rand()`.

## 2. Floating point: same operations, same order, double throughout

### Types

- Everything in double: y, weights, `tau`, `lambda`, `nu`, `sigest`, `sigma`, `offset`, leaf values, all control math
  (`lh`, `drawnodemu`, `pr`, `lalpha`). Today `glue.cpp` converts y and weights to float, and `bart_fit_cpu` takes
  float `tau`/`lambda`/`nu`/`sigest`/`offset` and does the Metropolis math in float.
- `sigma_out` as double (BART's `sdraw`). Leaf values stay double, and the tree text is written from the doubles with
  `precision(10)` (already so in `glue.cpp`).

### Running fit and residuals (`heterbart::draw`)

BART, per tree: `allfit[k] -= ftemp[k]` (old tree), `r[k] = y[k] - allfit[k]`; after the move and the mu draw,
`allfit[k] += ftemp[k]` (new tree). Mirror it exactly in the fused pass:

- store `am = af - th_old[leaf]` (this is BART's `allfit` while tree j is being updated), use `r = y - am`;
- in the next pass, `af = am + th_new[leaf']`.

Today the pass computes `af - thOld + thNew` and `r = y - af + th`, which round differently.

Initial `allfit` is BART's `predict()` over trees whose leaves are 0 (all zeros).

### Sufficient statistics

- **One accumulator per leaf, summed in row order.** BART sums each leaf's statistics over rows 0..n-1 in order.
  fastbart's single pass already visits rows in increasing order, but in the CPU build it splits them into 16 lanes and adds
  the lanes at the end. Use a single accumulator per leaf on the CPU (the lanes are only there for the FPGA).
- **Weights as BART computes them:** `svec[k] = iw[k] * sigma` (per iteration), then inside the sums
  `w = 1.0 / (svec[k] * svec[k])`, `b += w`, `M += w * r[k]` (`hetergetsuff`, `heterallsuff`). Today fastbart sums
  `wt * r` with `wt = 1/iw^2` and divides by sigma^2 afterwards. Precompute `w[k]` once per iteration (identical value).
  For pbart (`getsuff`, `allsuff`): `n` as a count and `sy += r[k]`.
- **Birth:** the split statistics for nx (`hetergetsuff` with v, c) are row-order sums over nx's rows, split by
  `x[v] < xi[v][c]`; already computed that way in the pass. When the birth is accepted, BART's `heterdrmu` recomputes
  the new leaves' sums over the same rows in the same order, so the split sums can be reused unchanged.
- **Death:** BART's merged leaf sum (in `heterdrmu` after an accepted death) runs over the union of the two children's
  rows in row order, which is not `S[l] + S[r]`. Add one more accumulator in the pass: rows whose leaf is l or r.
  The death test itself uses the separate children's sums (`hetergetsuff(l, r)`), as now.
- `nl`, `nr` as counts (for the `>= 5` test).

### Formulas copied verbatim

- `heterlh(b, M, tau)`: `t2 = tau*tau; k = b*t2 + 1; return -.5*log(k) + .5*M*M*t2/k`.
- `heterdrawnodemu(b, M, tau)`: `muhat = M/b; a = 1/(tau*tau); return (b*muhat)/(a+b) + normal()/sqrt(a+b)`. Note
  `(b*muhat)/(a+b)`, not `M/(a+b)`.
- Homoscedastic `lh` / `drawnodemu` for pbart, as in `bartfuns.cpp`.
- `pr` in `bprop` / `dprop`: the same expression and the same order of multiplication and division; `PGnx`, `PGly`,
  `PGry`, `PGny`, `PGlx`, `PGrx` from `alpha/pow(1.0 + depth, mybeta)` (a table of these `pow` results is fine: same calls,
  same values), including the `goodvars.size() > 1` branch and the `getbadcut` path.
- `lalpha = log(pr) + (lhl + lhr - lht)` for birth, `log(pr) + (lht - lhl - lhr)` for death; `std::min(0.0, lalpha)`;
  birth accepted iff `alpha > 0 && log(uu) < lalpha`.
- Sigma: `rss += ((y[k] - f[k]) / iw[k])^2` over k in order (as `restemp*restemp`), then
  `sigma = sqrt((nu*lambda + rss) / chi_square(n + nu))`.

### Outputs

- `yhat.train` (BART's `trdraw`) is the running `allfit` at the end of each kept iteration, not a re-sum over the trees
  (`glue.cpp` currently re-sums); `yhat.train.mean` is the running sum of those divided by `ndpost`, as in `cwbart.cpp`.
- `varcount`: BART's `nv` (incremented on birth, decremented on death), the same as the current count of splits per
  variable.

### Limits that must not bite

- Node cap (64 slots) and `MAXDEPTH` guard: never reached in testing, but if reached the chain differs. For bit-exactness,
  fall back to `BART::wbart`/`pbart` (or grow storage) instead of rejecting the move.
- Rows: the CPU build allows 65,536.

### Compiler flags

Build with R's default flags, as BART is, and add `-ffp-contract=off` so that no FMA contraction happens on targets
that have FMA (e.g. ARM / Apple silicon, or `-march=native` on x86-64). BART itself is compiled without that flag, so on
FMA targets BART's own result depends on the compiler; check there.

## Verification plan

1. **Call trace.** Instrument both (BART under `NoRcpp`-style logging or a patched copy, fastbart in a debug build) to log
   each RNG call (kind, value) and each accept/reject; compare until the first divergence. This finds order mistakes
   quickly.
2. **Unit fits.** For several seeds: `identical()` on `sigma`, `yhat.train`, `yhat.train.mean`, `varcount` and the
   `treedraws$trees` string, on
   - the six titer models and the hazard model of the g-formula pipeline;
   - synthetic edge cases: binary and constant columns, ties, few cutpoints (so bad-variable proposals happen),
     very small leaves (`nl < 5`), deep trees, weights far from 1, `n` not a multiple of 16.
3. **Pipeline.** Replicate 0 with `fwbart` + `fpbart` + `predict_draw`: the four output CSVs byte-identical to the original
   pipeline's; then a full bootstrap run, `replicates.csv` identical.
4. **Keep it identical.** Add these comparisons as package tests so later speed work cannot silently break them.

## Expected cost (estimate, not measured)

The restructuring that gives the speed-up stays: one pass over the rows per tree, with per-row leaf ids. What is added:

- single accumulators instead of 16 lanes (CPU only; lanes do not help on a CPU);
- a per-iteration weight array, and one extra accumulator for death moves;
- R's RNG for a handful of calls per tree, plus `rmultinom` per birth proposal;
- for `fpbart`, R's inversion `norm_rand` / `exp_rand` for the latent draws (about 24k per iteration on the hazard model),
  which is slower than Box-Muller; BART pays the same cost.

Guess: `fwbart` 10-25% slower than now (about 2.0-2.4x faster than `wbart` instead of 2.66x); `fpbart` about
3-3.5x faster than `pbart` instead of 3.95x. To be measured.

Suggested as an option, e.g. `fwbart(..., exact = TRUE)`, keeping the current faster mode (statistically equivalent,
not bit-identical) as the default; or the other way round if reproducibility against existing BART results matters more.

## FPGA

Not feasible. Row-order double-precision sums are a chain of dependent floating-point additions (about 7 cycles each),
so the data pass drops from 16 rows per cycle to about one row per 7 cycles; R's Mersenne Twister, inversion normals and
`rmultinom` would also have to be reproduced on chip. Bit-exactness and the FPGA speed-up are mutually exclusive.

// BART (heteroscedastic / weighted, as in BART::wbart) MCMC kernel -- FPGA proof of concept.
// Same sampler as the BART package (birth/death proposals, Chipman-George-McCulloch tree prior, conjugate leaf
// draws, inverse-chi^2 sigma draw), restructured for hardware:
//   * each row stores its current leaf id per tree, so one pass over the data per tree computes every sufficient
//     statistic (birth split, death children, all leaves for the mu draw) and applies the previous tree's update;
//   * K lanes process K rows per clock cycle; per-row arithmetic is fixed point (bit-exact sums, no drift).
#pragma once
#include <cstdint>

#ifndef FAST_CPU
#include <ap_int.h>
#include <ap_fixed.h>
#endif

#ifdef FAST_CPU
#define NMAX   65536     // max training rows (CPU build)
#else
#define NMAX   8192      // max training rows (FPGA build: on-chip memory)
#endif
#define PMAX   32        // max predictors
#define MMAX   200       // max trees
#ifndef NODES
#define NODES  64        // node slots per tree (the FPGA build uses -DNODES=32)
#endif
#ifndef KL
#define KL     16        // lanes (rows per clock; the FPGA build uses -DKL=8)
#endif
#ifdef FAST_CPU
#define NL     (NMAX / KL + 24)   // padded lane stride: avoids every lane mapping to the same cache sets
#else
#define NL     (NMAX / KL)
#endif
#define MAXDEPTH 24

#ifdef FAST_CPU
typedef double fx_t;     // fitted values / residuals
typedef double wt_t;     // per-row precision weight 1/iw^2
typedef double acc_t;    // accumulators
typedef uint8_t u8;
typedef uint64_t mask_t;
#else
typedef ap_fixed<40, 12> fx_t;    // +-2048, 28 fractional bits
typedef ap_ufixed<20, 4> wt_t;    // [0,16), 16 fractional bits
typedef ap_fixed<64, 24> acc_t;   // 40 fractional bits
typedef ap_uint<8> u8;
typedef ap_uint<NODES> mask_t;   // one bit per node slot
#endif

extern "C" void bart_wfit(const uint8_t* qin,      // [p][n] cutpoint rank of x: x < xi[v][c] <=> q <= c
                          const float* yin,        // [n] centred y
                          const float* wtin,       // [n] 1/iw^2   (wbart's w = iw)
                          const int* numcut,       // [p]
                          const float* pgtab,      // [64] alpha/(1+d)^beta
                          float* sigma_out,        // [nskip+ndpost]
                          float* trmean_out,       // [n] posterior mean of f (centred)
                          uint64_t* trees_out,     // [ndpost][m][NODES] packed nodes (if keep_trees)
                          int n, int p, int m, int nskip, int ndpost, float tau, float lambda, float nu, float sigest,
                          unsigned seed, int keep_trees, int* stats_out);

#ifdef FAST_CPU
// CPU build only: probit = 1 runs BART::pbart's sampler (latent z, sigma fixed at 1); ybin are the 0/1 outcomes and offset is
// pbart's binaryOffset. probit = 0 is bart_wfit.
extern "C" void bart_fit_cpu(const uint8_t* qin, const float* yin, const float* wtin, const int* numcut, const float* pgtab,
                             float* sigma_out, float* trmean_out, uint64_t* trees_out,
                             int n, int p, int m, int nskip, int ndpost, float tau, float lambda, float nu, float sigest,
                             unsigned seed, int keep_trees, int* stats_out, int probit, const int* ybin, float offset);
// Random numbers: install R's generator (unif_rand, norm_rand, exp_rand, rchisq) before a fit; pass nulls to use the
// built-in xoshiro128** generator (seeded by `seed`).
extern "C" void bart_set_rng(double (*u)(), double (*nrm)(), double (*e)(), double (*chisq)(double));
#endif

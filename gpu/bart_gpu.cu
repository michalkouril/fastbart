// Batched BART (weighted, as BART::wbart) on a GPU: many independent fits at once, one fit per thread block.
// Same sampler as fastbart's fwbart / the FPGA kernel: birth/death proposals, conjugate leaf draws, inverse-chi^2 sigma;
// each row keeps its leaf id per tree so one data pass per tree gives all sufficient statistics. The block's threads
// share the rows of the data pass (sums in exact 64-bit fixed point, so independent of thread order); thread 0 runs the
// serial tree control (proposal, Metropolis-Hastings, leaf draws). The MCMC runs in chunks of iterations per launch,
// with all state in global memory.
//
// Pipeline version: fits a list of exported training sets in one batch and writes each fit's posterior draws.
// Usage: bart_gpu <fit list> <threads per block>
//   fit list: one line per fit, "<prefix> <seed>"; reads <prefix>_q.txt, _yw.txt, _meta.txt (as written by gpu_export() in
//   gpu_bart.R) and writes <prefix>_sigma.txt (all nskip + ndpost draws) and <prefix>_trees.txt (kept tree draws in
//   BART's text format: "ndpost ntree p", then per draw and tree a node count and "nid v c theta" lines, pre-order).
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <random>
#include <string>
#include <vector>
#include <chrono>
#include "host_common.h"

#define NODES 32
#define MAXDEPTH 24
#define PMAX 32
#define MTREE 200
#define MAXWARP 8           // up to 256 threads per block
#define FX 4294967296.0          // fixed-point scale 2^32

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); std::exit(1); } } while (0)

struct Tree {                     // one tree, 32 node slots (360 bytes = 90 words)
  uint8_t l[NODES], r[NODES], par[NODES], v[NODES], c[NODES], d[NODES], ngv[NODES], pad[NODES];
  float th[NODES];
  uint32_t used, leaf;
};
static_assert(sizeof(Tree) % 4 == 0, "Tree must be whole words");
#define TREE_WORDS (sizeof(Tree) / 4)

struct Fit {                      // one fit's data and state (offsets into the concatenated arrays)
  int n, p, nc[PMAX];
  long long row;                  // offset of the per-row arrays (Y, W, WFX, AF, TM)
  long long q;                    // offset of Q ([p][n])
  long long leaf;                 // offset of LEAF ([MTREE][n])
  long long tree;                 // index of the first tree
  double tau, lambda, nu, sigma;  // sigma: current draw (state)
  uint32_t rng[4];                // xoshiro128** state
  int births, deaths, cap;
};

__constant__ double c_pg[64];     // alpha / (1 + depth)^beta

// ---------------------------------------------------------------- RNG (thread 0 of a block)
__device__ __forceinline__ uint32_t rotl(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }
struct Rng {
  uint32_t s0, s1, s2, s3; float spare; bool has;
  __device__ uint32_t next() { uint32_t res = rotl(s1 * 5, 7) * 9, t = s1 << 9; s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3; s2 ^= t; s3 = rotl(s3, 11); return res; }
  __device__ double unif() { return ((double)(next() >> 8) + 0.5) * (1.0 / 16777216.0); }
  __device__ double norm() {
    if (has) { has = false; return spare; }
    float u1 = (float)unif(), u2 = (float)unif(), rad = sqrtf(-2.f * __logf(u1)), a = 6.28318531f * u2;
    spare = rad * __sinf(a); has = true; return rad * __cosf(a);
  }
  __device__ double gamma_big(double a) {   // Marsaglia-Tsang, shape >= 1
    double d = a - 1.0 / 3.0, c = 1.0 / sqrt(9.0 * d);
    for (int it = 0; it < 32; it++) {
      double x = norm(), v = 1.0 + c * x;
      if (v <= 0) continue;
      v = v * v * v;
      if (log(unif()) < 0.5 * x * x + d - d * v + d * log(v)) return d * v;
    }
    return d;
  }
};

// ---------------------------------------------------------------- tree helpers (thread 0)
__device__ int select_kth(uint32_t m, int k) { for (int i = 0; i < k; i++) m &= m - 1; return __ffs(m) - 1; }
__device__ int first_free(uint32_t used, int skip) { uint32_t f = ~used & ~1u; if (skip >= 0) f &= ~(1u << skip); return f ? __ffs(f) - 1 : -1; }
__device__ void node_range(const Tree& t, int x, int v, int nc, int& L, int& U) {
  L = 0; U = nc - 1;
  for (int node = x, s = 0; node != 0 && s < MAXDEPTH; s++) {
    int pa = t.par[node];
    if (t.v[pa] == v) { if (t.l[pa] == node) U = min(U, (int)t.c[pa] - 1); else L = max(L, (int)t.c[pa] + 1); }
    node = pa;
  }
}
__device__ int count_goodvars(const Tree& t, int x, const int* nc, int p) {
  int L[PMAX], U[PMAX];
  for (int v = 0; v < p; v++) { L[v] = 0; U[v] = nc[v] - 1; }
  for (int node = x, s = 0; node != 0 && s < MAXDEPTH; s++) {
    int pa = t.par[node], v = t.v[pa], c = t.c[pa];
    if (t.l[pa] == node) U[v] = min(U[v], c - 1); else L[v] = max(L[v], c + 1);
    node = pa;
  }
  int g = 0;
  for (int v = 0; v < p; v++) g += (U[v] >= L[v]);
  return g;
}
__device__ __forceinline__ double lh(double b, double M, double t2) { double k = b * t2 + 1.0; return -0.5 * log(k) + 0.5 * M * M * t2 / k; }
__device__ __forceinline__ long long to_fx(double x) { return llrint(x * FX); }

// ---------------------------------------------------------------- the kernel: iterations [it0, it1) of every fit
__global__ void bart_iters(Fit* fits, Tree* trees, const long long* YFX, const long long* WFX, const float* W, const uint8_t* Q,
                           long long* AF, float* TM, uint8_t* LEAF, float* SIG, int sig_stride, int it0, int it1, int nskip, int m,
                           Tree* TOUT, int ndpost) {
  Fit& F = fits[blockIdx.x];
  const int tid = threadIdx.x, nth = blockDim.x, warp = tid >> 5, nwarp = (nth + 31) >> 5;
  const int n = F.n, p = F.p;
  const long long* Y = YFX + F.row; const long long* WF = WFX + F.row; const float* Wf = W + F.row;
  const uint8_t* q = Q + F.q; long long* af_ = AF + F.row; float* tm = TM + F.row; uint8_t* leafb = LEAF + F.leaf;
  Tree* tr = trees + F.tree;

  __shared__ Tree t;
  __shared__ long long th_cur[NODES], th_old[NODES], th_new[NODES];
  __shared__ unsigned long long a0[MAXWARP][NODES], a1[MAXWARP][NODES], sp[MAXWARP][4];
  __shared__ unsigned int cnt[MAXWARP][2];
  __shared__ double S0[NODES], S1[NODES], wsum[MAXWARP];
  __shared__ int cmode, cnx, cv, cc, pmode, pnx, pv, pc, pL, pR, pl, pr;
  __shared__ double s_pr;
  __shared__ Rng rng;
  __shared__ double sigma;

  if (tid == 0) { rng.s0 = F.rng[0]; rng.s1 = F.rng[1]; rng.s2 = F.rng[2]; rng.s3 = F.rng[3]; rng.has = false; sigma = F.sigma; pmode = 0; }
  __syncthreads();
  const double t2 = F.tau * F.tau, a_prior = 1.0 / t2, pb = 0.5;

  for (int it = it0; it < it1; it++) {
    const double inv_s2 = 1.0 / (sigma * sigma);
    for (int j = 0; j <= m; j++) {
      const bool last = (j == m);              // last "tree": apply tree m-1, residual sum of squares, posterior mean
      // ---- load tree j, proposal (thread 0)
      if (!last) {
        for (int w = tid; w < (int)TREE_WORDS; w += nth) ((uint32_t*)&t)[w] = ((const uint32_t*)&tr[j])[w];
        __syncthreads();
        if (tid < NODES) th_cur[tid] = to_fx(t.th[tid]);
        if (tid == 0) {
          uint32_t goodm = 0, nogm = 0;
          for (int s = 0; s < NODES; s++) {
            bool isleaf = (t.leaf >> s) & 1, isused = (t.used >> s) & 1;
            if (isused && isleaf && t.ngv[s] > 0) goodm |= 1u << s;
            if (isused && !isleaf && ((t.leaf >> t.l[s]) & 1) && ((t.leaf >> t.r[s]) & 1)) nogm |= 1u << s;
          }
          int ngood = __popc(goodm), nnogs = __popc(nogm);
          double PBx = ngood == 0 ? 0.0 : (t.used == 1u ? 1.0 : pb), pr = 0;
          int mode = 0, nx = 0, v = 0, c = 0;
          if (rng.unif() < PBx) {
            nx = select_kth(goodm, min((int)(rng.unif() * ngood), ngood - 1));
            v = min((int)(rng.unif() * p), p - 1);
            int bL, bU; node_range(t, nx, v, F.nc[v], bL, bU);
            if (bU >= bL && t.d[nx] + 1 < MAXDEPTH) {
              c = min(bL + (int)(rng.unif() * (bU - bL + 1)), bU);
              int dnx = t.d[nx];
              double PGnx = c_pg[dnx], PGly, PGry;
              if (t.ngv[nx] > 1) PGly = PGry = c_pg[dnx + 1];
              else { PGly = (c - 1 < bL) ? 0 : c_pg[dnx + 1]; PGry = (bU < c + 1) ? 0 : c_pg[dnx + 1]; }
              double PDy = (ngood > 1) ? 1 - pb : ((PGry == 0 && PGly == 0) ? 1 : 1 - pb), Pnogy;
              if (nx == 0) Pnogy = 1;
              else { int pa = t.par[nx]; bool panog = ((t.leaf >> t.l[pa]) & 1) && ((t.leaf >> t.r[pa]) & 1); Pnogy = panog ? 1.0 / nnogs : 1.0 / (nnogs + 1); }
              pr = (PGnx * (1 - PGly) * (1 - PGry) * PDy * Pnogy) / ((1 - PGnx) * (1.0 / ngood) * PBx);
              mode = 1;
            }
          } else if (nnogs > 0) {
            nx = select_kth(nogm, min((int)(rng.unif() * nnogs), nnogs - 1));
            int dny = t.d[nx], l = t.l[nx], r = t.r[nx];
            double PGny = c_pg[dny], PGlx = t.ngv[l] > 0 ? c_pg[dny + 1] : 0, PGrx = t.ngv[r] > 0 ? c_pg[dny + 1] : 0;
            double PBy = (nx == 0) ? 1 : pb;
            int ng2 = ngood - (t.ngv[l] > 0) - (t.ngv[r] > 0) + 1;
            pr = ((1 - PGny) * PBy * (1.0 / ng2)) / (PGny * (1 - PGlx) * (1 - PGrx) * (1 - PBx) * (1.0 / nnogs));
            mode = 2;
          }
          cmode = mode; cnx = nx; cv = v; cc = c; s_pr = pr;
        }
      }
      // ---- clear accumulators
      for (int k = tid; k < MAXWARP * NODES; k += nth) { a0[k / NODES][k % NODES] = 0; a1[k / NODES][k % NODES] = 0; }
      for (int k = tid; k < MAXWARP * 4; k += nth) sp[k / 4][k % 4] = 0;
      for (int k = tid; k < MAXWARP * 2; k += nth) cnt[k / 2][k % 2] = 0;
      __syncthreads();
      // ---- data pass: apply the previous tree's change; statistics of tree j (or rss / posterior mean for the last)
      const int jp = j - 1;
      const bool apply = (j > 0) && pmode >= 0;
      const bool keep = last && it >= nskip;
      double rss = 0;
      const uint8_t* qpv = q + (long long)pv * n; const uint8_t* qcv = q + (long long)cv * n;
      uint8_t* lp = leafb + (long long)jp * n; const uint8_t* lc = leafb + (long long)j * n;
      const int lane = tid & 31;
      for (int i0 = 0; i0 < n; i0 += nth) {      // whole warps iterate together (needed for the warp reductions)
        const int i = i0 + tid;
        const bool valid = i < n;
        long long af = valid ? af_[i] : 0;
        if (apply && valid) {
          int lf = lp[i], lf2 = lf;
          if (pmode == 1 && lf == pnx) lf2 = (qpv[i] <= pc) ? pL : pR;
          else if (pmode == 2 && (lf == pl || lf == pr)) lf2 = pnx;
          af = af - th_old[lf] + th_new[lf2];
          if (lf2 != lf) lp[i] = (uint8_t)lf2;
          af_[i] = af;
        }
        if (!last) {
          int lf = valid ? lc[i] : -1;
          long long w = 0, wr = 0;
          if (valid) { long long r = Y[i] - af + th_cur[lf]; w = WF[i]; wr = (long long)(Wf[i] * (float)r); }
          // warp aggregation by leaf: one pass per distinct leaf in this warp
          unsigned todo = __ballot_sync(0xffffffffu, valid);
          while (todo) {
            const int key = __shfl_sync(0xffffffffu, lf, __ffs(todo) - 1);
            const bool mine = valid && lf == key;
            long long v0 = mine ? w : 0, v1 = mine ? wr : 0;
            for (int o = 16; o > 0; o >>= 1) { v0 += __shfl_xor_sync(0xffffffffu, v0, o); v1 += __shfl_xor_sync(0xffffffffu, v1, o); }
            if (lane == 0) { a0[warp][key] += (unsigned long long)v0; a1[warp][key] += (unsigned long long)v1; }
            todo &= ~__ballot_sync(0xffffffffu, mine);
          }
          if (cmode == 1) {                         // birth proposal: split statistics of node cnx
            const bool in = valid && lf == cnx;
            const int side = in ? ((qcv[i] <= cc) ? 0 : 1) : 0;
            const bool inL = in && side == 0, inR = in && side == 1;
            long long l0 = inL ? w : 0, l1 = inL ? wr : 0, r0 = inR ? w : 0, r1 = inR ? wr : 0;
            for (int o = 16; o > 0; o >>= 1) {
              l0 += __shfl_xor_sync(0xffffffffu, l0, o); l1 += __shfl_xor_sync(0xffffffffu, l1, o);
              r0 += __shfl_xor_sync(0xffffffffu, r0, o); r1 += __shfl_xor_sync(0xffffffffu, r1, o);
            }
            const unsigned cl = __popc(__ballot_sync(0xffffffffu, inL)), cr = __popc(__ballot_sync(0xffffffffu, inR));
            if (lane == 0) { sp[warp][0] += l0; sp[warp][1] += l1; sp[warp][2] += r0; sp[warp][3] += r1; cnt[warp][0] += cl; cnt[warp][1] += cr; }
          }
        } else if (valid) {
          double e = (double)(Y[i] - af) / FX;
          rss += (double)Wf[i] * e * e;
          if (keep) tm[i] += (float)((double)af / FX);
        }
      }
      __syncthreads();
      if (last) {                                 // ---- sigma draw
        for (int o = 16; o > 0; o >>= 1) rss += __shfl_down_sync(0xffffffffu, rss, o);
        if ((tid & 31) == 0) wsum[warp] = rss;
        __syncthreads();
        if (tid == 0) {
          double s = 0;
          for (int k = 0; k < nwarp; k++) s += wsum[k];
          sigma = sqrt((F.nu * F.lambda + s) / (2.0 * rng.gamma_big((n + F.nu) / 2.0)));
          SIG[(long long)blockIdx.x * sig_stride + it] = (float)sigma;
          pmode = 0;
        }
        __syncthreads();
        if (it >= nskip) {                        // keep this draw's trees
          Tree* dst = TOUT + ((long long)blockIdx.x * ndpost + (it - nskip)) * m;
          for (long long w = tid; w < (long long)m * TREE_WORDS; w += nth) ((uint32_t*)dst)[w] = ((const uint32_t*)tr)[w];
        }
        break;
      }
      if (tid < NODES) {                         // ---- reduce the per-warp sums
        long long x0 = 0, x1 = 0;
        for (int w = 0; w < nwarp; w++) { x0 += (long long)a0[w][tid]; x1 += (long long)a1[w][tid]; }
        S0[tid] = (double)x0 / FX; S1[tid] = (double)x1 / FX;
      }
      __syncthreads();
      // ---- Metropolis-Hastings, leaf draws, store tree (thread 0)
      if (tid == 0) {
        long long sl0 = 0, sl1 = 0, sr0 = 0, sr1 = 0; int nl = 0, nr = 0;
        for (int w = 0; w < nwarp; w++) { sl0 += (long long)sp[w][0]; sl1 += (long long)sp[w][1]; sr0 += (long long)sp[w][2]; sr1 += (long long)sp[w][3]; nl += cnt[w][0]; nr += cnt[w][1]; }
        int newmode = 0, L = 0, R = 0, l = 0, r = 0;
        double lu = log(rng.unif());
        if (cmode == 1 && nl >= 5 && nr >= 5) {
          double bl = sl0 / FX * inv_s2, Ml = sl1 / FX * inv_s2, br = sr0 / FX * inv_s2, Mr = sr1 / FX * inv_s2;
          double lalpha = fmin(0.0, log(s_pr) + lh(bl, Ml, t2) + lh(br, Mr, t2) - lh(bl + br, Ml + Mr, t2));
          if (lu < lalpha) {
            L = first_free(t.used, -1); R = (L >= 0) ? first_free(t.used, L) : -1;
            if (L < 0 || R < 0) F.cap++;
            else {
              int nx = cnx;
              t.v[nx] = cv; t.c[nx] = cc; t.l[nx] = L; t.r[nx] = R;
              t.par[L] = nx; t.par[R] = nx; t.l[L] = t.r[L] = t.l[R] = t.r[R] = 0; t.d[L] = t.d[R] = t.d[nx] + 1;
              t.used |= (1u << L) | (1u << R);
              t.leaf = (t.leaf & ~(1u << nx)) | (1u << L) | (1u << R);
              t.ngv[L] = count_goodvars(t, L, F.nc, p); t.ngv[R] = count_goodvars(t, R, F.nc, p);
              S0[L] = sl0 / FX; S1[L] = sl1 / FX; S0[R] = sr0 / FX; S1[R] = sr1 / FX;
              newmode = 1; F.births++;
            }
          }
        } else if (cmode == 2) {
          int nx = cnx; l = t.l[nx]; r = t.r[nx];
          double bl = S0[l] * inv_s2, Ml = S1[l] * inv_s2, br = S0[r] * inv_s2, Mr = S1[r] * inv_s2;
          double lalpha = fmin(0.0, log(s_pr) + lh(bl + br, Ml + Mr, t2) - lh(bl, Ml, t2) - lh(br, Mr, t2));
          if (lu < lalpha) {
            S0[nx] = S0[l] + S0[r]; S1[nx] = S1[l] + S1[r];
            t.used &= ~((1u << l) | (1u << r));
            t.leaf = (t.leaf & ~((1u << l) | (1u << r))) | (1u << nx);
            t.l[nx] = 0; t.r[nx] = 0; t.v[nx] = 0; t.c[nx] = 0;
            newmode = 2; F.deaths++;
          }
        }
        for (uint32_t lm = t.leaf & t.used; lm; lm &= lm - 1) {     // draw every leaf's mu
          int s = __ffs(lm) - 1;
          double b = S0[s] * inv_s2, M = S1[s] * inv_s2;
          t.th[s] = (float)(M / (a_prior + b) + rng.norm() / sqrt(a_prior + b));
        }
        pmode = newmode; pnx = cnx; pv = cv; pc = cc; pL = L; pR = R; pl = l; pr = r;
      }
      __syncthreads();
      if (tid < NODES) { th_old[tid] = th_cur[tid]; th_new[tid] = to_fx(t.th[tid]); }
      for (int w = tid; w < (int)TREE_WORDS; w += nth) ((uint32_t*)&tr[j])[w] = ((const uint32_t*)&t)[w];
      __syncthreads();
    }
  }
  if (tid == 0) { F.rng[0] = rng.s0; F.rng[1] = rng.s1; F.rng[2] = rng.s2; F.rng[3] = rng.s3; F.sigma = sigma; }
}

// ---------------------------------------------------------------- host
#include <fstream>
#include <sstream>
static void write_trees(const std::string& file, const Tree* T, int ndpost, int m, int p) {
  std::ostringstream os; os.precision(10);
  os << ndpost << " " << m << " " << p << "\n";
  for (int dr = 0; dr < ndpost; dr++)
    for (int j = 0; j < m; j++) {
      const Tree& t = T[(size_t)dr * m + j];
      std::vector<std::pair<int, long long>> st{{0, 1}}, order;            // pre-order; node id: root 1, children 2id, 2id+1
      while (!st.empty()) { auto e = st.back(); st.pop_back(); order.push_back(e); int s = e.first;
        if (!((t.leaf >> s) & 1)) { st.push_back({t.r[s], 2 * e.second + 1}); st.push_back({t.l[s], 2 * e.second}); } }
      os << order.size() << "\n";
      for (auto& e : order) { int s = e.first; bool leaf = (t.leaf >> s) & 1;
        os << e.second << " " << (leaf ? 0 : (int)t.v[s]) << " " << (leaf ? 0 : (int)t.c[s]) << " " << (leaf ? (double)t.th[s] : 0.0) << "\n"; }
    }
  std::ofstream(file) << os.str();
}

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: %s <fit list> <threads per block>\n", argv[0]); return 1; }
  const int nth = std::atoi(argv[2]), m = MTREE, chunk = 25;
  if (nth > 32 * MAXWARP || nth % 32) { std::fprintf(stderr, "threads per block: multiple of 32, <= %d\n", 32 * MAXWARP); return 1; }
  auto T0 = std::chrono::steady_clock::now();
  std::vector<std::string> prefixes; std::vector<uint32_t> seeds;
  { std::ifstream in(argv[1]); std::string pre; uint32_t sd; while (in >> pre >> sd) { prefixes.push_back(pre); seeds.push_back(sd); } }
  const int nf = (int)prefixes.size();
  std::vector<Model> md(nf);
  for (int f = 0; f < nf; f++) md[f] = load_model(prefixes[f]);
  const int nskip = md[0].nskip, ndpost = md[0].ndpost, total = nskip + ndpost;
  if (nf == 0) { std::fprintf(stderr, "no fits in %s\n", argv[1]); return 1; }
  for (int f = 0; f < nf; f++) {
    const Model& d = md[f];
    if (d.nskip != nskip || d.ndpost != ndpost || d.p < 1 || d.p > PMAX || d.n <= 0 || (int)d.numcut.size() != d.p ||
        (int)d.y.size() != d.n || (int)d.wt.size() != d.n || d.q.size() != (size_t)d.n * d.p) {
      std::fprintf(stderr, "bad input for fit %s (n %d, p %d, values: y %zu, w %zu, numcut %zu)\n", prefixes[f].c_str(), d.n, d.p, d.y.size(), d.wt.size(), d.numcut.size());
      return 1;
    }
  }

  std::vector<Fit> fits(nf);
  std::vector<long long> yfx, wfx; std::vector<float> wf; std::vector<uint8_t> qv;
  long long rows = 0, qoff = 0, loff = 0, toff = 0;
  for (int f = 0; f < nf; f++) {
    const Model& d = md[f]; Fit& F = fits[f]; std::memset(&F, 0, sizeof F);
    F.n = d.n; F.p = d.p; for (int v = 0; v < PMAX; v++) F.nc[v] = v < d.p ? d.numcut[v] : 0;
    F.row = rows; F.q = qoff; F.leaf = loff; F.tree = toff;
    F.tau = d.tau; F.lambda = d.lambda; F.nu = d.nu; F.sigma = d.sigest;
    uint32_t seed = seeds[f];
    F.rng[0] = seed ^ 0x9E3779B9u; F.rng[1] = seed * 2654435761u + 1u; F.rng[2] = 0x6A09E667u ^ (seed << 7); F.rng[3] = 0xBB67AE85u + seed;
    for (int k = 0; k < 16; k++) { uint32_t* s = F.rng; uint32_t t = s[1] << 9; s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = (s[3] << 11) | (s[3] >> 21); }
    for (int i = 0; i < d.n; i++) { yfx.push_back(llrint((double)d.y[i] * FX)); wfx.push_back(llrint((double)d.wt[i] * FX)); wf.push_back(d.wt[i]); }
    qv.insert(qv.end(), d.q.begin(), d.q.end());
    rows += d.n; qoff += (long long)d.p * d.n; loff += (long long)m * d.n; toff += m;
  }
  std::vector<Tree> trees(toff);
  for (int f = 0; f < nf; f++) {
    int ngv = 0; for (int v = 0; v < fits[f].p; v++) ngv += fits[f].nc[v] >= 1;
    for (int j = 0; j < m; j++) { Tree& T = trees[fits[f].tree + j]; std::memset(&T, 0, sizeof T); T.ngv[0] = ngv; T.used = 1; T.leaf = 1; }
  }
  std::vector<double> pg(64); for (int d = 0; d < 64; d++) pg[d] = 0.95 / std::pow(1.0 + d, 2.0);
  CK(cudaMemcpyToSymbol(c_pg, pg.data(), sizeof(double) * 64));
  const long long nout = (long long)nf * ndpost * m;
  Fit* dF; Tree *dT, *dTO; long long *dY, *dWF, *dAF; float *dW, *dTM, *dSIG; uint8_t *dQ, *dL;
  CK(cudaMalloc(&dF, sizeof(Fit) * nf)); CK(cudaMalloc(&dT, sizeof(Tree) * toff)); CK(cudaMalloc(&dTO, sizeof(Tree) * nout));
  CK(cudaMalloc(&dY, 8 * rows)); CK(cudaMalloc(&dWF, 8 * rows)); CK(cudaMalloc(&dAF, 8 * rows));
  CK(cudaMalloc(&dW, 4 * rows)); CK(cudaMalloc(&dTM, 4 * rows)); CK(cudaMalloc(&dSIG, 4LL * total * nf));
  CK(cudaMalloc(&dQ, qoff)); CK(cudaMalloc(&dL, loff));
  CK(cudaMemcpy(dF, fits.data(), sizeof(Fit) * nf, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dT, trees.data(), sizeof(Tree) * toff, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dY, yfx.data(), 8 * rows, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dWF, wfx.data(), 8 * rows, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dW, wf.data(), 4 * rows, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dQ, qv.data(), qoff, cudaMemcpyHostToDevice));
  CK(cudaMemset(dAF, 0, 8 * rows)); CK(cudaMemset(dTM, 0, 4 * rows)); CK(cudaMemset(dL, 0, loff));
  auto T1 = std::chrono::steady_clock::now();
  for (int it = 0; it < total; it += chunk) {
    bart_iters<<<nf, nth>>>(dF, dT, dY, dWF, dW, dQ, dAF, dTM, dL, dSIG, total, it, std::min(it + chunk, total), nskip, m, dTO, ndpost);
    CK(cudaGetLastError());
  }
  CK(cudaDeviceSynchronize());
  auto T2 = std::chrono::steady_clock::now();
  std::vector<Fit> out(nf); std::vector<float> sig((size_t)total * nf); std::vector<Tree> tout(nout);
  CK(cudaMemcpy(out.data(), dF, sizeof(Fit) * nf, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(sig.data(), dSIG, 4LL * total * nf, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(tout.data(), dTO, sizeof(Tree) * nout, cudaMemcpyDeviceToHost));
  long long cap = 0;
  #pragma omp parallel for schedule(dynamic) reduction(+:cap)
  for (int f = 0; f < nf; f++) {
    std::ofstream so(prefixes[f] + "_sigma.txt"); so.precision(10);
    for (int i = 0; i < total; i++) so << sig[(size_t)f * total + i] << "\n";
    write_trees(prefixes[f] + "_trees.txt", &tout[(size_t)f * ndpost * m], ndpost, m, md[f].p);
    cap += out[f].cap;
  }
  auto T3 = std::chrono::steady_clock::now();
  auto sec = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
  std::printf("bart_gpu: %d fits | load %.1f s | GPU %.1f s (%.1f fits/s) | write %.1f s | node cap hits %lld\n",
              nf, sec(T0, T1), sec(T1, T2), nf / sec(T1, T2), sec(T2, T3), cap);
  return 0;
}

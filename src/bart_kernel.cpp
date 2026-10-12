#include "bart_kernel.h"
#include <cmath>
#if defined(__SYNTHESIS__)
#include <hls_math.h>
#define LOGF(x)  hls::logf(x)
#define SQRTF(x) hls::sqrtf(x)
#define COSF(x)  hls::cosf(x)
#define LOGD(x)  hls::log(x)
#define SQRTD(x) hls::sqrt(x)
#else
#define LOGF(x)  logf(x)
#define SQRTF(x) sqrtf(x)
#define COSF(x)  cosf(x)
#define LOGD(x)  log(x)
#define SQRTD(x) sqrt(x)
#endif
#ifdef FAST_CPU
#define BIT(m, i) (((m) >> (i)) & 1ull)
#else
#define BIT(m, i) ((m)[i])
#endif

// ---------------------------------------------------------------- RNG (xoshiro128**, Box-Muller, Marsaglia-Tsang)
struct Rng { uint32_t s0, s1, s2, s3; };
static inline uint32_t rotl(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }
static inline uint32_t rng_next(Rng& r) {
  uint32_t res = rotl(r.s1 * 5, 7) * 9, t = r.s1 << 9;
  r.s2 ^= r.s0; r.s3 ^= r.s1; r.s1 ^= r.s2; r.s0 ^= r.s3; r.s2 ^= t; r.s3 = rotl(r.s3, 11);
  return res;
}
static inline float unif(Rng& r) { return ((float)(rng_next(r) >> 8) + 0.5f) * (1.0f / 16777216.0f); }   // (0,1)
static inline float rnorm(Rng& r) { float u1 = unif(r), u2 = unif(r); return SQRTF(-2.f * LOGF(u1)) * COSF(6.28318531f * u2); }
static double rgamma_big(Rng& r, double a) {        // shape a >= 1, rate 1
  double d = a - 1.0 / 3.0, c = 1.0 / SQRTD(9.0 * d);
  for (int it = 0; it < 32; it++) {
    double x = rnorm(r), v = 1.0 + c * x;
    if (v <= 0) continue;
    v = v * v * v;
    double u = unif(r);
    if (LOGD(u) < 0.5 * x * x + d - d * v + d * LOGD(v)) return d * v;
  }
  return d;
}

// ---------------------------------------------------------------- mask helpers (64 node slots)
static int popc(mask_t m) {
  int c = 0;
  for (int i = 0; i < NODES; i++) {
#pragma HLS UNROLL
    c += (int)BIT(m, i);
  }
  return c;
}
static int select_kth(mask_t m, int k) {          // index of the k-th (0-based) set bit
  int c = 0, r = 0;
  for (int i = 0; i < NODES; i++) {
#pragma HLS UNROLL
    if (BIT(m, i)) { if (c == k) r = i; c++; }
  }
  return r;
}
static int first_free(mask_t used, int skip) {    // lowest unused slot other than `skip`
  int r = -1;
  for (int i = NODES - 1; i >= 1; i--) {
#pragma HLS UNROLL
    if (!BIT(used, i) && i != skip) r = i;
  }
  return r;
}

// ---------------------------------------------------------------- on-chip state
struct Tree {
  u8 l[NODES], r[NODES], par[NODES], v[NODES], c[NODES], d[NODES], ngv[NODES];
  float th[NODES];
  mask_t used, leaf;
};
static u8 T_l[MMAX][NODES], T_r[MMAX][NODES], T_p[MMAX][NODES], T_v[MMAX][NODES], T_c[MMAX][NODES], T_d[MMAX][NODES], T_g[MMAX][NODES];
static float T_th[MMAX][NODES];
static mask_t T_used[MMAX], T_leaf[MMAX];

static u8 LE[KL][(MMAX / 2) * NL];   // leaf id of each row, even trees
static u8 LO[KL][(MMAX / 2) * NL];   // odd trees
static u8 Q[KL][PMAX * NL];          // cutpoint ranks
static fx_t Y[KL][NL], AF[KL][NL], TM[KL][NL];
static wt_t W[KL][NL];

struct PrevUpd { int mode, nx, v, c, L, R, l, r; fx_t thOld[NODES], thNew[NODES]; };   // mode 0 none, 1 birth, 2 death
struct CurProp { int mode, nx, v, c; fx_t th[NODES]; };
struct Stats { acc_t S0[NODES], S1[NODES], sL0, sL1, sR0, sR1, rss; int nL, nR; };

// ---------------------------------------------------------------- the data pass: apply tree jp's update, sufficient statistics of tree j
template <int PAR>   // parity of tree j; tree jp has the other parity
static void pass_t(int jp, int j, const PrevUpd& pu, bool apply, const CurProp& cp, bool stats, bool rss, bool keep, int n, Stats& st) {
  acc_t a0[KL][NODES], a1[KL][NODES], bl0[KL], bl1[KL], br0[KL], br1[KL], ra[KL];
  int nl[KL], nr[KL];
#pragma HLS ARRAY_PARTITION variable=a0 complete dim=0
#pragma HLS ARRAY_PARTITION variable=a1 complete dim=0
#pragma HLS ARRAY_PARTITION variable=bl0 complete
#pragma HLS ARRAY_PARTITION variable=bl1 complete
#pragma HLS ARRAY_PARTITION variable=br0 complete
#pragma HLS ARRAY_PARTITION variable=br1 complete
#pragma HLS ARRAY_PARTITION variable=ra complete
#pragma HLS ARRAY_PARTITION variable=nl complete
#pragma HLS ARRAY_PARTITION variable=nr complete
  for (int k = 0; k < KL; k++) {
#pragma HLS UNROLL
    for (int s = 0; s < NODES; s++) { a0[k][s] = 0; a1[k][s] = 0; }
    bl0[k] = bl1[k] = br0[k] = br1[k] = ra[k] = 0; nl[k] = nr[k] = 0;
  }
  fx_t tho[NODES], thn[NODES], thc[NODES];     // register copies: every lane reads them every cycle
#pragma HLS ARRAY_PARTITION variable=tho complete
#pragma HLS ARRAY_PARTITION variable=thn complete
#pragma HLS ARRAY_PARTITION variable=thc complete
  for (int s = 0; s < NODES; s++) {
#pragma HLS PIPELINE II=1
    tho[s] = pu.thOld[s]; thn[s] = pu.thNew[s]; thc[s] = cp.th[s];
  }
  const int pmode = pu.mode, pnx = pu.nx, pc = pu.c, pL = pu.L, pR = pu.R, pl = pu.l, pr = pu.r;
  const int cmode = cp.mode, cnx = cp.nx, cc = cp.c;
  const int nper = (n + KL - 1) / KL;
  const int pbase = (jp >> 1) * NL, cbase = (j >> 1) * NL;
  const int pvb = pu.v * NL, cvb = cp.v * NL;
ROWS:
  for (int ii = 0; ii < nper; ii++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=200 max=512
#pragma HLS DEPENDENCE variable=LE inter false
#pragma HLS DEPENDENCE variable=LO inter false
#pragma HLS DEPENDENCE variable=AF inter false
#pragma HLS DEPENDENCE variable=TM inter false
    for (int k = 0; k < KL; k++) {
#pragma HLS UNROLL
      bool valid = ii * KL + k < n;
      fx_t af = AF[k][ii];
      if (apply) {
        u8 lf = PAR == 0 ? LO[k][pbase + ii] : LE[k][pbase + ii];
        u8 lf2 = lf;
        if (pmode == 1 && lf == pnx) lf2 = (Q[k][pvb + ii] <= pc) ? pL : pR;
        else if (pmode == 2 && (lf == pl || lf == pr)) lf2 = pnx;
        af = af - tho[lf] + thn[lf2];
        if (PAR == 0) LO[k][pbase + ii] = lf2; else LE[k][pbase + ii] = lf2;
        AF[k][ii] = af;
      }
      if (rss) {
        fx_t e = Y[k][ii] - af;
        ra[k] += (acc_t)(W[k][ii] * e) * e;
        if (keep) TM[k][ii] += af;
      }
      if (stats) {
        u8 lf = PAR == 0 ? LE[k][cbase + ii] : LO[k][cbase + ii];
        fx_t r = Y[k][ii] - af + thc[lf];
        wt_t w = W[k][ii];
        acc_t wr = w * r;
        a0[k][lf] += w; a1[k][lf] += wr;
        if (cmode == 1 && lf == cnx && valid) {
          if (Q[k][cvb + ii] <= cc) { nl[k]++; bl0[k] += w; bl1[k] += wr; }
          else                        { nr[k]++; br0[k] += w; br1[k] += wr; }
        }
      }
    }
  }
  for (int s = 0; s < NODES; s++) {
#pragma HLS PIPELINE II=1
    acc_t x0 = 0, x1 = 0;
    for (int k = 0; k < KL; k++) { x0 += a0[k][s]; x1 += a1[k][s]; }
    st.S0[s] = x0; st.S1[s] = x1;
  }
  st.sL0 = st.sL1 = st.sR0 = st.sR1 = st.rss = 0; st.nL = st.nR = 0;
  for (int k = 0; k < KL; k++) {
#pragma HLS UNROLL
    st.sL0 += bl0[k]; st.sL1 += bl1[k]; st.sR0 += br0[k]; st.sR1 += br1[k]; st.rss += ra[k]; st.nL += nl[k]; st.nR += nr[k];
  }
}

static void do_pass(int jp, int j, const PrevUpd& pu, bool apply, const CurProp& cp, bool stats, bool rss, bool keep, int n, Stats& st) {
  if (j & 1) pass_t<1>(jp, j, pu, apply, cp, stats, rss, keep, n, st);
  else       pass_t<0>(jp, j, pu, apply, cp, stats, rss, keep, n, st);
}

// ---------------------------------------------------------------- tree helpers
static void load_tree(int j, Tree& t) {
  for (int s = 0; s < NODES; s++) {
#pragma HLS UNROLL
    t.l[s] = T_l[j][s]; t.r[s] = T_r[j][s]; t.par[s] = T_p[j][s]; t.v[s] = T_v[j][s]; t.c[s] = T_c[j][s];
    t.d[s] = T_d[j][s]; t.ngv[s] = T_g[j][s]; t.th[s] = T_th[j][s];
  }
  t.used = T_used[j]; t.leaf = T_leaf[j];
}
static void store_tree(int j, const Tree& t) {
  for (int s = 0; s < NODES; s++) {
#pragma HLS UNROLL
    T_l[j][s] = t.l[s]; T_r[j][s] = t.r[s]; T_p[j][s] = t.par[s]; T_v[j][s] = t.v[s]; T_c[j][s] = t.c[s];
    T_d[j][s] = t.d[s]; T_g[j][s] = t.ngv[s]; T_th[j][s] = t.th[s];
  }
  T_used[j] = t.used; T_leaf[j] = t.leaf;
}
// available cutpoint range [L,U] of variable v at node x (BART's tree::rg)
static void node_range(const Tree& t, int x, int v, int nc, int& L, int& U) {
  L = 0; U = nc - 1;
  int node = x;
  for (int s = 0; s < MAXDEPTH; s++) {
#pragma HLS PIPELINE II=1
    if (node == 0) break;
    int pa = t.par[node];
    if ((int)t.v[pa] == v) {
      if ((int)t.l[pa] == node) { if ((int)t.c[pa] - 1 < U) U = (int)t.c[pa] - 1; }
      else                      { if ((int)t.c[pa] + 1 > L) L = (int)t.c[pa] + 1; }
    }
    node = pa;
  }
}
// number of variables with a non-empty cutpoint range at node x (getgoodvars().size())
static int count_goodvars(const Tree& t, int x, const int nc[PMAX], int p) {
  int L[PMAX], U[PMAX];
#pragma HLS ARRAY_PARTITION variable=L complete
#pragma HLS ARRAY_PARTITION variable=U complete
  for (int v = 0; v < PMAX; v++) {
#pragma HLS UNROLL
    L[v] = 0; U[v] = nc[v] - 1;
  }
  int node = x;
  for (int s = 0; s < MAXDEPTH; s++) {
#pragma HLS PIPELINE II=1
    if (node == 0) break;
    int pa = t.par[node], v = t.v[pa], c = t.c[pa];
    if ((int)t.l[pa] == node) { if (c - 1 < U[v]) U[v] = c - 1; }
    else                      { if (c + 1 > L[v]) L[v] = c + 1; }
    node = pa;
  }
  int g = 0;
  for (int v = 0; v < PMAX; v++) {
#pragma HLS UNROLL
    if (v < p && U[v] >= L[v]) g++;
  }
  return g;
}

static inline float lh(float b, float M, float t2) { float k = b * t2 + 1.f; return -0.5f * LOGF(k) + 0.5f * M * M * t2 / k; }

#ifdef FAST_CPU
// truncated normal N(mean, 1) restricted to > tau (BART's rtnorm)
static double rtnorm(double mean, double tau, Rng& r) {
  double z;
  tau = tau - mean;
  if (tau <= 0.) {
    do { z = rnorm(r); } while (z < tau);
  } else {
    double lambda = 0.5 * (tau + sqrt(tau * tau + 4.0));
    do { z = -log((double)unif(r)) / lambda + tau; } while ((double)unif(r) > exp(-0.5 * (z - lambda) * (z - lambda)));
  }
  return z + mean;
}
#endif

// ---------------------------------------------------------------- top
#ifdef FAST_CPU
extern "C" void bart_wfit(const uint8_t* qin, const float* yin, const float* wtin, const int* numcut, const float* pgtab,
                          float* sigma_out, float* trmean_out, uint64_t* trees_out,
                          int n, int p, int m, int nskip, int ndpost, float tau, float lambda, float nu, float sigest,
                          unsigned seed, int keep_trees, int* stats_out) {
  bart_fit_cpu(qin, yin, wtin, numcut, pgtab, sigma_out, trmean_out, trees_out, n, p, m, nskip, ndpost, tau, lambda, nu, sigest,
               seed, keep_trees, stats_out, 0, nullptr, 0.f);
}
extern "C" void bart_fit_cpu(const uint8_t* qin, const float* yin, const float* wtin, const int* numcut, const float* pgtab,
                             float* sigma_out, float* trmean_out, uint64_t* trees_out,
                             int n, int p, int m, int nskip, int ndpost, float tau, float lambda, float nu, float sigest,
                             unsigned seed, int keep_trees, int* stats_out, int probit, const int* ybin, float offset) {
#else
extern "C" void bart_wfit(const uint8_t* qin, const float* yin, const float* wtin, const int* numcut, const float* pgtab,
                          float* sigma_out, float* trmean_out, uint64_t* trees_out,
                          int n, int p, int m, int nskip, int ndpost, float tau, float lambda, float nu, float sigest,
                          unsigned seed, int keep_trees, int* stats_out) {
#endif
#pragma HLS INTERFACE m_axi port=qin offset=slave bundle=gmem0 depth=262144
#pragma HLS INTERFACE m_axi port=yin offset=slave bundle=gmem0 depth=8192
#pragma HLS INTERFACE m_axi port=wtin offset=slave bundle=gmem0 depth=8192
#pragma HLS INTERFACE m_axi port=numcut offset=slave bundle=gmem0 depth=32
#pragma HLS INTERFACE m_axi port=pgtab offset=slave bundle=gmem0 depth=64
#pragma HLS INTERFACE m_axi port=sigma_out offset=slave bundle=gmem1 depth=4096
#pragma HLS INTERFACE m_axi port=trmean_out offset=slave bundle=gmem1 depth=8192
#pragma HLS INTERFACE m_axi port=trees_out offset=slave bundle=gmem1 depth=1280000
#pragma HLS INTERFACE m_axi port=stats_out offset=slave bundle=gmem1 depth=8
#pragma HLS ARRAY_PARTITION variable=T_l complete dim=2
#pragma HLS ARRAY_PARTITION variable=T_r complete dim=2
#pragma HLS ARRAY_PARTITION variable=T_p complete dim=2
#pragma HLS ARRAY_PARTITION variable=T_v complete dim=2
#pragma HLS ARRAY_PARTITION variable=T_c complete dim=2
#pragma HLS ARRAY_PARTITION variable=T_d complete dim=2
#pragma HLS ARRAY_PARTITION variable=T_g complete dim=2
#pragma HLS ARRAY_PARTITION variable=T_th complete dim=2
#pragma HLS ARRAY_PARTITION variable=LE complete dim=1
#pragma HLS ARRAY_PARTITION variable=LO complete dim=1
#pragma HLS ARRAY_PARTITION variable=Q complete dim=1
#pragma HLS ARRAY_PARTITION variable=Y complete dim=1
#pragma HLS ARRAY_PARTITION variable=AF complete dim=1
#pragma HLS ARRAY_PARTITION variable=TM complete dim=1
#pragma HLS ARRAY_PARTITION variable=W complete dim=1
#pragma HLS BIND_STORAGE variable=LE type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=LO type=ram_2p impl=uram

  int nc[PMAX];
  float PG[NODES];
#pragma HLS ARRAY_PARTITION variable=nc complete
#pragma HLS ARRAY_PARTITION variable=PG complete
  for (int v = 0; v < PMAX; v++) nc[v] = v < p ? numcut[v] : 0;
  for (int d = 0; d < NODES; d++) PG[d] = pgtab[d];

  // ---- load data into lane banks
  const int nper = (n + KL - 1) / KL;
LOAD_ROWS:
  for (int i = 0; i < nper * KL; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT max=8192
    int k = i % KL, ii = i / KL;
    bool ok = i < n;
    Y[k][ii] = ok ? (fx_t)yin[i] : (fx_t)0;
    W[k][ii] = ok ? (wt_t)wtin[i] : (wt_t)0;
    AF[k][ii] = 0; TM[k][ii] = 0;
  }
LOAD_Q:
  for (int v = 0; v < p; v++)
    for (int i = 0; i < nper * KL; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT max=8192
      Q[i % KL][v * NL + i / KL] = i < n ? (u8)qin[v * n + i] : (u8)0;
    }
CLEAR_LEAF:
  for (int a = 0; a < (MMAX / 2) * NL; a++) {
#pragma HLS PIPELINE II=1
    for (int k = 0; k < KL; k++) { LE[k][a] = 0; LO[k][a] = 0; }
  }
  int root_ngv = 0;
  for (int v = 0; v < PMAX; v++) if (v < p && nc[v] >= 1) root_ngv++;
INIT_TREES:
  for (int j = 0; j < m; j++) {
#pragma HLS PIPELINE II=1
    for (int s = 0; s < NODES; s++) { T_l[j][s] = 0; T_r[j][s] = 0; T_p[j][s] = 0; T_v[j][s] = 0; T_c[j][s] = 0; T_d[j][s] = 0; T_g[j][s] = 0; T_th[j][s] = 0; }
    T_g[j][0] = root_ngv; T_used[j] = 1; T_leaf[j] = 1;
  }

  Rng rng; rng.s0 = seed ^ 0x9E3779B9u; rng.s1 = seed * 2654435761u + 1u; rng.s2 = 0x6A09E667u ^ (seed << 7); rng.s3 = 0xBB67AE85u + seed;
  for (int i = 0; i < 16; i++) rng_next(rng);
#ifdef FAST_CPU
  if (probit)    // latent z (pbart: f excludes binaryOffset)
    for (int i = 0; i < n; i++) Y[i % KL][i / KL] = ybin[i] == 0 ? -rtnorm(0., offset, rng) : rtnorm(0., -offset, rng);
#endif

  const float t2 = tau * tau, a_prior = 1.f / t2, pb = 0.5f;
  double sigma = sigest;
  int n_birth = 0, n_death = 0, n_cap = 0;
  PrevUpd pu; pu.mode = 0; pu.nx = pu.v = pu.c = pu.L = pu.R = pu.l = pu.r = 0;
  Stats st;
  const int total = nskip + ndpost;

ITER:
  for (int it = 0; it < total; it++) {
#pragma HLS LOOP_TRIPCOUNT min=1100 max=1100
    const float inv_s2 = (float)(1.0 / (sigma * sigma));
  TREES:
    for (int j = 0; j < m; j++) {
#pragma HLS LOOP_TRIPCOUNT min=200 max=200
      Tree t;
#pragma HLS ARRAY_PARTITION variable=t.l complete
#pragma HLS ARRAY_PARTITION variable=t.r complete
#pragma HLS ARRAY_PARTITION variable=t.par complete
#pragma HLS ARRAY_PARTITION variable=t.v complete
#pragma HLS ARRAY_PARTITION variable=t.c complete
#pragma HLS ARRAY_PARTITION variable=t.d complete
#pragma HLS ARRAY_PARTITION variable=t.ngv complete
#pragma HLS ARRAY_PARTITION variable=t.th complete
      load_tree(j, t);

      // ---- proposal (depends on the tree only)
      mask_t goodm = 0, nogm = 0;
      for (int s = 0; s < NODES; s++) {
#pragma HLS UNROLL
        bool isleaf = BIT(t.leaf, s), isused = BIT(t.used, s);
        if (isused && isleaf && t.ngv[s] > 0) goodm |= ((mask_t)1 << s);
        if (isused && !isleaf && BIT(t.leaf, (int)t.l[s]) && BIT(t.leaf, (int)t.r[s])) nogm |= ((mask_t)1 << s);
      }
      int ngood = popc(goodm), nnogs = popc(nogm);
      bool single = (t.used == (mask_t)1);
      float PBx = ngood == 0 ? 0.f : (single ? 1.f : pb);
      CurProp cp; cp.mode = 0; cp.nx = 0; cp.v = 0; cp.c = 0;
      for (int s = 0; s < NODES; s++) {
#pragma HLS UNROLL
        cp.th[s] = (fx_t)t.th[s];
      }
      float pr = 0.f;
      int bL = 0, bU = 0;
      bool birth = unif(rng) < PBx;
      if (birth) {
        int nx = select_kth(goodm, (int)(unif(rng) * ngood));
        int v = (int)(unif(rng) * p); if (v >= p) v = p - 1;
        node_range(t, nx, v, nc[v], bL, bU);
        if (bU >= bL && (int)t.d[nx] + 1 < MAXDEPTH) {        // a "bad" variable gives an empty child: always rejected in BART
          int c = bL + (int)(unif(rng) * (bU - bL + 1)); if (c > bU) c = bU;
          int dnx = t.d[nx];
          float PGnx = PG[dnx], PGly, PGry;
          if (t.ngv[nx] > 1) { PGly = PGry = PG[dnx + 1]; }
          else { PGly = (c - 1 < bL) ? 0.f : PG[dnx + 1]; PGry = (bU < c + 1) ? 0.f : PG[dnx + 1]; }
          float PDy = (ngood > 1) ? 1.f - pb : ((PGry == 0.f && PGly == 0.f) ? 1.f : 1.f - pb);
          float Pnogy;
          if (nx == 0) Pnogy = 1.f;
          else {
            int pa = t.par[nx];
            bool panog = BIT(t.leaf, (int)t.l[pa]) && BIT(t.leaf, (int)t.r[pa]);
            Pnogy = panog ? 1.f / nnogs : 1.f / (nnogs + 1.f);
          }
          float Pbotx = 1.f / ngood;
          pr = (PGnx * (1.f - PGly) * (1.f - PGry) * PDy * Pnogy) / ((1.f - PGnx) * Pbotx * PBx);
          cp.mode = 1; cp.nx = nx; cp.v = v; cp.c = c;
        }
      } else {
        int nx = select_kth(nogm, (int)(unif(rng) * nnogs));
        int dny = t.d[nx], l = t.l[nx], r = t.r[nx];
        float PGny = PG[dny];
        float PGlx = t.ngv[l] > 0 ? PG[dny + 1] : 0.f, PGrx = t.ngv[r] > 0 ? PG[dny + 1] : 0.f;
        float PBy = (nx == 0) ? 1.f : pb;
        int ng2 = ngood - (t.ngv[l] > 0) - (t.ngv[r] > 0) + 1;
        float Pboty = 1.f / ng2, PDx = 1.f - PBx, Pnogx = 1.f / nnogs;
        pr = ((1.f - PGny) * PBy * Pboty) / (PGny * (1.f - PGlx) * (1.f - PGrx) * PDx * Pnogx);
        cp.mode = 2; cp.nx = nx;
      }

      // ---- data pass: apply previous tree's change, sufficient statistics for this tree
      do_pass(j - 1, j, pu, j > 0, cp, true, false, false, n, st);

      // ---- Metropolis-Hastings
      PrevUpd nu_; nu_.mode = 0; nu_.nx = cp.nx; nu_.v = cp.v; nu_.c = cp.c; nu_.L = 0; nu_.R = 0; nu_.l = 0; nu_.r = 0;
      for (int s = 0; s < NODES; s++) {
#pragma HLS UNROLL
        nu_.thOld[s] = cp.th[s];
      }
      acc_t S0[NODES], S1[NODES];
#pragma HLS ARRAY_PARTITION variable=S0 complete
#pragma HLS ARRAY_PARTITION variable=S1 complete
      for (int s = 0; s < NODES; s++) {
#pragma HLS UNROLL
        S0[s] = st.S0[s]; S1[s] = st.S1[s];
      }
      float lu = LOGF(unif(rng));
      if (cp.mode == 1) {
        if (st.nL >= 5 && st.nR >= 5) {
          float bl = (float)st.sL0 * inv_s2, Ml = (float)st.sL1 * inv_s2, br = (float)st.sR0 * inv_s2, Mr = (float)st.sR1 * inv_s2;
          float lalpha = LOGF(pr) + lh(bl, Ml, t2) + lh(br, Mr, t2) - lh(bl + br, Ml + Mr, t2);
          if (lalpha > 0.f) lalpha = 0.f;
          if (lu < lalpha) {
            int L = first_free(t.used, -1), R = first_free(t.used, L);
            if (L < 0 || R < 0) n_cap++;
            else {
              int nx = cp.nx;
              t.v[nx] = cp.v; t.c[nx] = cp.c; t.l[nx] = L; t.r[nx] = R;
              t.par[L] = nx; t.par[R] = nx; t.l[L] = t.r[L] = t.l[R] = t.r[R] = 0;
              t.d[L] = t.d[R] = t.d[nx] + 1;
              t.used |= ((mask_t)1 << L) | ((mask_t)1 << R);
              t.leaf = (t.leaf & ~((mask_t)1 << nx)) | ((mask_t)1 << L) | ((mask_t)1 << R);
              t.ngv[L] = count_goodvars(t, L, nc, p); t.ngv[R] = count_goodvars(t, R, nc, p);
              S0[L] = st.sL0; S1[L] = st.sL1; S0[R] = st.sR0; S1[R] = st.sR1;
              nu_.mode = 1; nu_.L = L; nu_.R = R;
              n_birth++;
            }
          }
        }
      } else if (cp.mode == 2) {
        int nx = cp.nx, l = t.l[nx], r = t.r[nx];
        float bl = (float)S0[l] * inv_s2, Ml = (float)S1[l] * inv_s2, br = (float)S0[r] * inv_s2, Mr = (float)S1[r] * inv_s2;
        float lalpha = LOGF(pr) + lh(bl + br, Ml + Mr, t2) - lh(bl, Ml, t2) - lh(br, Mr, t2);
        if (lalpha > 0.f) lalpha = 0.f;
        if (lu < lalpha) {
          S0[nx] = S0[l] + S0[r]; S1[nx] = S1[l] + S1[r];
          t.used &= ~(((mask_t)1 << l) | ((mask_t)1 << r));
          t.leaf = (t.leaf & ~(((mask_t)1 << l) | ((mask_t)1 << r))) | ((mask_t)1 << nx);
          t.l[nx] = 0; t.r[nx] = 0; t.v[nx] = 0; t.c[nx] = 0;
          nu_.mode = 2; nu_.l = l; nu_.r = r;
          n_death++;
        }
      }
      // ---- draw every leaf mu (drmu)
    DRMU:
      for (int s = 0; s < NODES; s++) {
#pragma HLS PIPELINE II=1
        float z = rnorm(rng);
        if (BIT(t.leaf, s) && BIT(t.used, s)) {
          float b = (float)S0[s] * inv_s2, M = (float)S1[s] * inv_s2;
          t.th[s] = M / (a_prior + b) + z / SQRTF(a_prior + b);
        }
      }
      for (int s = 0; s < NODES; s++) {
#pragma HLS UNROLL
        nu_.thNew[s] = (fx_t)t.th[s];
      }
      store_tree(j, t);
      pu = nu_;
    }
    // ---- apply the last tree, residual sum of squares, posterior-mean accumulation
    bool keep = it >= nskip;
    CurProp dummy; dummy.mode = 0; dummy.nx = 0; dummy.v = 0; dummy.c = 0;
    for (int s = 0; s < NODES; s++) dummy.th[s] = 0;
    do_pass(m - 1, m, pu, true, dummy, false, true, keep, n, st);
    pu.mode = 0;
#ifdef FAST_CPU
    if (probit) {    // sigma stays 1; redraw the latent z given the current fit
      for (int i = 0; i < n; i++) {
        double f = AF[i % KL][i / KL];
        Y[i % KL][i / KL] = ybin[i] == 0 ? -rtnorm(-f, offset, rng) : rtnorm(f, -offset, rng);
      }
    } else
#endif
    {
      double rss = (double)st.rss;
      sigma = SQRTD((nu * lambda + rss) / (2.0 * rgamma_big(rng, (n + nu) / 2.0)));
    }
    sigma_out[it] = (float)sigma;
    if (keep && keep_trees) {
      int base = (it - nskip) * m * NODES;
    OUT_TREES:
      for (int j = 0; j < m; j++)
        for (int s = 0; s < NODES; s++) {
#pragma HLS PIPELINE II=1
          bool used = BIT(T_used[j], s);
          union { float f; uint32_t u; } th; th.f = T_th[j][s];
          uint64_t w = (uint64_t)(used ? (unsigned)T_v[j][s] : 255u) | ((uint64_t)(unsigned)T_c[j][s] << 8) |
                       ((uint64_t)(unsigned)T_l[j][s] << 16) | ((uint64_t)(unsigned)T_r[j][s] << 24) | ((uint64_t)th.u << 32);
          trees_out[base + j * NODES + s] = w;
        }
    }
  }
OUT_MEAN:
  for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT max=8192
    trmean_out[i] = (float)TM[i % KL][i / KL] / (float)ndpost;
  }
  stats_out[0] = n_birth; stats_out[1] = n_death; stats_out[2] = n_cap;
}

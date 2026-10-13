// Shared host-side loading and checking for the BART kernel (software build and XRT build).
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct Model {
  int n = 0, p = 0, ndpost = 0, nskip = 0;
  double fmean = 0, tau = 0, lambda = 0, sigest = 0, nu = 3;
  std::vector<int> numcut;
  std::vector<uint8_t> q;          // [p][n]
  std::vector<float> y, wt;        // centred y, 1/iw^2
  std::vector<double> ref_yhat, ref_sigma;
};

static std::vector<double> read_col(const std::string& f, int col, int ncol) {
  std::ifstream in(f); std::vector<double> out; double x;
  for (int k = 0; in >> x; k++) if (k % ncol == col) out.push_back(x);
  return out;
}

static Model load_model(const std::string& pre) {
  Model md;
  std::ifstream meta(pre + "_meta.txt"); std::string line;
  while (std::getline(meta, line)) {
    std::istringstream ss(line); std::string key; ss >> key;
    if (key == "n") ss >> md.n; else if (key == "p") ss >> md.p; else if (key == "fmean") ss >> md.fmean;
    else if (key == "tau") ss >> md.tau; else if (key == "lambda") ss >> md.lambda; else if (key == "sigest") ss >> md.sigest;
    else if (key == "nu") ss >> md.nu; else if (key == "ndpost") ss >> md.ndpost; else if (key == "nskip") ss >> md.nskip;
    else if (key == "numcut") { int c; while (ss >> c) md.numcut.push_back(c); }
  }
  std::vector<double> qq = read_col(pre + "_q.txt", 0, 1);    // row-major n x p
  md.q.resize((size_t)md.p * md.n);
  for (int i = 0; i < md.n; i++) for (int v = 0; v < md.p; v++) md.q[(size_t)v * md.n + i] = (uint8_t)qq[(size_t)i * md.p + v];
  std::vector<double> y = read_col(pre + "_yw.txt", 0, 2), iw = read_col(pre + "_yw.txt", 1, 2);
  for (int i = 0; i < md.n; i++) { md.y.push_back((float)y[i]); md.wt.push_back((float)(1.0 / (iw[i] * iw[i]))); }
  md.ref_yhat = read_col(pre + "_ref_yhat.txt", 0, 1);
  md.ref_sigma = read_col(pre + "_ref_sigma.txt", 0, 1);
  return md;
}

static std::vector<float> pg_table() {
  std::vector<float> pg(64);
  for (int d = 0; d < 64; d++) pg[d] = (float)(0.95 / std::pow(1.0 + d, 2.0));
  return pg;
}

// predictions of the kept tree draws on the training rows, averaged (checks the tree dump against trmean)
static std::vector<double> predict_mean(const Model& md, const std::vector<uint64_t>& trees, int m, int nodes) {
  std::vector<double> f(md.n, 0.0);
  for (int dr = 0; dr < md.ndpost; dr++)
    for (int j = 0; j < m; j++) {
      const uint64_t* T = &trees[((size_t)dr * m + j) * nodes];
      for (int i = 0; i < md.n; i++) {
        int s = 0;
        for (;;) {
          uint64_t w = T[s]; int l = (w >> 16) & 255, r = (w >> 24) & 255, v = w & 255, c = (w >> 8) & 255;
          if (l == 0) { uint32_t u = (uint32_t)(w >> 32); float th; std::memcpy(&th, &u, 4); f[i] += th; break; }
          s = md.q[(size_t)v * md.n + i] <= c ? l : r;
        }
      }
    }
  for (auto& x : f) x /= md.ndpost;
  return f;
}

static void report(const char* tag, const Model& md, const std::vector<float>& sig, const std::vector<float>& trmean,
                   const std::vector<double>& fromtrees, double secs, const int* st) {
  double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, d2 = 0, dt = 0;
  for (int i = 0; i < md.n; i++) {
    double a = trmean[i], b = md.ref_yhat[i];
    sx += a; sy += b; sxx += a * a; syy += b * b; sxy += a * b; d2 += (a - b) * (a - b);
    if (!fromtrees.empty()) dt = std::max(dt, std::fabs(fromtrees[i] - a));
  }
  int n = md.n;
  double cor = (sxy - sx * sy / n) / std::sqrt((sxx - sx * sx / n) * (syy - sy * sy / n));
  double ms = 0, mr = 0;
  for (int i = md.nskip; i < md.nskip + md.ndpost; i++) { ms += sig[i]; mr += md.ref_sigma[i]; }
  ms /= md.ndpost; mr /= md.ndpost;
  std::printf("%s n=%d p=%d  time %.3f s  | cor(yhat, BART::wbart yhat)=%.5f  rmsdiff=%.4f  sd(yhat)=%.3f | sigma %.4f vs ref %.4f"
              " | births %d deaths %d cap %d | tree-dump max|diff| %.2e\n",
              tag, md.n, md.p, secs, cor, std::sqrt(d2 / n), std::sqrt((syy - sy * sy / n) / n), ms, mr, st[0], st[1], st[2], dt);
}

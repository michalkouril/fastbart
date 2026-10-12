#include <Rcpp.h>
#include <sstream>
#include <vector>
#include "bart_kernel.h"

// [[Rcpp::export]]
Rcpp::List fwbart_cpp(Rcpp::IntegerMatrix q, Rcpp::NumericVector y, Rcpp::NumericVector wt, Rcpp::IntegerVector numcut,
                      Rcpp::NumericVector pgtab, int ntree, int nskip, int ndpost, double tau, double lambda, double nu,
                      double sigest, unsigned int seed) {
  const int n = q.nrow(), p = q.ncol(), m = ntree;
  std::vector<uint8_t> qv((size_t)n * p);
  for (int v = 0; v < p; v++) for (int i = 0; i < n; i++) qv[(size_t)v * n + i] = (uint8_t)q(i, v);
  std::vector<float> yf(y.begin(), y.end()), wf(wt.begin(), wt.end()), pg(pgtab.begin(), pgtab.end());
  std::vector<int> nc(numcut.begin(), numcut.end());
  std::vector<float> sig(nskip + ndpost), trm(n);
  std::vector<uint64_t> trees((size_t)ndpost * m * NODES);
  int st[8] = {0};
  bart_wfit(qv.data(), yf.data(), wf.data(), nc.data(), pg.data(), sig.data(), trm.data(), trees.data(), n, p, m, nskip, ndpost,
            (float)tau, (float)lambda, (float)nu, (float)sigest, seed, 1, st);

  // per-draw training fits (wbart's yhat.train), split counts (varcount) and BART's tree text format
  Rcpp::NumericMatrix yhat(ndpost, n);
  Rcpp::IntegerMatrix varcount(ndpost, p);
  std::ostringstream ts; ts.precision(10);
  ts << ndpost << " " << m << " " << p << std::endl;
  for (int dr = 0; dr < ndpost; dr++)
    for (int j = 0; j < m; j++) {
      const uint64_t* T = &trees[((size_t)dr * m + j) * NODES];
      auto L = [&](int s) { return (int)((T[s] >> 16) & 255); };
      auto R = [&](int s) { return (int)((T[s] >> 24) & 255); };
      auto TH = [&](int s) { uint32_t u = (uint32_t)(T[s] >> 32); float f; std::memcpy(&f, &u, 4); return (double)f; };
      std::vector<std::pair<int, size_t>> stack{{0, 1}}, order;            // pre-order, node id: root 1, children 2id, 2id+1
      while (!stack.empty()) {
        auto [s, id] = stack.back(); stack.pop_back(); order.push_back({s, id});
        if (L(s)) { stack.push_back({R(s), 2 * id + 1}); stack.push_back({L(s), 2 * id}); }
      }
      ts << order.size() << std::endl;
      for (auto& [s, id] : order) {
        bool leaf = L(s) == 0;
        int v = leaf ? 0 : (int)(T[s] & 255), c = leaf ? 0 : (int)((T[s] >> 8) & 255);
        if (!leaf) varcount(dr, v) += 1;
        ts << id << " " << v << " " << c << " " << (leaf ? TH(s) : 0.0) << std::endl;
      }
      for (int i = 0; i < n; i++) {
        int s = 0;
        while (L(s)) s = qv[(size_t)(T[s] & 255) * n + i] <= (int)((T[s] >> 8) & 255) ? L(s) : R(s);
        yhat(dr, i) += TH(s);
      }
    }
  return Rcpp::List::create(Rcpp::Named("sigma") = Rcpp::NumericVector(sig.begin(), sig.end()),
                            Rcpp::Named("yhat.train.mean") = Rcpp::NumericVector(trm.begin(), trm.end()),
                            Rcpp::Named("yhat.train") = yhat, Rcpp::Named("varcount") = varcount,
                            Rcpp::Named("trees") = ts.str(), Rcpp::Named("moves") = Rcpp::IntegerVector(st, st + 3));
}

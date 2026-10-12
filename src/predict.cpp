// Fast single-draw prediction from BART tree draws (wbart / pbart / fwbart / fpbart fits).
// compile_trees_cpp parses the treedraws text once into flat node arrays, each node's two children stored next to each other
// so a step is branch-free: next = child + (x >= cut). predict_draw_cpp evaluates one posterior draw on the columns of newdata
// with BART's rule (left if x[v] < cutpoints[v][c]) and BART's summation order (trees 1..ntree per row), so its results equal
// pwbart()[m, ] exactly.
#include <Rcpp.h>
#include <sstream>
#include <unordered_map>
#include <vector>

// [[Rcpp::export]]
Rcpp::List compile_trees_cpp(std::string trees, Rcpp::List cutpoints) {
  std::istringstream in(trees);
  int nd, m, p;
  in >> nd >> m >> p;
  std::vector<std::vector<double>> xi(cutpoints.size());
  for (int v = 0; v < cutpoints.size(); v++) xi[v] = Rcpp::as<std::vector<double>>(cutpoints[v]);
  std::vector<int> var, child, root(nd * m);
  std::vector<double> cut, theta;
  for (int t = 0; t < nd * m; t++) {
    int nn; in >> nn;
    std::vector<size_t> nid(nn); std::vector<int> v(nn), c(nn); std::vector<double> th(nn);
    std::unordered_map<size_t, int> pos;
    for (int k = 0; k < nn; k++) { in >> nid[k] >> v[k] >> c[k] >> th[k]; pos[nid[k]] = k; }
    // breadth-first from the root (node id 1), placing each node's children consecutively
    std::vector<int> q{pos[1]}, slot{(int)var.size()};
    root[t] = slot[0];
    var.push_back(0); child.push_back(-1); cut.push_back(0); theta.push_back(0);
    for (size_t h = 0; h < q.size(); h++) {
      int k = q[h], me = slot[h];
      auto L = pos.find(2 * nid[k]);
      theta[me] = th[k];
      if (L != pos.end()) {
        int ch = (int)var.size();
        var[me] = v[k]; cut[me] = xi[v[k]][c[k]]; child[me] = ch;
        for (int z = 0; z < 2; z++) { var.push_back(0); child.push_back(-1); cut.push_back(0); theta.push_back(0); }
        q.push_back(L->second); slot.push_back(ch);
        q.push_back(pos[2 * nid[k] + 1]); slot.push_back(ch + 1);
      }
    }
  }
  return Rcpp::List::create(Rcpp::Named("ndpost") = nd, Rcpp::Named("ntree") = m, Rcpp::Named("p") = p,
                            Rcpp::Named("var") = var, Rcpp::Named("child") = child, Rcpp::Named("cut") = cut,
                            Rcpp::Named("theta") = theta, Rcpp::Named("root") = root);
}

// newdata: list of numeric columns (already restricted to the model's columns); draw m is 1-based
// [[Rcpp::export]]
Rcpp::NumericVector predict_draw_cpp(Rcpp::List ct, Rcpp::List newdata, int m, double mu) {
  const int ntree = Rcpp::as<int>(ct["ntree"]), p = Rcpp::as<int>(ct["p"]), nd = Rcpp::as<int>(ct["ndpost"]);
  if (newdata.size() != p) Rcpp::stop("newdata has %d columns, the trees use %d", (int)newdata.size(), p);
  if (m < 1 || m > nd) Rcpp::stop("draw %d out of range 1..%d", m, nd);
  const int* var = INTEGER(ct["var"]); const int* child = INTEGER(ct["child"]); const int* root = INTEGER(ct["root"]);
  const double* cut = REAL(ct["cut"]); const double* theta = REAL(ct["theta"]);
  std::vector<Rcpp::NumericVector> keep(p);
  std::vector<const double*> col(p);
  for (int v = 0; v < p; v++) { keep[v] = Rcpp::as<Rcpp::NumericVector>(newdata[v]); col[v] = keep[v].begin(); }
  const int n = p == 0 ? 0 : keep[0].size();
  Rcpp::NumericVector out(n);
  double* f = out.begin();
  const int* r0 = root + (size_t)(m - 1) * ntree;
  // tree-outer, row-inner: independent rows overlap in the CPU; each row still adds trees in order 1..ntree
  for (int t = 0; t < ntree; t++) {
    const int k0 = r0[t];
    for (int i = 0; i < n; i++) {
      int k = k0;
      while (child[k] >= 0) k = child[k] + (int)(col[var[k]][i] >= cut[k]);
      f[i] += theta[k];
    }
  }
  for (int i = 0; i < n; i++) f[i] += mu;
  return out;
}

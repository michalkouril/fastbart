// Runs the kernel as ordinary C++ (FAST_CPU: double arithmetic; otherwise the bit-accurate HLS fixed-point types).
// Usage: host_sw <data prefix>... e.g. data/titer1 data/titer2
#include "bart_kernel.h"
#include "host_common.h"

int main(int argc, char** argv) {
  const int m = 200, NOD = NODES;
  for (int a = 1; a < argc; a++) {
    Model md = load_model(argv[a]);
    std::vector<float> pg = pg_table(), sig(md.nskip + md.ndpost), trm(md.n);
    std::vector<uint64_t> trees((size_t)md.ndpost * m * NOD);
    int st[8] = {0};
    auto t0 = std::chrono::steady_clock::now();
    bart_wfit(md.q.data(), md.y.data(), md.wt.data(), md.numcut.data(), pg.data(), sig.data(), trm.data(), trees.data(), md.n, md.p, m,
              md.nskip, md.ndpost, (float)md.tau, (float)md.lambda, (float)md.nu, (float)md.sigest, 12345u + a, 1, st);
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    report(argv[a], md, sig, trm, predict_mean(md, trees, m, NOD), secs, st);
  }
}

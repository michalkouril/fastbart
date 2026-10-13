// Runs bart_wfit on the Alveo card through XRT and checks the result like host_sw.
// Usage: host_xrt <xclbin> <data prefix>...      (build with -DNODES=<the kernel's NODES>)
#include "host_common.h"
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#ifndef NODES
#define NODES 64
#endif

int main(int argc, char** argv) {
  const int m = 200, NOD = NODES;
  xrt::device dev(0);
  auto uuid = dev.load_xclbin(argv[1]);
  xrt::kernel krn(dev, uuid, "bart_wfit");
  std::printf("xclbin loaded\n"); std::fflush(stdout);
  for (int a = 2; a < argc; a++) {
    Model md = load_model(argv[a]);
    std::vector<float> pg = pg_table(), sig(md.nskip + md.ndpost), trm(md.n);
    std::vector<uint64_t> trees((size_t)md.ndpost * m * NOD);
    int st[8] = {0};
    auto t0 = std::chrono::steady_clock::now();
    auto mk = [&](size_t bytes, int arg) { return xrt::bo(dev, bytes, krn.group_id(arg)); };
    xrt::bo bq = mk(md.q.size(), 0), by = mk(md.n * 4, 1), bw = mk(md.n * 4, 2), bnc = mk(md.p * 4, 3), bpg = mk(64 * 4, 4),
            bsig = mk(sig.size() * 4, 5), btrm = mk(md.n * 4, 6), btr = mk(trees.size() * 8, 7), bst = mk(8 * 4, 19);
    bq.write(md.q.data()); by.write(md.y.data()); bw.write(md.wt.data()); bnc.write(md.numcut.data()); bpg.write(pg.data());
    for (auto* b : {&bq, &by, &bw, &bnc, &bpg}) b->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto t1 = std::chrono::steady_clock::now();
    auto run = krn(bq, by, bw, bnc, bpg, bsig, btrm, btr, md.n, md.p, m, md.nskip, md.ndpost, (float)md.tau, (float)md.lambda,
                   (float)md.nu, (float)md.sigest, 12345u + (a - 1), 1, bst);   // same seeds as host_sw: 12345 + model position
    run.wait();
    auto t2 = std::chrono::steady_clock::now();
    for (auto* b : {&bsig, &btrm, &btr, &bst}) b->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bsig.read(sig.data()); btrm.read(trm.data()); btr.read(trees.data()); bst.read(st);
    auto t3 = std::chrono::steady_clock::now();
    double kern = std::chrono::duration<double>(t2 - t1).count(), tot = std::chrono::duration<double>(t3 - t0).count();
    std::printf("[kernel %.3f s, with transfers %.3f s] ", kern, tot);
    report(argv[a], md, sig, trm, predict_mean(md, trees, m, NOD), kern, st);
    std::fflush(stdout);
  }
}

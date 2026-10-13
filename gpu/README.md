# fastbart on a GPU (CUDA)

A single BART fit is a long chain of small, dependent steps (220,000 tree updates of a few thousand rows each), which a
GPU cannot speed up. A bootstrap, however, has hundreds of independent fits, and those a GPU can run at once:
`bart_gpu` fits many models in one batch, **one fit per thread block**. Within a block the threads share the rows of each
tree's data pass (sums in exact 64-bit fixed point, combined per leaf with warp shuffles before one add per warp), and
one thread runs the serial tree control (proposal, Metropolis-Hastings, leaf draws). Same sampler as `fwbart` and the
FPGA kernel; its own xoshiro128** generator, so results are not bit-identical to `fwbart` (statistically equivalent).
Results and benchmarks are in the main README ("GPU").

## Files

| file | what |
|---|---|
| `bart_batch.cu` | throughput benchmark: replicates each exported titer model (bootstrap row resamples) and fits them all; checks replicate 0 against `BART::wbart` |
| `bart_gpu.cu` | batch fitter for the pipeline: reads a list of exported fits, writes each fit's sigma draws and tree draws (BART's text format) |
| `gpu_bart.R` | `gpu_export()` (writes a fit's input, prepared exactly as `fwbart` does) and `gpu_import()` (reads a fit back as a `"wbart"` object) |
| `bootstrap_worker_gpu.R` | one replicate, one phase (`prep` or `finish`) |
| `make_pipeline.py` | builds the GPU pipeline from your own copy of the bootstrap package (see below) |
| `Makefile` | builds `bart_batch` and `bart_gpu` (CUDA; set `ARCH` for your GPU) |

## Throughput benchmark

```sh
make bart_batch
Rscript ../fpga/export_titer_data.R path/to/BART_gformula_week6_mean.R path/to/data/seed_01 data
./bart_batch data 101 256 1,2,3,4,5,6       # 101 replicates x 6 titer models = 606 fits, 256 threads per fit
```

## GPU bootstrap pipeline

The pipeline is generated from your copy of the bootstrap package (its driver and estimator are not part of this
repository):

```sh
make bart_gpu
python3 make_pipeline.py path/to/bootstrap_bart_package path/to/BART_gformula_week6_mean_fastbart.R
cd path/to/bootstrap_bart_package
BART_GPU=/path/to/bart_gpu NDPOST=100 NSKIP=1000 FRESH=1 Rscript run_bootstrap_gpu.R 100 10 1
```

Three phases: (1) **prep**, on the CPU in parallel: each replicate's resample, CBPS, the hazard model (`fpbart`) and the
six titer training sets, exported with `gpu_export()`, plus the R session; (2) **GPU**: all titer fits of all replicates in
one `bart_gpu` batch; (3) **finish**, on the CPU in parallel: each replicate reloads its session, reads its six fits with
`gpu_import()` and runs the rest of the estimator unchanged. The outputs (`bootstrap_bart_week6_<tag>_replicates.csv`,
`_summary.csv`) are the same as from `run_bootstrap_bart_week6.R`.

Memory: about 1 MB per fit plus the kept tree draws (7 MB per fit at ndpost = 100); 606 fits need about 5 GB of GPU memory.

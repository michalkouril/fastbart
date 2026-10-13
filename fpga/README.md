# fastbart on an FPGA (AMD/Xilinx Alveo U200)

FPGA build of the same sampler as `fwbart` (`../src/bart_kernel.cpp`, the file the R package compiles with `-DFAST_CPU`).
Results and the build history are in the main README ("FPGA proof of concept"): one kernel at 100 MHz fits a
4,600-row, 200-tree weighted BART model (1,100 iterations) in about 2 s, 2.4x faster than one CPU core running `fwbart`.

## Files

| file | what |
|---|---|
| `../src/bart_kernel.cpp`, `../src/bart_kernel.h` | the kernel `bart_wfit` (HLS pragmas; fixed-point data pass, float control, on-chip RNG) |
| `host_xrt.cpp` | runs the kernel on the card through XRT and checks the result |
| `host_sw.cpp` | runs the same kernel source as ordinary C++ with the HLS fixed-point types: a bit-accurate simulation |
| `host_common.h` | loading of the exported data, comparison with `BART::wbart`, tree-dump check |
| `export_titer_data.R` | writes the input files from a BART g-formula pipeline (see below) |
| `Makefile` | simulation, HLS, link, host build, run |

FPGA settings (passed as defines; the CPU defaults in the header differ): `NODES=32` node slots per tree, `KL=8` rows
per clock cycle. Limits: 8,192 rows, 32 predictors, 200 trees.

## Requirements

Alveo U200 with the `xilinx_u200_gen3x16_xdma_2_202110_1` platform, XRT (tested 2.17), Vitis 2023.2 (Vitis 2026.1 no
longer supports this platform). For the data export: R with BART and CBPS.

## Use

```sh
# 1. input files (six titer models) from the pipeline's estimator and data
Rscript export_titer_data.R path/to/BART_gformula_week6_mean.R path/to/data/seed_01 data

# 2. bit-accurate simulation (minutes; no FPGA needed)
make sim

# 3. hardware: HLS (~6 min) and link / place and route (~1.5 h at 100 MHz), then run on the card
source /opt/Xilinx/Vitis/2023.2/settings64.sh; source /opt/xilinx/xrt/setup.sh
make                # bart_wfit.xclbin and host_xrt
make run
```

`make sim` and `make run` use the same seeds and should print identical numbers (fitted values, sigma, numbers of
accepted birth and death moves): the hardware is bit-identical to the simulation. Each line also reports the
correlation with a reference `BART::wbart` fit (from the export) and the posterior mean of sigma for both.

`FREQ=200 make` links at 200 MHz; on the U200 that did not close timing (the kernel spans two SLRs), see the main README.

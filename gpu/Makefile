# GPU (CUDA) builds of the fastbart sampler: many independent BART fits at once, one fit per thread block.
NVCC ?= $(or $(shell command -v nvcc 2>/dev/null),/usr/local/cuda/bin/nvcc)
ARCH ?= sm_75                 # RTX 20xx (Turing); e.g. sm_86 for RTX 30xx, sm_89 for RTX 40xx
FLAGS := -O3 -arch=$(ARCH) -std=c++17 -I../fpga -diag-suppress 177

all: bart_batch bart_gpu

# throughput benchmark on exported titer models (see ../fpga/export_titer_data.R)
bart_batch: bart_batch.cu ../fpga/host_common.h
	$(NVCC) $(FLAGS) -o $@ bart_batch.cu
# batch fitter used by the GPU bootstrap pipeline (make_pipeline.py)
bart_gpu: bart_gpu.cu ../fpga/host_common.h
	$(NVCC) $(FLAGS) -Xcompiler -fopenmp -o $@ bart_gpu.cu

.PHONY: all

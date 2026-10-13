# FPGA build of the fastbart sampler (src/bart_kernel.cpp) for an AMD/Xilinx Alveo U200.
# Needs Vitis 2023.2 (source settings64.sh), XRT (source /opt/xilinx/xrt/setup.sh) and the U200 platform below.
PLATFORM ?= /opt/xilinx/platforms/xilinx_u200_gen3x16_xdma_2_202110_1/xilinx_u200_gen3x16_xdma_2_202110_1.xpfm
FREQ     ?= 100                       # kernel clock in MHz used for linking (200 MHz did not close timing)
DEFS     := -DNODES=32 -DKL=8          # FPGA settings: 32 node slots per tree, 8 rows per clock
SRC      := ../src/bart_kernel.cpp
HLS_INC  ?= /opt/Xilinx/Vitis_HLS/2023.2/include
XRT      ?= /opt/xilinx/xrt
DATA     ?= data
MODELS   := $(foreach i,1 2 3 4 5 6,$(DATA)/titer$(i))

all: host_xrt bart_wfit.xclbin

# bit-accurate C++ simulation of the kernel (HLS fixed-point types)
host_sim: host_sw.cpp host_common.h $(SRC)
	g++ -O2 $(DEFS) -I$(HLS_INC) -I../src -o $@ host_sw.cpp $(SRC)
sim: host_sim
	./host_sim $(MODELS)

# HLS (minutes) and link / place and route (1.5-2 h at 100 MHz)
bart_wfit.xo: $(SRC)
	v++ -c -t hw --platform $(PLATFORM) --kernel_frequency 200 -k bart_wfit -D NODES=32 -D KL=8 -I../src -o $@ $(SRC)
bart_wfit.xclbin: bart_wfit.xo
	v++ -l -t hw --platform $(PLATFORM) --kernel_frequency $(FREQ) --vivado.synth.jobs 8 --vivado.impl.jobs 4 -o $@ $<

host_xrt: host_xrt.cpp host_common.h
	g++ -O2 -std=c++17 $(DEFS) -I$(XRT)/include -o $@ host_xrt.cpp -L$(XRT)/lib -lxrt_coreutil -pthread
run: host_xrt bart_wfit.xclbin
	./host_xrt bart_wfit.xclbin $(MODELS)

.PHONY: all sim run

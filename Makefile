# =============================================================================
# Makefile for mr_blackwell - Native Miller-Rabin GPU Kernel
# Author: Camillo / pscamillo
#
# Usage:
#   make                    # Build for Blackwell (default)
#   make ARCH=sm_86         # Build for Ampere (RTX 3090)
#   make ARCH=sm_89         # Build for Ada Lovelace (RTX 4090)
#   make ARCH=sm_120        # Build for Blackwell (RTX 5070/5080/5090)
#   make test               # Quick correctness test (1024 candidates)
#   make bench              # Standard benchmark (65536 candidates, 683-bit)
#   make bench-1240         # Benchmark with 1240-bit numbers (P=907)
#   make clean              # Clean build artifacts
# =============================================================================

# Default architecture: Blackwell
ARCH ?= sm_120

# Compiler settings
NVCC     = nvcc
CFLAGS   = -O3 -arch=$(ARCH)
LDFLAGS  = -lgmp

# Source files
STANDALONE = mr_blackwell.cu

# Output
TARGET = mr_blackwell_test

# =============================================================================
# Build targets
# =============================================================================

all: $(TARGET)

$(TARGET): $(STANDALONE)
	$(NVCC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo ""
	@echo "Built for $(ARCH). Run 'make test' to verify."

# =============================================================================
# Test and benchmark targets
# =============================================================================

test: $(TARGET)
	@echo "=== Correctness Test (1024 candidates, 683-bit) ==="
	./$(TARGET) 1024 683
	@echo ""
	@echo "=== Correctness Test (1024 candidates, 1240-bit) ==="
	./$(TARGET) 1024 1240

bench: $(TARGET)
	@echo "=== Benchmark (65536 candidates, 683-bit) ==="
	./$(TARGET) 65536 683

bench-1240: $(TARGET)
	@echo "=== Benchmark (16384 candidates, 1240-bit) ==="
	./$(TARGET) 16384 1240

bench-all: $(TARGET)
	@echo "=== 512-bit ==="
	./$(TARGET) 65536 512
	@echo ""
	@echo "=== 683-bit (P=503) ==="
	./$(TARGET) 65536 683
	@echo ""
	@echo "=== 1024-bit ==="
	./$(TARGET) 65536 1024
	@echo ""
	@echo "=== 1240-bit (P=907) ==="
	./$(TARGET) 16384 1240

clean:
	rm -f $(TARGET) mr_blackwell_v2 mr_blackwell_v2b mr_blackwell_v2c mr_blackwell_v2d

# =============================================================================
# Info target
# =============================================================================

info:
	@echo "mr_blackwell - Native Miller-Rabin GPU Kernel"
	@echo ""
	@echo "Architecture: $(ARCH)"
	@echo "Common architectures:"
	@echo "  sm_86   - Ampere (RTX 3090, A100)"
	@echo "  sm_89   - Ada Lovelace (RTX 4090)"
	@echo "  sm_90   - Hopper (H100)"
	@echo "  sm_120  - Blackwell (RTX 5070/5080/5090)"
	@echo ""
	@echo "Usage: make ARCH=sm_XX"

.PHONY: all test bench bench-1240 bench-all clean info

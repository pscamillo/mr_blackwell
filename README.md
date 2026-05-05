# mr_blackwell — Native Miller-Rabin GPU Kernel for NVIDIA Blackwell

**A high-performance Miller-Rabin primality testing kernel for NVIDIA GPUs, built from scratch as a drop-in replacement for CGBN.**

## Why?

[CGBN](https://github.com/NVlabs/CGBN) (CUDA Generic Big Number library by NVlabs) is the standard library for big-number arithmetic on NVIDIA GPUs. However, **CGBN has severe performance issues on Blackwell architecture (SM 12.0)** — kernels compile and launch without errors, but run orders of magnitude slower than expected, making it impractical for real workloads. The CGBN Makefile only lists architectures through sm_80 (Ampere), and as of April 2026, no official Blackwell support has been announced.

This was discovered in April 2026 during integration with [sethtroisi/prime-gap](https://github.com/sethtroisi/prime-gap) for prime gap searches. No prior reports of this issue existed online.

**mr_blackwell** solves this by implementing Montgomery multiplication from scratch using:
- Montgomery CIOS (Coarsely Integrated Operand Scanning)
- PTX inline assembly carry chains (`mad.lo.cc` / `madc.hi` / `add.cc` / `addc`)
- Fixed-window modular exponentiation (4-bit, CGBN-style)
- Montgomery-space comparisons (eliminates `from_mont` in hot loops)
- Template-based LIMBS dispatch for optimal sizing

## Performance

Tested on RTX 5070 (SM 12.0 / Blackwell), CUDA 12.9, Linux Mint:

### Standalone benchmark (random candidates, base-2 MR)

| Version | Bits | LIMBS | PRP/sec | vs CPU | Notes |
|---------|------|-------|---------|--------|-------|
| CPU (GMP) | 683 | — | 13,700 | 1x | Ryzen 9 8800X3D |
| CGBN (Blackwell) | 683 | — | impractical | — | Orders of magnitude slower |
| v1 (binary exp) | 683 | 32 | 537,000 | 39x | Baseline native kernel |
| v2b (correct LIMBS) | 683 | 22 | 1,033,000 | 75x | LIMBS matched to bits |
| v2c (windowed) | 683 | 22 | 1,526,000 | 111x | 4-bit window exp |
| **v2d (PTX)** | **683** | **22** | **1,805,000** | **132x** | **PTX carry chains** |
| v2d | 1240 | 40 | 295,000 | 21x | P=907 prime gaps |

### Integrated with prime-gap pipeline

| Configuration | Time (P=503, 13K tests) | Kernel PRP/sec |
|---------------|------------------------|----------------|
| CPU (gap_test_simple) | 28.0s | 13,700 |
| GPU BITS=1024 | 9.1s | 605,000 |
| GPU v1 (BITS=704 + Mont compare) | 4.5s | 1,340,000 |
| **GPU v2 (constants on device)** | **3.1s** | **~1,800,000** |

### Results

Using this kernel integrated with sethtroisi/prime-gap, **47 world records** have been set as of April 25, 2026, all from a single RTX 5070 running the parallel sieve+GPU pipeline at P=907.

#### Production results (as of 2026-04-25)

- **Total records:** 47
- **Best so far:** gap 27894, merit 31.7283 (mstart = 398,360,389)

Distribution of 2026 records in the official `gaps.db` by merit band:

| Merit band | Camillo records | All 2026 records |
|------------|-----------------|------------------|
| 24–25      | 1               | 5                |
| 25–26      | 4               | 6                |
| 26–27      | 9               | 14               |
| 27–28      | 14              | 14               |
| 28–29      | 8               | 8                |
| 29–30      | 8               | 8                |
| 30–31      | 2               | 2                |
| 31–32      | 1               | 1                |

#### Initial records (April 18, 2026)

The first 5 records, set within 24 hours of the kernel's first use:

| Gap | Merit | Previous record | Previous holder |
|-----|-------|-----------------|-----------------|
| 25542 | 29.12 | 27.11 | S.Troisi |
| 24568 | 28.04 | 27.71 | Loizides |
| 23840 | 27.18 | 26.89 | Loizides |
| 23566 | 26.87 | 26.56 | Gapcoin |
| 22858 | 26.08 | 25.15 | Loizides |

Records verified and accepted at [primegaps.cloudygo.com](https://primegaps.cloudygo.com) (under "Camillo") and committed to [primegap-list-project](https://github.com/primegap-list-project/prime-gap-list).

Discussion thread: [Native Miller-Rabin GPU kernel for Blackwell — mersenneforum.org](https://www.mersenneforum.org/node/1111135)

## Architecture

```
┌─────────────────────────────────────────┐
│  gap_test_gpu.cu (sethtroisi/prime-gap) │
│  #include "miller_rabin_native.h"       │
├─────────────────────────────────────────┤
│  miller_rabin_native.h                  │
│  ├── test_runner_t  (same interface)    │
│  ├── mont_mul()     (PTX carry chains)  │
│  ├── mont_sqr()     (calls mont_mul)    │
│  ├── mont_powm_windowed() (4-bit)       │
│  └── kernel_miller_rabin_native()       │
├─────────────────────────────────────────┤
│  PTX primitives: mad.lo.cc, madc.hi,    │
│  add.cc, addc (hardware carry chains)   │
└─────────────────────────────────────────┘
```

## Quick Start

### Standalone benchmark

```bash
# Build (adjust -arch for your GPU)
make ARCH=sm_120    # Blackwell (RTX 5070/5080/5090)
# make ARCH=sm_89   # Ada Lovelace (RTX 4090)
# make ARCH=sm_86   # Ampere (RTX 3090)

# Test correctness (verifies against GMP)
make test

# Benchmark
make bench
```

### Integration with prime-gap

```bash
cd /path/to/prime-gap

# 1. Copy the header
cp /path/to/mr_blackwell/miller_rabin_native.h .

# 2. Change the include in gap_test_gpu.cu line 41
sed -i 's/#include "miller_rabin.h"/#include "miller_rabin_native.h"/' gap_test_gpu.cu

# 3. Update Makefile: remove CGBN dependency, add -arch flag
# (see Integration Guide below)

# 4. Build and test
make gap_test_gpu BITS=1280
./gap_test_gpu --unknown-filename your_unknowns.txt --min-merit 20
```

## Files

| File | Description |
|------|-------------|
| `mr_blackwell.cu` | Standalone benchmark/test program |
| `miller_rabin_native.h` | Drop-in replacement for CGBN miller_rabin.h |
| `Makefile` | Multi-architecture build system |

## Build Requirements

- NVIDIA GPU (Kepler or newer, SM 3.5+)
- CUDA Toolkit 11.0+ (12.x recommended)
- GMP library (`libgmp-dev`)
- Linux (tested on Linux Mint)

```bash
# Ubuntu/Debian
sudo apt install libgmp-dev

# Build
make ARCH=sm_120
```

## Integration Guide

### Replacing CGBN in prime-gap

The `miller_rabin_native.h` provides the same `mr_params_t` and `test_runner_t` interface that `gap_test_gpu.cu` expects. The only changes needed are:

1. **Include**: Change `#include "miller_rabin.h"` to `#include "miller_rabin_native.h"`
2. **Makefile**: Remove `-I../CGBN/include`, add `-arch=sm_XXX` for your GPU
3. **BITS**: Compile with `BITS=` matching your primorial (e.g., `BITS=704` for P=503, `BITS=1280` for P=907)

### Supported architectures

| Architecture | SM | GPUs | Status |
|-------------|-----|------|--------|
| Kepler | 3.5+ | GTX 780, K80 | Should work (untested) |
| Maxwell | 5.x | GTX 980, Titan X | Should work (untested) |
| Pascal | 6.x | GTX 1080, P100 | Should work (untested) |
| Volta | 7.0 | V100 | Should work (untested) |
| Turing | 7.5 | RTX 2080 | Should work (untested) |
| Ampere | 8.x | RTX 3090, A100 | Should work (untested) |
| Ada Lovelace | 8.9 | RTX 4090 | Should work (untested) |
| Hopper | 9.0 | H100 | Should work (untested) |
| **Blackwell** | **12.0** | **RTX 5070/5080/5090** | **Tested ✓** |

The PTX instructions used (`mad.lo.cc.u32`, `madc.hi.u32`, `add.cc.u32`, `addc.u32`) are available on all architectures since Kepler. Contributions of benchmark results on other GPUs are welcome!

## The CGBN Problem

CGBN (NVlabs/CGBN) was designed for Volta through Ampere. Its Makefile only lists architectures up to `sm_80`. On Blackwell (SM 12.0):

- Kernels **compile** without errors
- Kernels **launch** without CUDA errors  
- Kernels **run** orders of magnitude slower than expected
- `ncu` profiling shows the kernel takes effectively forever
- Root cause: likely related to SM 12.0's redesigned compute architecture (neural shader focus)

Multiple other projects have reported SM 12.0 compatibility issues (FlashInfer, CUTLASS, PyTorch). As of April 2026, NVlabs has not updated CGBN for Blackwell.

## Technical Details

### Montgomery CIOS with PTX

The core multiplication uses a 4-instruction carry chain pattern per limb:

```
lo = madlo_cc(ai, b[j], T[j])   // lo(ai*b[j]) + T[j], set CC
hi = madhic(ai, b[j], 0)         // hi(ai*b[j]) + CC
T[j] = add_cc(lo, carry)         // lo + carry_in, set CC  
carry = addc(hi, 0)               // hi + CC
```

This avoids the 3-instruction pattern that loses carries on double overflow.

### Optimization history

1. **LIMBS dispatch**: Template on actual bit size (22 for 683-bit, not 32). Gain: ~2x
2. **Windowed exponentiation**: 4-bit window, CGBN-style position tracking. Gain: ~1.5x  
3. **PTX carry chains**: Explicit carry propagation. Gain: ~1.2x
4. **Montgomery-space comparisons**: Eliminates from_mont in squaring loop. Gain: pipeline-dependent
5. **Parallel sieve+GPU pipeline**: CPU sieve overlapped with GPU test. Gain: ~1.4x throughput
6. **v2: Montgomery constants on device**: R² mod n and n0inv computed inside the kernel via repeated doubling and Newton iteration, eliminating CPU/GMP bottleneck. Instance size: 268→136 bytes (-49%), PCIe transfer: -49%. Gain: 31% pipeline speedup (measured A/B on prime-gap, 92s→70s with same workload, identical results)

### What didn't work

- **SOS dedicated squaring**: 3x slower due to register pressure (P[2*LIMBS+1] array causes spills to local memory on Blackwell)
- **Window size 5**: Only 2.3% improvement, not worth the extra table memory

## Author

pscamillo — independent developer working on GPU kernels for number-theoretic computation.

## See also

[PSCKangaroo](https://github.com/pscamillo/PSCKangaroo) — GPU-accelerated Pollard's Kangaroo for secp256k1 ECDLP.
## License

Apache 2.0

## Acknowledgments

- [Seth Troisi](https://github.com/sethtroisi) for the prime-gap pipeline and combined sieve algorithm
- [NVlabs/CGBN](https://github.com/NVlabs/CGBN) for the reference implementation (windowed exponentiation logic)
- The [Prime Gap Searches](https://www.mersenneforum.org/forumdisplay.php?f=131) community at mersenneforum.org
- Dr. Thomas Ray Nicely (1943-2019) for creating and maintaining the prime gap records
- Anthropic Claude — pair-programming partner 

## Supporting development

If this project is useful to you, consider **starring the repository** — it helps visibility and signals to others that the work matters.

If you want to support continued development directly:

**BTC (bech32):** `bc1q0eck70y3p486ceuyggz7e68ea7pzz5sc7hwzqp`

Hardware, electricity, and tooling costs are real, though partially offset by solar power. Contributions of code, benchmarks, bug reports, and documentation are also welcome — open an issue or pull request.

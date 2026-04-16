/*
 * miller_rabin_native.h - Native Miller-Rabin for Blackwell SM 12.0
 * Drop-in replacement for CGBN-based miller_rabin.h
 * 
 * Author: pscamillo (Paulo S. Camillo)
 * License: Apache 2.0
 * 
 * Replaces CGBN (broken on SM 12.0) with native Montgomery CIOS + PTX.
 * Provides same test_runner_t interface expected by gap_test_gpu.cu.
 */

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

#include <cuda.h>
#include <gmp.h>

// ============================================================================
// Configuration matching gap_test_gpu.cu expectations
// ============================================================================

#define MR_WINDOW_BITS 5
#define MR_WINDOW_SIZE (1 << MR_WINDOW_BITS)

// ============================================================================
// Compatibility: mr_params_t (matches CGBN interface)
// ============================================================================

template<uint32_t tpi, uint32_t bits, uint32_t window_bits>
class mr_params_t {
  public:
  static const uint32_t TPB = 0;
  static const uint32_t MAX_ROTATION = 4;
  static const uint32_t SHM_LIMIT = 0;
  static const bool     CONSTANT_TIME = false;
  static const uint32_t TPI = tpi;
  static const uint32_t BITS = bits;
  static const uint32_t WINDOW_BITS = window_bits;
};

// ============================================================================
// Error checking
// ============================================================================

static void cuda_check(cudaError_t status, const char *action = NULL,
                        const char *file = NULL, int32_t line = 0) {
    if (status != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(status));
        if (action) printf("While running %s (file %s, line %d)\n", action, file, line);
        exit(1);
    }
}
#define CUDA_CHECK(action) cuda_check(action, #action, __FILE__, __LINE__)

// ============================================================================
// PTX carry chain primitives
// ============================================================================

__device__ __forceinline__ uint32_t ptx_add_cc(uint32_t a, uint32_t b) {
    uint32_t r;
    asm volatile("add.cc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

__device__ __forceinline__ uint32_t ptx_addc(uint32_t a, uint32_t b) {
    uint32_t r;
    asm volatile("addc.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

__device__ __forceinline__ uint32_t ptx_madlo_cc(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    asm volatile("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

__device__ __forceinline__ uint32_t ptx_madhic(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t r;
    asm volatile("madc.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    return r;
}

// ============================================================================
// Device: Montgomery CIOS with PTX carry chains
// ============================================================================

template<int LIMBS>
__device__ __forceinline__ void mont_mul(
    uint32_t *r, const uint32_t *a, const uint32_t *b,
    const uint32_t *n, uint32_t n0inv)
{
    uint32_t T[LIMBS + 1];
    #pragma unroll
    for (int k = 0; k <= LIMBS; k++) T[k] = 0;

    for (int i = 0; i < LIMBS; i++) {
        uint32_t ai = a[i];

        // T += ai * b (PTX carry chain)
        T[0] = ptx_madlo_cc(ai, b[0], T[0]);
        uint32_t carry = ptx_madhic(ai, b[0], 0);

        #pragma unroll
        for (int j = 1; j < LIMBS; j++) {
            uint32_t lo = ptx_madlo_cc(ai, b[j], T[j]);
            uint32_t hi = ptx_madhic(ai, b[j], 0);
            T[j] = ptx_add_cc(lo, carry);
            carry = ptx_addc(hi, 0);
        }
        T[LIMBS] += carry;

        // Reduction: m = T[0] * n0inv
        uint32_t m = T[0] * n0inv;

        // T = (T + m * n) >> 32
        ptx_madlo_cc(m, n[0], T[0]);
        carry = ptx_madhic(m, n[0], 0);

        #pragma unroll
        for (int j = 1; j < LIMBS; j++) {
            uint32_t lo = ptx_madlo_cc(m, n[j], T[j]);
            uint32_t hi = ptx_madhic(m, n[j], 0);
            T[j - 1] = ptx_add_cc(lo, carry);
            carry = ptx_addc(hi, 0);
        }
        T[LIMBS - 1] = ptx_add_cc(T[LIMBS], carry);
        T[LIMBS] = ptx_addc(0, 0);
    }

    // Conditional subtraction
    int borrow = 0;
    uint32_t tmp[LIMBS];
    #pragma unroll
    for (int j = 0; j < LIMBS; j++) {
        int64_t diff = (int64_t)T[j] - n[j] - borrow;
        tmp[j] = (uint32_t)diff;
        borrow = (diff < 0) ? 1 : 0;
    }
    int use_sub = (T[LIMBS] > 0) || (borrow == 0);
    #pragma unroll
    for (int j = 0; j < LIMBS; j++)
        r[j] = use_sub ? tmp[j] : T[j];
}

template<int LIMBS>
__device__ __forceinline__ void mont_sqr(uint32_t *r, const uint32_t *a,
                                          const uint32_t *n, uint32_t n0inv) {
    mont_mul<LIMBS>(r, a, a, n, n0inv);
}

// ============================================================================
// Device: helpers
// ============================================================================

template<int LIMBS>
__device__ __forceinline__ void to_mont(uint32_t *aR, const uint32_t *a,
    const uint32_t *R2, const uint32_t *n, uint32_t n0inv) {
    mont_mul<LIMBS>(aR, a, R2, n, n0inv);
}

template<int LIMBS>
__device__ __forceinline__ void from_mont(uint32_t *a, const uint32_t *aR,
    const uint32_t *n, uint32_t n0inv) {
    uint32_t one[LIMBS];
    #pragma unroll
    for (int j = 0; j < LIMBS; j++) one[j] = 0;
    one[0] = 1;
    mont_mul<LIMBS>(a, aR, one, n, n0inv);
}

template<int LIMBS>
__device__ __forceinline__ bool bignum_eq_ui(const uint32_t *a, uint32_t val) {
    if (a[0] != val) return false;
    #pragma unroll
    for (int j = 1; j < LIMBS; j++) if (a[j] != 0) return false;
    return true;
}

template<int LIMBS>
__device__ __forceinline__ int bignum_cmp(const uint32_t *a, const uint32_t *b) {
    for (int j = LIMBS - 1; j >= 0; j--) {
        if (a[j] > b[j]) return 1;
        if (a[j] < b[j]) return -1;
    }
    return 0;
}

template<int LIMBS>
__device__ __forceinline__ int bignum_ctz(const uint32_t *a) {
    int count = 0;
    for (int j = 0; j < LIMBS; j++) {
        if (a[j] == 0) { count += 32; }
        else { count += __ffs(a[j]) - 1; break; }
    }
    return count;
}

template<int LIMBS>
__device__ __forceinline__ void bignum_shr(uint32_t *r, const uint32_t *a, int shift) {
    int ws = shift / 32, bs = shift % 32;
    #pragma unroll
    for (int j = 0; j < LIMBS; j++) {
        int src = j + ws;
        uint32_t lo = (src < LIMBS) ? a[src] : 0;
        uint32_t hi = (src + 1 < LIMBS) ? a[src + 1] : 0;
        r[j] = (bs == 0) ? lo : (lo >> bs) | (hi << (32 - bs));
    }
}

template<int LIMBS>
__device__ __forceinline__ int bignum_bits(const uint32_t *a) {
    for (int j = LIMBS - 1; j >= 0; j--)
        if (a[j] != 0) return j * 32 + (32 - __clz(a[j]));
    return 0;
}

template<int LIMBS>
__device__ __forceinline__ void bignum_copy(uint32_t *dst, const uint32_t *src) {
    #pragma unroll
    for (int j = 0; j < LIMBS; j++) dst[j] = src[j];
}

template<int LIMBS>
__device__ __forceinline__ uint32_t extract_bits(const uint32_t *exp, int pos) {
    int word = pos / 32, bit = pos % 32;
    if (word >= LIMBS) return 0;
    uint32_t val = exp[word] >> bit;
    if (bit + MR_WINDOW_BITS > 32 && word + 1 < LIMBS)
        val |= exp[word + 1] << (32 - bit);
    return val & ((1u << MR_WINDOW_BITS) - 1);
}

// ============================================================================
// Device: Windowed modular exponentiation
// ============================================================================

template<int LIMBS>
__device__ void mont_powm_windowed(
    uint32_t *result, const uint32_t *base_mont, const uint32_t *exp,
    const uint32_t *n, const uint32_t *R2, uint32_t n0inv, int exp_bits)
{
    uint32_t table[MR_WINDOW_SIZE * LIMBS];

    uint32_t one[LIMBS];
    #pragma unroll
    for (int j = 0; j < LIMBS; j++) one[j] = 0;
    one[0] = 1;
    to_mont<LIMBS>(table, one, R2, n, n0inv);

    bignum_copy<LIMBS>(table + LIMBS, base_mont);
    for (int i = 2; i < MR_WINDOW_SIZE; i++)
        mont_mul<LIMBS>(table + i * LIMBS, table + (i-1) * LIMBS, base_mont, n, n0inv);

    int position = exp_bits;
    int offset = position % MR_WINDOW_BITS;
    if (offset == 0) position -= MR_WINDOW_BITS;
    else position -= offset;

    uint32_t idx = extract_bits<LIMBS>(exp, position);
    bignum_copy<LIMBS>(result, table + idx * LIMBS);

    while (position > 0) {
        for (int s = 0; s < MR_WINDOW_BITS; s++)
            mont_sqr<LIMBS>(result, result, n, n0inv);
        position -= MR_WINDOW_BITS;
        idx = extract_bits<LIMBS>(exp, position);
        uint32_t t[LIMBS];
        bignum_copy<LIMBS>(t, table + idx * LIMBS);
        mont_mul<LIMBS>(result, result, t, n, n0inv);
    }
}

// ============================================================================
// Instance data for GPU
// ============================================================================

template<int LIMBS>
struct NativeInstance {
    uint32_t candidate[LIMBS];
    uint32_t R2[LIMBS];
    uint32_t n0inv;
    int      bits;
    int      passed;
};

// ============================================================================
// Miller-Rabin kernel
// ============================================================================

template<int LIMBS>
__global__ void kernel_miller_rabin_native(NativeInstance<LIMBS> *instances, uint32_t count) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    NativeInstance<LIMBS> &inst = instances[idx];
    const uint32_t *n = inst.candidate;
    const uint32_t *R2 = inst.R2;
    uint32_t n0inv = inst.n0inv;

    uint32_t nm1[LIMBS];
    bignum_copy<LIMBS>(nm1, n);
    int borrow = 1;
    for (int j = 0; j < LIMBS; j++) {
        uint64_t diff = (uint64_t)nm1[j] - borrow;
        nm1[j] = (uint32_t)diff;
        borrow = (diff >> 63) & 1;
    }

    int trailing = bignum_ctz<LIMBS>(nm1);
    uint32_t d[LIMBS];
    bignum_shr<LIMBS>(d, nm1, trailing);
    int d_bits = bignum_bits<LIMBS>(d);
    if (d_bits == 0) { inst.passed = 0; return; }

    uint32_t base[LIMBS];
    #pragma unroll
    for (int j = 0; j < LIMBS; j++) base[j] = 0;
    base[0] = 2;
    uint32_t base_mont[LIMBS];
    to_mont<LIMBS>(base_mont, base, R2, n, n0inv);

    // Pre-compute 1 and n-1 in Montgomery space (avoids from_mont in hot loop)
    uint32_t one_mont[LIMBS], nm1_mont[LIMBS];
    {
        uint32_t one[LIMBS];
        #pragma unroll
        for (int j = 0; j < LIMBS; j++) one[j] = 0;
        one[0] = 1;
        to_mont<LIMBS>(one_mont, one, R2, n, n0inv);
    }
    to_mont<LIMBS>(nm1_mont, nm1, R2, n, n0inv);

    // x = 2^d mod n (windowed, in Montgomery form)
    uint32_t x_mont[LIMBS];
    mont_powm_windowed<LIMBS>(x_mont, base_mont, d, n, R2, n0inv, d_bits);

    // Compare in Montgomery space — NO from_mont needed!
    if (bignum_cmp<LIMBS>(x_mont, one_mont) == 0 ||
        bignum_cmp<LIMBS>(x_mont, nm1_mont) == 0) {
        inst.passed = 1;
        return;
    }

    // Repeated squaring — all comparisons in Montgomery space
    for (int i = 1; i < trailing; i++) {
        mont_sqr<LIMBS>(x_mont, x_mont, n, n0inv);
        if (bignum_cmp<LIMBS>(x_mont, one_mont) == 0) { inst.passed = 0; return; }
        if (bignum_cmp<LIMBS>(x_mont, nm1_mont) == 0) { inst.passed = 1; return; }
    }

    inst.passed = 0;
}

// ============================================================================
// Host: Montgomery constant computation
// ============================================================================

static void compute_R2_mod_n(uint32_t *R2, const uint32_t *n_limbs, int limbs) {
    mpz_t n, R, R2_mpz;
    mpz_init(n); mpz_init(R); mpz_init(R2_mpz);
    mpz_import(n, limbs, -1, sizeof(uint32_t), 0, 0, n_limbs);
    mpz_setbit(R, 32 * limbs);
    mpz_mul(R2_mpz, R, R);
    mpz_mod(R2_mpz, R2_mpz, n);
    size_t count;
    memset(R2, 0, limbs * sizeof(uint32_t));
    mpz_export(R2, &count, -1, sizeof(uint32_t), 0, 0, R2_mpz);
    mpz_clear(n); mpz_clear(R); mpz_clear(R2_mpz);
}

static uint32_t compute_n0inv(uint32_t n0) {
    uint32_t x = 1;
    for (int i = 0; i < 5; i++) x = x * (2 - n0 * x);
    return (uint32_t)(-(int64_t)x);
}

static void from_mpz_native(mpz_t s, uint32_t *x, uint32_t count) {
    size_t words;
    if (mpz_sizeinbase(s, 2) > count * 32) {
        fprintf(stderr, "from_mpz_native: number too large for %d limbs\n", count);
        exit(1);
    }
    mpz_export(x, &words, -1, sizeof(uint32_t), 0, 0, s);
    while (words < count) x[words++] = 0;
}

// ============================================================================
// test_runner_t: compatible interface for gap_test_gpu.cu
// ============================================================================

template<class params>
class test_runner_t {
  public:
    static const int LIMBS = params::BITS / 32;

    const size_t n;       // max batch size
    const size_t rounds;  // MR rounds (base 2 = 1 round)

    NativeInstance<params::BITS / 32> *h_instances;
    NativeInstance<params::BITS / 32> *d_instances;

    size_t batch_count;
    double total_kernel_ms;

    test_runner_t(size_t n, size_t rounds)
        : n(n), rounds(rounds), batch_count(0), total_kernel_ms(0.0)
    {
        CUDA_CHECK(cudaSetDevice(0));

        // Pinned host memory for fast transfers
        CUDA_CHECK(cudaMallocHost((void**)&h_instances,
                   n * sizeof(NativeInstance<LIMBS>)));

        // Device memory
        CUDA_CHECK(cudaMalloc((void**)&d_instances,
                   n * sizeof(NativeInstance<LIMBS>)));

        printf("[Native MR] Initialized: BITS=%d, LIMBS=%d, batch=%ld, "
               "instance=%ld bytes\n",
               params::BITS, LIMBS, n, sizeof(NativeInstance<LIMBS>));
    }

    ~test_runner_t() {
        CUDA_CHECK(cudaFreeHost(h_instances));
        CUDA_CHECK(cudaFree(d_instances));

        if (batch_count > 0) {
            printf("\n[Native MR] Total: %ld batches, %.1f ms kernel, "
                   "avg %.1f ms/batch (%.0f PRP/sec avg)\n",
                   batch_count, total_kernel_ms,
                   total_kernel_ms / batch_count,
                   (batch_count * n) / (total_kernel_ms / 1000.0));
        }
    }

    void run_test(const std::vector<mpz_t*> &tests, std::vector<int> &results) {
        if (tests.size() == 0) return;
        assert(tests.size() <= n);

        // Convert mpz_t candidates to native format + compute Montgomery constants
        for (size_t i = 0; i < tests.size(); i++) {
            auto &inst = h_instances[i];

// Skip zero/even candidates (unused batch slots)
            if (mpz_sgn(*tests[i]) == 0 || mpz_even_p(*tests[i])) {
                memset(&inst, 0, sizeof(inst));
                inst.passed = 0;
                continue;
            }


            // Export candidate to uint32_t limbs
            from_mpz_native(*tests[i], inst.candidate, LIMBS);

            // Compute R^2 mod n
            compute_R2_mod_n(inst.R2, inst.candidate, LIMBS);

            // Compute -n^{-1} mod 2^32
            inst.n0inv = compute_n0inv(inst.candidate[0]);

            // Bit count
            inst.bits = mpz_sizeinbase(*tests[i], 2);

            inst.passed = -1;
        }

        // Copy to GPU
        CUDA_CHECK(cudaMemcpy(d_instances, h_instances,
                   sizeof(NativeInstance<LIMBS>) * tests.size(),
                   cudaMemcpyHostToDevice));

        // Launch kernel
        int tpb = 128;
        int blocks = (tests.size() + tpb - 1) / tpb;

        auto t0 = std::chrono::high_resolution_clock::now();

        kernel_miller_rabin_native<LIMBS><<<blocks, tpb>>>(
            d_instances, tests.size());
        CUDA_CHECK(cudaDeviceSynchronize());

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        batch_count++;
        total_kernel_ms += ms;

        if (batch_count <= 3 || batch_count % 10 == 0) {
            printf("[Native MR] Batch #%ld: %ld tests, %.1f ms (%.0f PRP/sec)\n",
                   batch_count, tests.size(), ms,
                   tests.size() / (ms / 1000.0));
        }

        // Copy results back
        CUDA_CHECK(cudaMemcpy(h_instances, d_instances,
                   sizeof(NativeInstance<LIMBS>) * tests.size(),
                   cudaMemcpyDeviceToHost));

        // Fill results vector
        for (size_t i = 0; i < tests.size(); i++) {
            results[i] = h_instances[i].passed == 1 ? 1 : 0;
        }
    }
};

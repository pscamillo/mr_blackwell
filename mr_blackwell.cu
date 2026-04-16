/*
 * mr_blackwell.cu - Native Miller-Rabin PRP Test for NVIDIA GPUs
 * 
 * Author: Camillo / pscamillo (Paulo S. Camillo)
 * License: Apache 2.0
 * Repository: https://github.com/pscamillo/mr_blackwell
 * 
 * A high-performance Miller-Rabin primality testing kernel built from scratch
 * as a drop-in replacement for CGBN (which is broken on Blackwell SM 12.0).
 * 
 * Features:
 *   - Montgomery CIOS multiplication with PTX carry chain primitives
 *   - Fixed-window modular exponentiation (5-bit default)
 *   - Montgomery-space comparisons (no from_mont in hot loop)
 *   - Template-based LIMBS dispatch for optimal sizing
 *   - Warmup run for accurate benchmarking
 *   - GMP verification for correctness testing
 * 
 * Build:
 *   nvcc -O3 -arch=sm_120 -o mr_blackwell mr_blackwell.cu -lgmp
 * 
 * Run:
 *   ./mr_blackwell [num_candidates] [bits]
 *   ./mr_blackwell 65536 683    # P=503 prime gaps (22 limbs)
 *   ./mr_blackwell 16384 1240   # P=907 prime gaps (40 limbs)
 * 
 * Used to set 5 world records in prime gap searches (April 2026).
 */

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

#include <cuda.h>
#include <gmp.h>

#ifndef MR_WINDOW_BITS
#define MR_WINDOW_BITS 5
#endif
#define MR_WINDOW_SIZE (1 << MR_WINDOW_BITS)

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// PTX carry chain primitives (available since Kepler SM 3.5+)
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

        uint32_t m = T[0] * n0inv;

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
// Device: Windowed modular exponentiation (CGBN-style)
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
// Instance data
// ============================================================================

template<int LIMBS>
struct MRInstance {
    uint32_t candidate[LIMBS];
    uint32_t R2[LIMBS];
    uint32_t n0inv;
    int      bits;
    int      passed;
};

// ============================================================================
// Miller-Rabin kernel (with Montgomery-space comparisons)
// ============================================================================

template<int LIMBS>
__global__ void kernel_miller_rabin(MRInstance<LIMBS> *instances, uint32_t count) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    MRInstance<LIMBS> &inst = instances[idx];
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

    // Pre-compute 1 and n-1 in Montgomery space
    uint32_t one_mont[LIMBS], nm1_mont[LIMBS];
    {
        uint32_t one[LIMBS];
        #pragma unroll
        for (int j = 0; j < LIMBS; j++) one[j] = 0;
        one[0] = 1;
        to_mont<LIMBS>(one_mont, one, R2, n, n0inv);
    }
    to_mont<LIMBS>(nm1_mont, nm1, R2, n, n0inv);

    uint32_t x_mont[LIMBS];
    mont_powm_windowed<LIMBS>(x_mont, base_mont, d, n, R2, n0inv, d_bits);

    // Compare in Montgomery space — no from_mont needed
    if (bignum_cmp<LIMBS>(x_mont, one_mont) == 0 ||
        bignum_cmp<LIMBS>(x_mont, nm1_mont) == 0) {
        inst.passed = 1;
        return;
    }

    for (int i = 1; i < trailing; i++) {
        mont_sqr<LIMBS>(x_mont, x_mont, n, n0inv);
        if (bignum_cmp<LIMBS>(x_mont, one_mont) == 0) { inst.passed = 0; return; }
        if (bignum_cmp<LIMBS>(x_mont, nm1_mont) == 0) { inst.passed = 1; return; }
    }

    inst.passed = 0;
}

// ============================================================================
// Host code
// ============================================================================

void compute_R2_mod_n(uint32_t *R2, const uint32_t *n_limbs, int limbs) {
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

uint32_t compute_n0inv(uint32_t n0) {
    uint32_t x = 1;
    for (int i = 0; i < 5; i++) x = x * (2 - n0 * x);
    return (uint32_t)(-(int64_t)x);
}

void generate_random_odd(uint32_t *limbs, int num_limbs, int bits, gmp_randstate_t state) {
    mpz_t r; mpz_init(r);
    mpz_urandomb(r, state, bits);
    mpz_setbit(r, bits - 1);
    mpz_setbit(r, 0);
    size_t count;
    memset(limbs, 0, num_limbs * sizeof(uint32_t));
    mpz_export(limbs, &count, -1, sizeof(uint32_t), 0, 0, r);
    mpz_clear(r);
}

int gmp_miller_rabin(const uint32_t *n_limbs, int limbs) {
    mpz_t n; mpz_init(n);
    mpz_import(n, limbs, -1, sizeof(uint32_t), 0, 0, n_limbs);
    int result = mpz_probab_prime_p(n, 1);
    mpz_clear(n);
    return (result > 0) ? 1 : 0;
}

void print_gpu_info() {
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s (SM %d.%d, %d SMs, %.0f MHz)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount, prop.clockRate / 1000.0);
    printf("VRAM: %ld MB, CUDA: %d.%d\n\n",
           prop.totalGlobalMem / (1024*1024),
           CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
}

template<int LIMBS>
void run_benchmark(int num_candidates, int bits) {
    printf("=== mr_blackwell — Native Miller-Rabin GPU Kernel ===\n");
    printf("Candidates: %d | Bits: %d | LIMBS: %d | Window: %d\n",
           num_candidates, bits, LIMBS, MR_WINDOW_BITS);

    gmp_randstate_t rng;
    gmp_randinit_mt(rng);
    gmp_randseed_ui(rng, 42);

    std::vector<MRInstance<LIMBS>> h_inst(num_candidates);

    printf("Generating candidates...\n");
    auto ts0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_candidates; i++) {
        generate_random_odd(h_inst[i].candidate, LIMBS, bits, rng);
        compute_R2_mod_n(h_inst[i].R2, h_inst[i].candidate, LIMBS);
        h_inst[i].n0inv = compute_n0inv(h_inst[i].candidate[0]);
        h_inst[i].bits = bits;
        h_inst[i].passed = -1;
    }
    auto ts1 = std::chrono::high_resolution_clock::now();
    printf("Host setup: %.1f ms\n",
           std::chrono::duration<double, std::milli>(ts1 - ts0).count());

    MRInstance<LIMBS> *d_inst;
    size_t mem = num_candidates * sizeof(MRInstance<LIMBS>);
    CUDA_CHECK(cudaMalloc(&d_inst, mem));
    CUDA_CHECK(cudaMemcpy(d_inst, h_inst.data(), mem, cudaMemcpyHostToDevice));

    int tpb = 128;
    int blocks = (num_candidates + tpb - 1) / tpb;

    // Warmup
    kernel_miller_rabin<LIMBS><<<blocks, tpb>>>(d_inst, num_candidates);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < num_candidates; i++) h_inst[i].passed = -1;
    CUDA_CHECK(cudaMemcpy(d_inst, h_inst.data(), mem, cudaMemcpyHostToDevice));

    // Timed run
    auto t0 = std::chrono::high_resolution_clock::now();
    kernel_miller_rabin<LIMBS><<<blocks, tpb>>>(d_inst, num_candidates);
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("[GPU] Kernel: %d tests in %.1f ms (%.0f PRP/sec)\n",
           num_candidates, ms, num_candidates / (ms / 1000.0));

    CUDA_CHECK(cudaMemcpy(h_inst.data(), d_inst, mem, cudaMemcpyDeviceToHost));

    printf("Verifying against GMP...\n");
    int correct = 0, primes = 0, errors = 0;
    for (int i = 0; i < num_candidates; i++) {
        int gpu = h_inst[i].passed;
        int ref = gmp_miller_rabin(h_inst[i].candidate, LIMBS);
        if (gpu == ref) correct++;
        else {
            errors++;
            if (errors <= 5) printf("  MISMATCH[%d]: GPU=%d GMP=%d\n", i, gpu, ref);
        }
        if (gpu == 1) primes++;
    }

    printf("\nResults:\n");
    printf("  Correct: %d / %d (%.2f%%)\n", correct, num_candidates,
           100.0 * correct / num_candidates);
    printf("  Primes: %d (%.2f%%)\n", primes, 100.0 * primes / num_candidates);
    printf("  Errors: %d\n", errors);
    printf("  Performance: %.0f PRP/sec (%.1fx vs CPU@13700)\n",
           num_candidates / (ms / 1000.0),
           (num_candidates / (ms / 1000.0)) / 13700.0);

    CUDA_CHECK(cudaFree(d_inst));
    gmp_randclear(rng);
}

int main(int argc, char **argv) {
    int num = 16384, bits = 683;
    if (argc > 1) num = atoi(argv[1]);
    if (argc > 2) bits = atoi(argv[2]);

    print_gpu_info();

    int limbs = (bits + 31) / 32;
    printf("Bits: %d -> Limbs: %d\n\n", bits, limbs);

    if      (limbs <= 16) run_benchmark<16>(num, bits);
    else if (limbs <= 22) run_benchmark<22>(num, bits);
    else if (limbs <= 32) run_benchmark<32>(num, bits);
    else if (limbs <= 40) run_benchmark<40>(num, bits);
    else if (limbs <= 48) run_benchmark<48>(num, bits);
    else if (limbs <= 64) run_benchmark<64>(num, bits);
    else { printf("ERROR: max 2048 bits (64 limbs)\n"); return 1; }

    return 0;
}

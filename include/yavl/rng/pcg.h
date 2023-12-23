#pragma once

// Most copied from:
// https://github.com/wjakob/pcg32/blob/master/pcg32.h

#define PCG32_DEFAULT_STATE  0x853c49e6748fea9bULL
#define PCG32_DEFAULT_STREAM 0xda3e39cb94b95bdbULL
#define PCG32_MULT           0x5851f42d4c957f2dULL

#include <inttypes.h>
#include <cmath>
#include <cassert>
#include <algorithm>

#include <yavl/platform.h>

namespace yavl
{

// PCG32 Pseudorandom number generator
struct pcg32 {
    uint64_t state;     // RNG state. All values are possible.
    uint64_t inc;       // Controls which RNG sequence (stream) is selected.

    // Initialize the pseudorandom number generator with default seed
    pcg32() : state(PCG32_DEFAULT_STATE), inc(PCG32_DEFAULT_STREAM) {}

    // Initialize the pseudorandom number generator with the \ref seed() function
    pcg32(uint64_t initstate, uint64_t initseq = 1u) { seed(initstate, initseq); }

    // Seed the pseudorandom number generator
    void seed(uint64_t initstate, uint64_t initseq = 1) {
        state = 0U;
        inc = (initseq << 1u) | 1u;
        next_uint();
        state += initstate;
        next_uint();
    }

    // Generate a uniformly distributed unsigned 32-bit random number
    uint32_t next_uint() {
        uint64_t oldstate = state;
        state = oldstate * PCG32_MULT + inc;
        uint32_t xorshifted = (uint32_t) (((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = (uint32_t) (oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
    }

    // Generate a uniformly distributed number, r, where 0 <= r < bound
    uint32_t next_uint(uint32_t bound) {
        // To avoid bias, we need to make the range of the RNG a multiple of
        // bound, which we do by dropping output less than a threshold.
        // A naive scheme to calculate the threshold would be to do
        //
        //     uint32_t threshold = 0x100000000ull % bound;
        //
        // but 64-bit div/mod is slower than 32-bit div/mod (especially on
        // 32-bit platforms).  In essence, we do
        //
        //     uint32_t threshold = (0x100000000ull-bound) % bound;
        //
        // because this version will calculate the same modulus, but the LHS
        // value is less than 2^32.

        uint32_t threshold = (~bound+1u) % bound;

        // Uniformity guarantees that this loop will terminate.  In practice, it
        // should usually terminate quickly; on average (assuming all bounds are
        // equally likely), 82.25% of the time, we can expect it to require just
        // one iteration.  In the worst case, someone passes a bound of 2^31 + 1
        // (i.e., 2147483649), which invalidates almost 50% of the range.  In
        // practice, bounds are typically small and only a tiny amount of the range
        // is eliminated.
        for (;;) {
            uint32_t r = next_uint();
            if (r >= threshold)
                return r % bound;
        }
    }

    // Generate a single precision floating point value on the interval [0, 1)
    float next_float() {
        /* Trick from MTGP: generate an uniformly distributed
           single precision number in [1,2) and subtract 1. */
        union {
            uint32_t u;
            float f;
        } x;
        x.u = (next_uint() >> 9) | 0x3f800000u;
        return x.f - 1.0f;
    }

    /**
     * \brief Generate a double precision floating point value on the interval [0, 1)
     *
     * \remark Since the underlying random number generator produces 32 bit output,
     * only the first 32 mantissa bits will be filled (however, the resolution is still
     * finer than in \ref nextFloat(), which only uses 23 mantissa bits)
     */
    double next_double() {
        /* Trick from MTGP: generate an uniformly distributed
           double precision number in [1,2) and subtract 1. */
        union {
            uint64_t u;
            double d;
        } x;
        x.u = ((uint64_t) next_uint() << 20) | 0x3ff0000000000000ULL;
        return x.d - 1.0;
    }

    /**
     * \brief Multi-step advance function (jump-ahead, jump-back)
     *
     * The method used here is based on Brown, "Random Number Generation
     * with Arbitrary Stride", Transactions of the American Nuclear
     * Society (Nov. 1994). The algorithm is very similar to fast
     * exponentiation.
     */
    void advance(int64_t delta_) {
        uint64_t
            cur_mult = PCG32_MULT,
            cur_plus = inc,
            acc_mult = 1u,
            acc_plus = 0u;

        /* Even though delta is an unsigned integer, we can pass a signed
           integer to go backwards, it just goes "the long way round". */
        uint64_t delta = (uint64_t) delta_;

        while (delta > 0) {
            if (delta & 1) {
                acc_mult *= cur_mult;
                acc_plus = acc_plus * cur_mult + cur_plus;
            }
            cur_plus = (cur_mult + 1) * cur_plus;
            cur_mult *= cur_mult;
            delta /= 2;
        }
        state = acc_mult * state + acc_plus;
    }

    /**
     * \brief Draw uniformly distributed permutation and permute the
     * given STL container
     *
     * From: Knuth, TAoCP Vol. 2 (3rd 3d), Section 3.4.2
     */
    template <typename Iterator> void shuffle(Iterator begin, Iterator end) {
        for (Iterator it = end - 1; it > begin; --it)
            std::iter_swap(it, begin + next_uint((uint32_t) (it - begin + 1)));
    }

    /// Compute the distance between two PCG32 pseudorandom number generators
    int64_t operator-(const pcg32 &other) const {
        assert(inc == other.inc);

        uint64_t
            cur_mult = PCG32_MULT,
            cur_plus = inc,
            cur_state = other.state,
            the_bit = 1u,
            distance = 0u;

        while (state != cur_state) {
            if ((state & the_bit) != (cur_state & the_bit)) {
                cur_state = cur_state * cur_mult + cur_plus;
                distance |= the_bit;
            }
            assert((state & the_bit) == (cur_state & the_bit));
            the_bit <<= 1;
            cur_plus = (cur_mult + 1ULL) * cur_plus;
            cur_mult *= cur_mult;
        }

        return (int64_t) distance;
    }

    /// Equality operator
    bool operator==(const pcg32 &other) const { return state == other.state && inc == other.inc; }

    /// Inequality operator
    bool operator!=(const pcg32 &other) const { return state != other.state || inc != other.inc; }
};

template <uint32_t N>
struct pcg32x {
    pcg32 rng[N];

    pcg32x() {
        std::array<uint64_t, N> initstate{ PCG32_DEFAULT_STATE };
        std::array<uint64_t, N> initseq = integer_range_array<uint64_t, 1, N>;

        seed(initstate, initseq);
    }

    pcg32x(const std::array<uint64_t, N>& initstate, const std::array<uint64_t, N>& initseq) {
        seed(initstate, initseq);
    }

    void seed(const std::array<uint64_t, N>& initstate, const std::array<uint64_t, N>& initseq) {
        for (int i = 0; i < N; ++i)
            rng[i].seed(initstate[i], initseq[i]);
    }

    void next_uints(std::array<uint32_t, N>& result) {
        for (int i = 0; i < N; ++i)
            result[i] = rng[i].next_uint();
    }

    void next_floats(std::array<float, N>& result) {
        for (int i = 0; i < N; ++i)
            result[i] = rng[i].next_float();
    }

    void next_doubles(std::array<double, N>& result) {
        for (int i = 0; i < N; ++i)
            result[i] = rng[i].next_double();
    }
};

#if !defined(YAVL_DISABLE_VECTORIZATION)

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-attributes"
#elif defined(_MSC_VER)
// Do nothing here for now
#endif

#if defined(YAVL_X86_AVX512VL)

template <>
struct alignas(64) pcg32x<8> {
    __m512i state;
    __m512i inc;

    // Ctors
    pcg32x() {
        std::array<uint64_t, 8> initstate{ PCG32_DEFAULT_STATE };
        std::array<uint64_t, 8> initseq{ 1, 2, 3, 4, 5, 6, 7, 8 };

        seed(initstate, initseq);
    }

    pcg32x(const std::array<uint64_t, 8>& initstate, const std::array<uint64_t, 8>& initseq) {
        seed(initstate, initseq);
    }

    // Seed the pseudorandom number generator
    void seed(const std::array<uint64_t, 8>& initstate, const std::array<uint64_t, 8>& initseq) {
        const __m512i one = _mm512_set1_epi64(1);

        state = _mm512_setzero_si512();
        inc = _mm512_or_si512(
            _mm512_slli_epi64(_mm512_load_si512((__m512i*) &initseq), 1), one);
        step();

        state = _mm512_add_epi64(state, _mm512_load_si256((__m512i*) &initstate));

        step();
    }

    // Generate 8 uniformly distributed unsigned 32-bit random numbers
    void next_uints(std::array<uint32_t, 8>& result) {
        _mm256_store_si256((__m256i*) result.data(), step());
    }

    __m256i next_uints() {
        return step();
    }

    // Generate 8 single precision floating point value on the interval [0, 1)
    __m256 next_floats() {
        const __m256i const1 = _mm256_set1_epi32((int) 0x3f800000u);

        __m256i value = step();
        __m256i fltval = _mm256_or_si256(_mm256_srli_epi32(value, 9), const1);

        return _mm256_sub_ps(_mm256_castsi256_ps(fltval),
                             _mm256_castsi256_ps(const1));
    }

    void next_floats(std::array<float, 8>& result) {
        _mm256_store_ps(result.data(), next_floats());
    }

    // Generate 8 double precision floating point value on the interval [0, 1)
    __m512d next_doubles() {
        const __m512i const1 =
            _mm512_set1_epi64(0x3ff0000000000000ull);

        __m256i value = step();
        __m512i ret = _mm512_cvtepu32_epi64(value);
        __m512i tret = _mm512_or_si512(_mm512_slli_epi64(ret, 20), const1);
        __m512d fret = _mm512_sub_pd(_mm512_castsi512_pd(tret),
                                     _mm512_castsi512_pd(const1));
        return fret;
    }

    void next_doubles(std::array<double, 8>& result) {
        auto value = next_doubles();
        _mm512_store_pd(result.data(), value);
    }

private:
    inline __m256i step() {
        const __m512i pcg32_mult_l = _mm512_set1_epi64(pcg32_mult & 0xffffffffu);
        const __m512i pcg32_mult_h = _mm512_set1_epi64(pcg32_mult >> 32);
        const __m512i mask_l       = _mm512_set1_epi64(0x3ff0000000000000ull);
        const __m512i shift        = _mm512_set_epi32(6, 4, 2, 0, 7, 7, 7, 7,
                                                      7, 7, 7, 7, 6, 4, 2, 0);
        const __m256i const32      = _mm256_set1_epi32(32);

        auto s = state;

        /* extract low and high words for partial products below */
        __m512i lo = _mm512_and_si512(s, mask_l);
        __m512i hi = _mm512_srli_epi64(s, 32);

        /* improve high bits using xorshift step */
        __m512i ss = _mm512_srli_epi64(s, 18);
        __m512 sx = _mm512_xor_si512(ss, s);
        __m512i sxs = _mm512_srli_epi64(sx, 27);
        __m512i xors = _mm512_and_si512(mask_l, sxs);

        /* use high bits to choose a bit-level rotation */
        __m512i rot = _mm512_srli_epi64(s, 59);

        /* 64 bit multiplication using 32 bit partial products :( */
        __m512i m_hl = _mm512_mul_epi32(hi, pcg32_mult_l);
        __m512i m_lh = _mm512_mul_epi32(lo, pcg32_mult_h);

        /* assemble lower 32 bits, will be merged into one 256 bit vector below */
        xors = _mm512_permutevar_epi32(xors, shift);
        rot = _mm512_permutevar_epi32(rot, shift);

        /* continue with partial products */
        __m512i m_ll = _mm512_mul_epi32(lo, pcg32_mult_l);
        __m512i mh = _mm512_add_epi64(m_hl, m_lh);
        __m512i mhs = _mm512_slli_epi64(mh, 32);
        __m512i sn = _mm512_add_epi64(mhs, m_ll);

        __m256i xors256 = _mm256_or_si256(_mm512_extracti32x8_epi32(xors, 0),
                                          _mm512_extracti32x8_epi32(xors, 1));
        __m256i rot256  = _mm256_or_si256(_mm512_extracti32x8_epi32(rot, 0),
                                          _mm512_extracti32x8_epi32(rot, 1));

        state = _mm512_add_epi64(sn, inc);

        /* finally, rotate and return the result */
        __m256i result = _mm256_or_si256(
            _mm256_srlv_epi32(xors256, rot256),
            _mm256_sllv_epi32(xors256, _mm256_sub_epi32(const32, rot256))
        );

        return result;
    }
};

#elif defined(YAVL_X86_AVX2)

// 8 parallel PCG32 pseudorandom number generators, mostly original impl from wenzel's
template <>
struct alignas(32) pcg32x<8> {
    __m256i state[2];
    __m256i inc[2];

    // Initialize the pseudorandom number generator with default seed
    pcg32x() {
        std::array<uint64_t, 8> initstate = {
            PCG32_DEFAULT_STATE, PCG32_DEFAULT_STATE,
            PCG32_DEFAULT_STATE, PCG32_DEFAULT_STATE,
            PCG32_DEFAULT_STATE, PCG32_DEFAULT_STATE,
            PCG32_DEFAULT_STATE, PCG32_DEFAULT_STATE
        };

        std::array<uint64_t, 8> initseq =
            { 1, 2, 3, 4, 5, 6, 7, 8 };

        seed(initstate, initseq);
    }

    // Initialize the pseudorandom number generator with the \ref seed() function
    pcg32x(const std::array<uint64_t, 8>& initstate, const std::array<uint64_t, 8>& initseq) {
        seed(initstate, initseq);
    }

    /**
     * \brief Seed the pseudorandom number generator
     *
     * Specified in two parts: a state initializer and a sequence selection
     * constant (a.k.a. stream id)
     */
    void seed(const std::array<uint64_t, 8>& initstate, const std::array<uint64_t, 8>& initseq) {
        const __m256i one = _mm256_set1_epi64x((long long) 1);

        state[0] = state[1] = _mm256_setzero_si256();
        inc[0] = _mm256_or_si256(
            _mm256_slli_epi64(_mm256_load_si256((__m256i *) &initseq[0]), 1),
            one);
        inc[1] = _mm256_or_si256(
            _mm256_slli_epi64(_mm256_load_si256((__m256i *) &initseq[4]), 1),
            one);
        step();

        state[0] = _mm256_add_epi64(state[0], _mm256_load_si256((__m256i *) &initstate[0]));
        state[1] = _mm256_add_epi64(state[1], _mm256_load_si256((__m256i *) &initstate[4]));

        step();
    }

    // Generate 8 uniformly distributed unsigned 32-bit random numbers
    void next_uints(std::array<uint32_t, 8>& result) {
        _mm256_store_si256((__m256i *) result.data(), step());
    }

    // Generate 8 uniformly distributed unsigned 32-bit random numbers
    __m256i next_uints() {
        return step();
    }

    // Generate eight single precision floating point value on the interval [0, 1)
    __m256 next_floats() {
        /* Trick from MTGP: generate an uniformly distributed
           single precision number in [1,2) and subtract 1. */

        const __m256i const1 = _mm256_set1_epi32((int) 0x3f800000u);

        __m256i value = step();
        __m256i fltval = _mm256_or_si256(_mm256_srli_epi32(value, 9), const1);

        return _mm256_sub_ps(_mm256_castsi256_ps(fltval),
                             _mm256_castsi256_ps(const1));
    }

    // Generate eight single precision floating point value on the interval [0, 1)
    void next_floats(std::array<float, 8>& result) {
        _mm256_store_ps(result.data(), next_floats());
    }

    /**
     * \brief Generate eight double precision floating point value on the interval [0, 1)
     *
     * \remark Since the underlying random number generator produces 32 bit output,
     * only the first 32 mantissa bits will be filled (however, the resolution is still
     * finer than in \ref nextFloat(), which only uses 23 mantissa bits)
     */
    std::pair<__m256d, __m256d> next_doubles() {
        /* Trick from MTGP: generate an uniformly distributed
           double precision number in [1,2) and subtract 1. */

        const __m256i const1 =
            _mm256_set1_epi64x((long long) 0x3ff0000000000000ull);

        __m256i value = step();

        __m256i lo = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(value));
        __m256i hi = _mm256_cvtepu32_epi64(_mm256_extractf128_si256(value, 1));

        __m256i tlo = _mm256_or_si256(_mm256_slli_epi64(lo, 20), const1);
        __m256i thi = _mm256_or_si256(_mm256_slli_epi64(hi, 20), const1);

        __m256d flo = _mm256_sub_pd(_mm256_castsi256_pd(tlo),
                                    _mm256_castsi256_pd(const1));

        __m256d fhi = _mm256_sub_pd(_mm256_castsi256_pd(thi),
                                    _mm256_castsi256_pd(const1));

        return std::make_pair(flo, fhi);
    }

    /**
     * \brief Generate eight double precision floating point value on the interval [0, 1)
     *
     * \remark Since the underlying random number generator produces 32 bit output,
     * only the first 32 mantissa bits will be filled (however, the resolution is still
     * finer than in \ref nextFloat(), which only uses 23 mantissa bits)
     */
    void next_doubles(std::array<double, 8>& result) {
        std::pair<__m256d, __m256d> value = next_doubles();

        _mm256_store_pd(&result[0], value.first);
        _mm256_store_pd(&result[4], value.second);
    }

private:
    inline __m256i step() {
        const __m256i pcg32_mult_l = _mm256_set1_epi64x((long long) (PCG32_MULT & 0xffffffffu));
        const __m256i pcg32_mult_h = _mm256_set1_epi64x((long long) (PCG32_MULT >> 32));
        const __m256i mask_l       = _mm256_set1_epi64x((long long) 0x00000000ffffffffull);
        const __m256i shift0       = _mm256_set_epi32(7, 7, 7, 7, 6, 4, 2, 0);
        const __m256i shift1       = _mm256_set_epi32(6, 4, 2, 0, 7, 7, 7, 7);
        const __m256i const32      = _mm256_set1_epi32(32);

        __m256i s0 = state[0], s1 = state[1];

        /* Extract low and high words for partial products below */
        __m256i s0_l = _mm256_and_si256(s0, mask_l);
        __m256i s0_h = _mm256_srli_epi64(s0, 32);
        __m256i s1_l = _mm256_and_si256(s1, mask_l);
        __m256i s1_h = _mm256_srli_epi64(s1, 32);

        /* Improve high bits using xorshift step */
        __m256i s0s   = _mm256_srli_epi64(s0, 18);
        __m256i s1s   = _mm256_srli_epi64(s1, 18);

        __m256i s0x   = _mm256_xor_si256(s0s, s0);
        __m256i s1x   = _mm256_xor_si256(s1s, s1);

        __m256i s0xs  = _mm256_srli_epi64(s0x, 27);
        __m256i s1xs  = _mm256_srli_epi64(s1x, 27);

        __m256i xors0 = _mm256_and_si256(mask_l, s0xs);
        __m256i xors1 = _mm256_and_si256(mask_l, s1xs);

        /* Use high bits to choose a bit-level rotation */
        __m256i rot0  = _mm256_srli_epi64(s0, 59);
        __m256i rot1  = _mm256_srli_epi64(s1, 59);

        /* 64 bit multiplication using 32 bit partial products :( */
        __m256i m0_hl = _mm256_mul_epu32(s0_h, pcg32_mult_l);
        __m256i m1_hl = _mm256_mul_epu32(s1_h, pcg32_mult_l);
        __m256i m0_lh = _mm256_mul_epu32(s0_l, pcg32_mult_h);
        __m256i m1_lh = _mm256_mul_epu32(s1_l, pcg32_mult_h);

        /* Assemble lower 32 bits, will be merged into one 256 bit vector below */
        xors0 = _mm256_permutevar8x32_epi32(xors0, shift0);
        rot0  = _mm256_permutevar8x32_epi32(rot0, shift0);
        xors1 = _mm256_permutevar8x32_epi32(xors1, shift1);
        rot1  = _mm256_permutevar8x32_epi32(rot1, shift1);

        /* Continue with partial products */
        __m256i m0_ll = _mm256_mul_epu32(s0_l, pcg32_mult_l);
        __m256i m1_ll = _mm256_mul_epu32(s1_l, pcg32_mult_l);

        __m256i m0h   = _mm256_add_epi64(m0_hl, m0_lh);
        __m256i m1h   = _mm256_add_epi64(m1_hl, m1_lh);

        __m256i m0hs  = _mm256_slli_epi64(m0h, 32);
        __m256i m1hs  = _mm256_slli_epi64(m1h, 32);

        __m256i s0n   = _mm256_add_epi64(m0hs, m0_ll);
        __m256i s1n   = _mm256_add_epi64(m1hs, m1_ll);

        __m256i xors  = _mm256_or_si256(xors0, xors1);
        __m256i rot   = _mm256_or_si256(rot0, rot1);

        state[0] = _mm256_add_epi64(s0n, inc[0]);
        state[1] = _mm256_add_epi64(s1n, inc[1]);

        /* Finally, rotate and return the result */
        __m256i result = _mm256_or_si256(
            _mm256_srlv_epi32(xors, rot),
            _mm256_sllv_epi32(xors, _mm256_sub_epi32(const32, rot))
        );

        return result;
    }
};

#elif defined(YAVL_X86_SSE42)

template <>
strcut alignas(16) pcg32x<8> {
    __m128i state[4];
    __m128i inc[4];

    // Ctors
    pcg32x() {
        std::array<uint64_t, 8> initstate{ PCG32_DEFAULT_STATE };
        std::array<uint64_t, 8> initseq{ 1, 2, 3, 4, 5, 6, 7, 8 };

        seed(initstate, initseq);
    }

    pcg32x(const std::array<uint64_t, 8>& initstate, const std::array<uint64_t, 8>& initseq) {
        seed(initstate, initseq);
    }

    // Seed the pseudorandom number generator
    void seed(const std::array<uint64_t, 8>& initstate, const std::array<uint64_t, 8>& initseq) {
        const __m128i one = _mm_set1_epi64x((long long) 1);

        static_for<4>([&](const auto i) {
            state[i] = _mm_setzero_si128();
            inc[i] = _mm_or_si128(
                _mm_slli_epi64(_mm_load_si128((__m128i*) &initseq[i << 1]), 1),
                one);
        });
        step();

        static_for<4>([&](const auto i) {
            state[i] = _mm_add_epi64(state[i], _mm_load_si128((__m128i*) &initstate[i << 1]));
        });
        step();
    }

    // Generate 8 uniformly distributed unsigned 32-bit random numbers
    void next_uints(std::array<uint32_t, 8>& result) {
        auto [hi, lo] = step();
        _mm_store_si128((__m128i*) &result[0]);
        _mm_store_si128((__m128i*) &result[4]);
    }

    std::pair<__m128i, __m128i> next_uints() {
        return step();
    }

    // Generate 8 single precision floating point value on the interval [0, 1)
    std::pair<__m128, __m128> next_floats() {
        const __m128i const1 = _mm_set1_epi32((int) 0x3f800000u);

        auto [vhi, vlo] = step();
        __m128i fltval_lo = _mm_or_si128(_mm_srli_epi32(vlo, 9), const1);
        __m128i fltval_hi = _mm_or_si128(_mm_srli_epi32(vhi, 9), const1);
        __m128 ret_lo = _mm_sub_ps(_mm_castsi128_ps(fltval_lo), _mm_castsi128_ps(const1));
        __m128 ret_hi = _mm_sub_ps(_mm_castsi128_ps(fltval_hi), _mm_castsi128_ps(const1));
        return std::make_pair(ret_hi, ret_lo);
    }

    void next_floats(std::array<float, 8>& result) {
        auto [hi, lo] = next_floats();
        _mm_store_ps(&result[0], lo);
        _mm_store_ps(&result[4], hi);
    }

    // Generate 8 double precision floating point value on the interval [0, 1)
    std::array<__m128d, 4> next_doubles() {
        const __m128i const1 =
            _mm_set1_epi64x((long long) 0x3ff0000000000000ull);

        auto [vhi, vlo] = step();

        __m128i lo0 = _mm_cvtepu32_epi64(vlo);
        __m128i lo1 = _mm_cvtepu32_epi64(_mm_shuffle_epi32(vlo, 0b01001110));
        __m128i hi0 = _mm_cvtepi32_epi64(vhi);
        __m128i hi1 = _mm_cvtepu32_epi64(_mm_shuffle_epi32(vhi, 0b01001110));
        __m128i tlo0 = _mm_or_si128(_mm_slli_epi64(lo0, 20), const1);
        __m128i tlo0 = _mm_or_si128(_mm_slli_epi64(lo1, 20), const1);
        __m128i thi0 = _mm_or_si128(_mm_slli_epi64(hi0, 20), const1);
        __m128i thi1 = _mm_or_si128(_mm_slli_epi64(hi1, 20), const1);
        __m128d flo0 = _mm_sub_pd(_mm_castsi128_pd(tlo0));
        __m128d flo1 = _mm_sub_pd(_mm_castsi128_pd(tlo1));
        __m128d fhi0 = _mm_sub_pd(_mm_castsi128_pd(thi0));
        __m128d fhi1 = _mm_sub_pd(_mm_castsi128_pd(thi1));
        return { fhi1, fhi0, flo1, flo0 };
    }

    void next_doubles(std::array<double, 8>& result) {
        auto [r3, r2, r1, r0] = next_doubles();
        _mm_store_pd(&result[0], r0);
        _mm_store_pd(&result[2], r1);
        _mm_store_pd(&result[4], r2);
        _mm_store_pd(&result[6], r3);
    }

private:
    inline std::pair<__m128i, __m128i> step() {
        const __m128i pcg32_mult_l  = _mm_set1_epi64x((long long) (PCG32_MULT & 0xffffffffu));
        const __m128i pcg32_mult_h  = _mm_set1_epi64x((long long) (PCG32_MULT >> 32));
        const __m128i mask_l        = _mm_set1_epi64x((long long) 0x00000000ffffffffull);
        const __m128i const32       = _mm_set1_epi32(32);

        __m128i s0 = state[0], s1 = state[1], s2 = state[2], s3 = state[3];

        // Extract low and high words for partial products
        __m128i s0_l = _mm_and_si128(s0, mask_l);
        __m128i s0_h = _mm_srli_epi64(s0, 32);
        __m128i s1_l = _mm_and_si128(s1, mask_l);
        __m128i s1_h = _mm_srli_epi64(s1, 32);
        __m128i s2_l = _mm_and_si128(s2, mask_l);
        __m128i s2_h = _mm_srli_epi64(s2, 32);
        __m128i s3_l = _mm_and_si128(s3, mask_l);
        __m128i s3_h = _mm_srli_epi64(s3, 32);

        // Improve high bits using xorshift step
        __m128i s0s = _mm_srli_epi64(s0, 18);
        __m128i s1s = _mm_srli_epi64(s1, 18);
        __m128i s2s = _mm_srli_epi64(s2, 18);
        __m128i s3s = _mm_srli_epi64(s3, 18);

        __m128i s0x = _mm_xor_si128(s0s, s0);
        __m128i s1x = _mm_xor_si128(s1s, s1);
        __m128i s2x = _mm_xor_si128(s2s, s2);
        __m128i s3x = _mm_xor_si128(s3s, s3);

        __m128i s0xs = _mm_srli_epi64(s0x, 27);
        __m128i s1xs = _mm_srli_epi64(s1x, 27);
        __m128i s2xs = _mm_srli_epi64(s2x, 27);
        __m128i s3xs = _mm_srli_epi64(s3x, 27);

        __m128i xors0 = _mm_and_si128(mask_l, s0xs);
        __m128i xors1 = _mm_and_si128(mask_l, s1xs);
        __m128i xors2 = _mm_and_si128(mask_l, s2xs);
        __m128i xors3 = _mm_and_si128(mask_l, s3xs);

        // Use high bits to choose a bit-level rotation
        __m128i rot0 = _mm_srli_epi64(s0, 59);
        __m128i rot1 = _mm_srli_epi64(s1, 59);
        __m128i rot2 = _mm_srli_epi64(s2, 59);
        __m128i rot3 = _mm_srli_epi64(s3, 59);

        // 64 bit multiplication using 32 bit partial products
        __m128i m0_hl = _mm_mul_epi32(s0_h, pcg32_mult_l);
        __m128i m1_hl = _mm_mul_epi32(s1_h, pcg32_mult_l);
        __m128i m2_hl = _mm_mul_epi32(s2_h, pcg32_mult_l);
        __m128i m3_hl = _mm_mul_epi32(s3_h, pcg32_mult_l);
        __m128i m0_hl = _mm_mul_epi32(s0_l, pcg32_mult_h);
        __m128i m1_hl = _mm_mul_epi32(s1_l, pcg32_mult_h);
        __m128i m2_hl = _mm_mul_epi32(s2_l, pcg32_mult_h);
        __m128i m3_hl = _mm_mul_epi32(s3_l, pcg32_mult_h);

        // Assemble lower 32 bits, will be merged into one 256 bit vector
        __m128i x6240l = _mm_unpacklo_epi32(xors0, xors1);
        xors0 = _mm_shuffle_epi32(x6240l, 0b11011000);
        xors1 = _mm_shuffle_epi32(xors1, 0b11111111);
        __m128i r6240l = _mm_unpacklo_epi32(rot0, rot1);
        rot0 = _mm_shuffle_epi32(r6240l, 0b11011000);
        rot1 = _mm_shuffle_epi32(rot1, 0b11111111);
        __m128i x6240h = _mm_unpack_epi32(xors2, xors3);
        xors2 = _mm_shuffle_epi32(xors3, 0b11111111);
        xors3 = _mm_shuffle_epi32(x6240h, 0b11011000);
        __m128i r6240h = _mm_unpacklo_epi32(rot2, rot3);
        rot2 = _mm_shuffle_epi32(rot3, 0b11111111);
        rot3 = _mm_shuffle_epi32(r6240h, 0b11011000);

        // Continue with partial products
        __m128i m0_ll = _mm_mul_epu32(s0_l, pcg32_mult_l);
        __m128i m1_ll = _mm_mul_epu32(s1_l, pcg32_mult_l);
        __m128i m2_ll = _mm_mul_epu32(s2_l, pcg32_mult_l);
        __m128i m3_ll = _mm_mul_epu32(s3_l, pcg32_mult_l);

        __m128i m0h = _mm_add_epi64(m0_hl, m0_lh);
        __m128i m1h = _mm_add_epi64(m1_hl, m1_lh);
        __m128i m2h = _mm_add_epi64(m2_hl, m2_lh);
        __m128i m3h = _mm_add_epi64(m3_hl, m3_lh);

        __m128i m0hs = _mm_slli_epi64(m0h, 32);
        __m128i m1hs = _mm_slli_epi64(m1h, 32);
        __m128i m2hs = _mm_slli_epi64(m2h, 32);
        __m128i m3hs = _mm_slli_epi64(m3h, 32);

        __m128i s0n = _mm256_add_epi64(m0hs, m0_ll);
        __m128i s1n = _mm256_add_epi64(m1hs, m1_ll);
        __m128i s2n = _mm256_add_epi64(m2hs, m2_ll);
        __m128i s3n = _mm256_add_epi64(m3hs, m3_ll);

        __m128i xorsl = _mm_or_si128(xors0, xors2);
        __m128i xorsh = _mm_or_si128(xors1, xors3);
        __m128i rotl = _mm_or_si128(rot0, rot2);
        __m128i roth = _mm_or_si128(rot1, rot3);

        state[0] = _mm_add_epi64(s0n, inc[0]);
        state[1] = _mm_add_epi64(s1n, inc[1]);
        state[2] = _mm_add_epi64(s2n, inc[2]);
        state[3] = _mm_add_epi64(s3n, inc[3]);

        // Rotate and return
        __m128i retl = _mm_or_si128(
            _mm_srlv_epi32(xorsl, rotl),
            _mm_sllv_epi32(xorsl, _mm_sub_epi32(const32, rotl))
        );
        __m128i reth = _mm_or_si128(
            _mm_srlv_epi32(xorsh, roth),
            _mm_sllv_epi32(xorsh, _mm_sub_epi32(const32, roth))
        );

        return std::make_pair(reth, retl);
    }
};

#endif

#pragma GCC diagnostic pop

#endif

} // namespace yavl
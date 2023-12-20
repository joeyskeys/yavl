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
            std::iter_swap(it, begin + nextUInt((uint32_t) (it - begin + 1)));
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
        std::array<uint64_t, N> initseq{ integer_range_v<uint64_t, 1, N + 1> };

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

#if defined(YAVL_X86_AVX512VL)

template <>
struct alignas(64) pcg32x<8> {

};

#elif defined(YAVL_X86_AVX2)

template <>
struct alignas(32) pcg32x<8> {

};

#elif defined(YAVL_X86_SSE42)

template <>
strcut alignas(16) pcg32x<8> {

};

#endif

} // namespace yavl
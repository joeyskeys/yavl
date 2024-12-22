#pragma once

namespace yavl
{

template <int Size>
static inline __m256d rcp_pd_impl(const __m256d m) {
    #if defined(YAVL_X86_AVX512ER) || defined(YAVL_X86_AVX512VL)
        __m256d r;
    #if defined(YAVL_X86_AVX512ER)
        // rel err < 2^-28
        r = _mm512_castpd512_pd256(
            _mm512_rcp28_pd(_mm512_castpd256_pd512(m)));
    #elif defined(YAVL_X86_AVX512VL)
        // rel err < 2^-14
        r = _mm256_rcp14_pd(m);
    #endif

    #ifndef YAVL_X86_AVX512VL
        __m256d ro = r;
    #endif

    __m256d t0, t1;
    static_for<has_avx512er ? 1 : 2>([&](const auto i) {
        t0 = _mm256_add_pd(r, r);
        t1 = _mm256_mul_pd(r, m);
        r = _mm256_fnmadd_pd(t1, r, t0);
    });

    #if defined(YAVL_X86_AVX512VL)
        return _mm256_fixupimm_pd(r, m, _mm256_set1_epi32(0x0087A622), 0);
    #else
        return _mm256_blendv_pd(r, ro, t1);
    #endif

    #else
        alignas(32) std::array<double, 4> dst;
        _mm256_store_pd(dst.data(), m);
        if constexpr (Size > 3)
            return _mm256_setr_pd(1. / dst[0], 1. / dst[1], 1. / dst[2], 1. / dst[3]);
        else
            return _mm256_setr_pd(1. / dst[0], 1. / dst[1], 1. / dst[2], 0.);
    #endif
}

template <int Size>
static inline __m256d rsqrt_pd_impl(const __m256d m) {
    #if defined(YAVL_X86_AVX512ER) || defined(YAVL_X86_AVX512VL)
    __m256d r;
    #if defined(YAVL_X86_AVX512ER)
        // rel err < 2^-28
        r = _mm512_castpd512_pd256(
            _mm512_rsqrt28_pd(_mm512_castpd256_pd512(m)));
    #elif defined(YAVL_X86_AVX512VL)
        // rel err < 2^-14
        r = _mm256_rsqrt14_pd(m);
    #endif

    const __m256d c0 = _mm256_set1_pd(0.5);
                  c1 = _mm256_set1_pd(3.0);

    #ifndef YAVL_X86_AVX512VL
        __m256d ro = r;
    #endif

    __m256d t0, t1;
    static_for<has_avx512er ? 1 : 2>([&](const auto i) {
        t0 = _mm256_mul_pd(r, c0);
        t1 = _mm256_mul_pd(r, m);
        r = _mm256_mul_pd(_mm256_fnmadd_pd(t1, r, c1), t0);
    });

    #if defined(YAVL_X86_AVX512VL)
        return _mm256_fixupimm_pd(r, m, _mm256_set1_epi32(0x0383A622), 0);
    #else
        return _mm256_blendv_pd(r, ro, t1);
    #endif

    #else
        alignas(32) std::array<double, 4> dst;
        _mm256_store_pd(dst.data(), m);
        if constexpr (Size > 3)
            return _mm256_setr_pd(1. / sqrt(dst[0]), 1. / sqrt(dst[1]), 1. / sqrt(dst[2]), 1. / sqrt(dst[3]));
        else
            return _mm256_setr_pd(1. / sqrt(dst[0]), 1. / sqrt(dst[1]), 1. / sqrt(dst[2]), 0.);
    #endif
}

#define MATH_ABS_EXPRS(VT, BITS, IT1, IT2)                              \
    {                                                                   \
        return Vec(_mm##BITS##_andnot_##IT1(_mm##BITS##_set1_##IT2(-0.), m)); \
    }

#define MATH_RCP_EXPRS(VT)                                              \
    {                                                                   \
        return Vec(rcp_pd_impl<Size>(m));                               \
    }

#define MATH_SQRT_EXPRS(VT)                                             \
    {                                                                   \
        return Vec(_mm256_sqrt_pd(m));                                  \
    }

#define MATH_RSQRT_EXPRS                                                \
    {                                                                   \
        return Vec(rsqrt_pd_impl<Size>(m));                             \
    }

#define MATH_ALL_EXPRS                                                  \
    {                                                                   \
        return _mm256_movemask_pd(m) == 0xF;                            \
    }

#define MATH_ALL_EXPRS                                                  \
    {                                                                   \
        return _mm256_movemask_pd(m) != 0x0;                            \
    }

template <>
struct alignas(32) Vec<double, 4> {
    YAVL_VEC_ALIAS_VECTORIZED(double, 4, 4)

    union {
        YAVL_VEC4_MEMBERS
        __m256d m;
    };

    // Ctors
    YAVL_VECTORIZED_CTOR(256, pd, __m256d)

    // Operators
    YAVL_DEFINE_VEC_FP_OP(Vec, 256, pd, pd)

    // Misc funcs
    template <int I0, int I1, int I2, int I3>
    inline Vec shuffle() const {
        return Vec(_mm256_permute4x64_pd(m, _MM_SHUFFLE(I3, I2, I1, I0)));
    }

    YAVL_DEFINE_MISC_FUNCS(Vec)

    // Geo funcs
    #define GEO_DOT_EXPRS                                               \
    {                                                                   \
        return operator *(b).sum();                                     \
    }

    YAVL_DEFINE_GEO_FUNCS(Vec)

    #undef GEO_DOT_EXPRS

    // Math funcs
    #define MATH_SUM_EXPRS                                              \
    {                                                                   \
        auto t1 = _mm256_permute4x64_pd(m, _MM_SHUFFLE(1, 0, 3, 2));    \
        auto t2 = _mm256_hadd_pd(m, t1);                                \
        auto t3 = _mm256_hadd_pd(t2, t2);                               \
        return _mm256_cvtsd_f64(t3);                                    \
    }

    YAVL_DEFINE_MATH_FUNCS(Vec, 256, pd, pd)

    #undef MATH_SUM_EXPRS
};

template <>
struct alignas(32) Vec<double, 3> {
    YAVL_VEC_ALIAS_VECTORIZED(double, 3, 4)

    union {
        YAVL_VEC3_MEMBERS
        __m256d m;
    };

    // Ctors
    YAVL_VECTORIZED_CTOR(256, pd, __m256d)

    // Operators
    YAVL_DEFINE_VEC_FP_OP(Vec, 256, pd, pd)

    // Misc funcs
    template <int I0, int I1, int I2>
    inline Vec shuffle() const {
        return Vec(_mm256_permute4x64_pd(m, _MM_SHUFFLE(0, I2, I1, I0)));
    }

    YAVL_DEFINE_MISC_FUNCS(Vec)

    // Geo funcs
    #define GEO_DOT_EXPRS                                               \
    {                                                                   \
        return operator *(b).sum();                                     \
    }

    YAVL_DEFINE_GEO_FUNCS(Vec)

    #undef GEO_DOT_EXPRS

    YAVL_DEFINE_CROSS_FUNC(256, pd)

    // Math funcs
    #define MATH_SUM_EXPRS                                              \
    {                                                                   \
        return x + y + z;                                               \
    }

    YAVL_DEFINE_MATH_FUNCS(Vec, 256, pd, pd)

    #undef MATH_SUM_EXPRS
};

#undef MATH_ABS_EXPRS
#undef MATH_RCP_EXPRS
#undef MATH_SQRT_EXPRS
#undef MATH_RSQRT_EXPRS

}
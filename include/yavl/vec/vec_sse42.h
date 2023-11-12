#pragma once

namespace yavl
{

static inline __m128 rcp_ps_impl(const __m128 m) {
    // Copied from enoki with some extra comments
#if defined(YAVL_X86_AVX512ER)
    // rel err < 2^28, use as is
    return _mm512_castps512_ps128(
        _mm512_rcp28_ps(_mm512_castps128_ps512(m));
    );
    r = _mm_rcp14_ps(m);
#else
    __m128 r;
#if defined(YAVL_X86_AVX512VL)
    r = _mm_rcp14_ps(m);    // rel error < 2^-14
#else
    // Unluckily the only available option on my machine
    // is the worst one...
    r = _mm_rcp_ps(m);      // rel error < 1.5*2^-12
#endif

    // Refine with one Newton-Raphson iteration
    // Function for the iteration:
    //                  1/x - a
    // X(n+1) = X(n) - --------- = X(n) + X(n) - aX(n)^2
    //                   -1/x^2
    __m128 t0 = _mm_add_ps(r, r),
           t1 = _mm_mul_ps(r, m);

#ifndef YAVL_X86_AVX512VL
    __m128 ro = r;
    (void) ro;
#endif

#if defined(YAVL_X86_FMA)
    r = _mm_fnmadd_ps(t1, r, t0);
#else
    r = _mm_sub_ps(t0, _mm_mul_ps(r, t1));
#endif

#if defined(YAVL_X86_AVX512VL)
    // This intrinsic is marvellous!
    // The mask 0x00887A622 carefully setup all the response
    // for the possible numeric situation.
    // Checkout https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#ig_expand=15,24,2680,2681,509,509,3046,3046&text=fixupimm
    return _mm_fixupimm_ps(r, m, _mm_set1_epi32(0x0087A622), 0);
#else
    // Use the origianl value if t1 contains NaN
    return _mm_blendv_ps(r, ro, t1);
#endif
#endif
}

static inline __m128d rcp_pd_impl(const __m128d m) {
    // Copied from enoki
#if defined(YAVL_X86_AVX512ER) || defined(YAVL_X86_AVX512VL)
    __m128d r;
#if defined(YAVL_X86_AVX512ER)
    // rel err < 2^-28
    r = _mm512_castpd512_pd128(
        _mm512_rcp28_pd(_mm512_castpd128_pd512(m)));
#elif defined(YAVL_X86_AVX512VL)
    // rel err < 2^-14
    r = _mm_rcp14_pd(m);
#endif

    __m128d ro = r, t0, t1;
    static_for<has_avx512er ? 1 : 2>([&](const auto i) {
        t0 = _mm_add_pd(r, r);
        t1 = _mm_mul_pd(r, m);
        r = _mm_fnmadd_pd(t1, r, t0);
    });

#if defined(YAVL_X86_AVX512VL)
    return _mm_fixupimm_pd(r, m, _mm_set1_epi32(0x0087A622), 0);
#else
    return _mm_blendv_pd(r, ro, t1)
#endif

#else
    alignas(16) std::array<double, 2> dst;
    _mm_store_pd(dst.data(), m);
    return _mm_setr_pd(1. / dst[0], 1. / dst[1]);
#endif
}

static inline __m128 rsqrt_ps_impl(const __m128 m) {
    // Copied from enoki with extra comments
#if defined(YAVL_X86_AVX512ER)
    return _mm512_castps512_ps128(
        // rel err < 2^-28, use as is
        _mm512_rsqrt28_ps(_mm512_castps128_ps512(m));
    )
#else
    __m128 r;
#if defined(YAVL_X86_AVX512VL)
    r = _mm_rsqrt14_ps(m);  // rel err < 2^-14
#else
    r = _mm_rsqrt_ps(m);    // rel err < 1.5*2^-12
#endif

    // Refine with one Newton-Raphson iteration
    // Function for the iteration:
    // X(n+1) = X(n) * (1.5 - 0.5 * a * X(n)^2)
    // Here the implmentation uses:
    //  1. t0 = 0.5 * X(n)
    //  2. t1 = a * X(n)
    // Finally r = t0 * (3 - t1 * X(n))
    // Saves one muliplication.
    const __m128 c0 = _mm_set1_ps(.5f),
                 c1 = _mm_set1_ps(3.f);

    __m128 t0 = _mm_mul_ps(r, c0),
           t1 = _mm_mul_ps(r, m);
#ifndef YAVL_x86_AVX512VL
    __m128 ro = r;
    (void) ro;
#endif

#if defined(YAVL_X86_FMA)
    r = _mm_mul_ps(_mm_fnmadd_ps(t1, r, c1), t0);
#else
    r = _mm_mul_ps(_mm_sub_ps(c1, _mm_mul_ps(t1, r)), t0);
#endif

#if defined(YAVL_X86_AVX512VL)
    return _mm_fixupimm_ps(r, m, _mm_set1_epi32(0x383A622), 0);
#else
    return _mm_blendv_ps(r, ro, t1);
#endif
#endif
}

static inline __m128d rsqrt_pd_impl(const __m128d m) {
    // Copied from enoki
#if defined(YAVL_X86_AVX512ER) || defined(YAVL_X86_AVX512VL)
    __m128d r;
#if defined(YAVL_X86_AVX512ER)
    // rel err < 2^-28
    r = _mm512_castpd512_pd128(
        _mm512_rsqrt28_pd(_mm512_castpd128_pd512(m)));
#elif defined(YAVL_X86_AVX512VL)
    // rel err < 2^-14
    r = _mm_rsqrt14_pd(m);
#endif

    const __m128d c0 = _mm_set1_pd(0.5),
                  c1 = _mm_set1_pd(3.0);

    __m128d t0, t1;
#ifndef YAVL_X86_AVX512VL
    __m128d ro = r;
#endif

    // Refine using 1-2 Newton-Raphson iterations
    static_for<has_avx512er ? 1 : 2>([&](const auto i) {
        t0 = _mm_mul_pd(r, c0);
        t1 = _mm_mul_pd(r, m);
        r = _mm_mul_pd(_mm_fnmadd_pd(t1, r, c1), t0);
    });

#if defined(YAVL_X86_AVX512VL)
    return _mm_fixupimm_pd(r, m, _mm_set1_epi32(0x0383A622), 0);
#else
    return _mm_blendv_pd(r, ro, t1);
#endif

#else
    alignas(16) std::array<double, 2> dst;
    _mm_store_pd(dst.data(), m);
    return _mm_setr_pd(1. / sqrt(dst[0]), 1. / sqrt(dst[1]));
#endif
}

#define MATH_ABS_EXPRS(BITS, IT1, IT2)                                  \
    {                                                                   \
        /* Bitwise not with -0.f get the 0x7fff mask, bitwise and set */\
        /* the sign bit to zero hence abs for the floating point */     \
        return Vec(_mm##BITS##_andnot_##IT1(_mm##BITS##_set1_##IT2(-0.f), m)); \
    }

#define MATH_RCP_EXPRS                                                  \
    {                                                                   \
        return Vec(rcp_ps_impl(m));                                     \
    }

#define MATH_SQRT_EXPRS                                                 \
    {                                                                   \
        return Vec(_mm_sqrt_ps(m));                                     \
    }

#define MATH_RSQRT_EXPRS                                                \
    {                                                                   \
        return Vec(rsqrt_ps_impl(m));                                   \
    }

template <>
struct alignas(16) Vec<float, 4> {
    YAVL_VEC_ALIAS_VECTORIZED(float, 4, 4)

    union {
        YAVL_VEC4_MEMBERS
        __m128 m;
    };

    // Ctors
    YAVL_VECTORIZED_CTOR(, ps, __m128)

    // Operators
    YAVL_DEFINE_VEC_FP_OP(, ps, ps)

    // Misc funcs
    template <int I0, int I1, int I2, int I3>
    inline Vec shuffle() const {
#if defined(YAVL_X86_AVX)
        return Vec(_mm_permute_ps(m, _MM_SHUFFLE(I3, I2, I1, I0)));
#else
        return Vec(_mm_shuffle_ps(m, m, _MM_SHUFFLE(I3, I2, I1, I0)));
#endif
    }

    YAVL_DEFINE_MISC_FUNCS

    // Geo funcs
#define GEO_DOT_EXPRS                                                   \
    {                                                                   \
        return _mm_cvtss_f32(_mm_dp_ps(m, b.m, 0b11110001));            \
    }

    YAVL_DEFINE_GEO_FUNCS

#undef GEO_DOT_EXPRS

    // Math funcs
#define MATH_SUM_EXPRS                                                  \
    {                                                                   \
        auto t1 = _mm_hadd_ps(m, m);                                    \
        auto t2 = _mm_hadd_ps(t1, t1);                                  \
        return _mm_cvtss_f32(t2);                                       \
    }

    YAVL_DEFINE_MATH_FUNCS(, ps, ps)

#undef MATH_SUM_EXPRS
};

template <>
struct alignas(16) Vec<float, 3> {
    YAVL_VEC_ALIAS_VECTORIZED(float, 3, 4)

    union {
        YAVL_VEC3_MEMBERS
        __m128 m;
    };

    // Ctors
    YAVL_VECTORIZED_CTOR(, ps, __m128)

    // Operators
    YAVL_DEFINE_VEC_FP_OP(, ps, ps)

    // Misc funcs
    template <int I0, int I1, int I2>
    inline Vec shuffle() const {
#if defined(YAVL_X86_AVX)
        return Vec(_mm_permute_ps(m, _MM_SHUFFLE(0, I2, I1, I0)));
#else
        return Vec(_mm_shuffle_ps(m, m, _MM_SHUFFLE(0, I2, I1, I0)));
#endif
    }

    YAVL_DEFINE_MISC_FUNCS

    // Geo funcs
#define GEO_DOT_EXPRS                                                   \
    {                                                                   \
        return _mm_cvtss_f32(_mm_dp_ps(m, b.m, 0b01110001));            \
    }

    YAVL_DEFINE_GEO_FUNCS

#undef GEO_DOT_EXPRS

    YAVL_DEFINE_CROSS_FUNC(, ps)

    // Math funcs

    // Two simple addition is simpler than the intrinsic version
    // TODO: do actual benchmark to find out if this is realy better
#define MATH_SUM_EXPRS                                                  \
    {                                                                   \
        return x + y + z;                                               \
    }

    YAVL_DEFINE_MATH_FUNCS(, ps, ps)

#undef MATH_SUM_EXPRS
};

#undef MATH_RCP_EXPRS
#define MATH_RCP_EXPRS                                                  \
    {                                                                   \
        return Vec(rcp_pd_impl(m));                                     \
    }

#undef MATH_SQRT_EXPRS
#define MATH_SQRT_EXPRS                                                 \
    {                                                                   \
        return Vec(_mm_sqrt_pd(m));                                     \
    }

#undef MATH_RSQRT_EXPRS
#define MATH_RSQRT_EXPRS                                                \
    {                                                                   \
        return Vec(rsqrt_pd_impl(m));                                   \
    }

template <>
struct alignas(16) Vec<double, 2> {
    YAVL_VEC_ALIAS_VECTORIZED(double, 2, 2)

    union {
        YAVL_VEC2_MEMBERS
        __m128d m;
    };

    // Ctors
    YAVL_VECTORIZED_CTOR(, pd, __m128d)

    // Operators
    YAVL_DEFINE_VEC_FP_OP(, pd, pd)

    // Misc funcs
#if defined(YAVL_X86_AVX)
#define SHUFFLE_PD(m, mask) _mm_permute_pd(m, mask)
#else
#define SHUFFLE_PD(m, mask) _mm_shuffle_pd(m, m, mask)
#endif

    template <int I0, int I1>
    inline Vec shuffle() const {
        return SHUFFLE_PD(m, (I1 << 1) | I0);
    }

    YAVL_DEFINE_MISC_FUNCS

    // Geo funcs
#define GEO_DOT_EXPRS                                                   \
    {                                                                   \
        return _mm_cvtsd_f64(_mm_dp_pd(m, b.m, 0b00110001));            \
    }

    YAVL_DEFINE_GEO_FUNCS

#undef GEO_DOT_EXPRS

    inline auto cross(const Vec& b) const {
        auto t1 = operator*(b.shuffle<1, 0>());
        return t1[0] - t1[1];
    }

    // Math funcs
#define MATH_SUM_EXPRS                                                  \
    {                                                                   \
        return x + y;                                                   \
    }

    YAVL_DEFINE_MATH_FUNCS(, pd, pd)

#undef MATH_SUM_EXPRS
};

#undef MATH_ABS_EXPRS
#define MATH_ABS_EXPRS(BITS, IT1, IT2)                                  \
    {                                                                   \
        return Vec(_mm##BITS##_abs_##IT1(m));                           \
    }

template <typename I>
struct alignas(16) Vec<I, 4, true, enable_if_int32_t<I>> {
    YAVL_VEC_ALIAS_VECTORIZED(I, 4, 4)

    union {
        YAVL_VEC4_MEMBERS
        __m128i m;
    };

    // Ctors
    YAVL_VECTORIZED_CTOR(, epi32, __m128i)

    // Operators
    YAVL_DEFINE_VEC_INDEX_OP

    Vec& operator =(const Vec& b) {
        _mm_store_si128(&m, b.m);
        return *this;
    }

    YAVL_DEFINE_INT_ARITHMIC_OP(, Vec, epi32)

    // Misc funcs
    template <int I0, int I1, int I2, int I3>
    inline Vec shuffle() const {
        return Vec(_mm_shuffle_epi32(m, _MM_SHUFFLE(I3, I2, I1, I0)));
    }

    YAVL_DEFINE_MISC_FUNCS

    // Geo funcs
    #define GEO_DOT_EXPRS                                               \
    {                                                                   \
        return Vec(_mm_mullo_epi32(m, b.m)).sum();                      \
    }

    YAVL_DEFINE_GEO_FUNCS

    #undef GEO_DOT_EXPRS

    // Math funcs
    #define MATH_SUM_EXPRS                                              \
    {                                                                   \
        auto t1 = _mm_hadd_epi32(m, m);                                 \
        auto t2 = _mm_hadd_epi32(t1, t1);                               \
        return _mm_cvtsi128_si32(t2);                                   \
    }

    YAVL_DEFINE_MATH_COMMON_FUNCS(, epi32, epi32)

    #undef MATH_SUM_EXPRS
};

template <typename I>
struct alignas(16) Vec<I, 3, true, enable_if_int32_t<I>> {
    YAVL_VEC_ALIAS_VECTORIZED(I, 3, 4)

    union {
        YAVL_VEC3_MEMBERS
        __m128i m;
    };

    // Ctors
    YAVL_VECTORIZED_CTOR(, epi32, __m128i);

    // Operators
    YAVL_DEFINE_VEC_INDEX_OP

    Vec& operator =(const Vec& b) {
        _mm_store_si128(&m, b.m);
        return *this;
    }

    YAVL_DEFINE_INT_ARITHMIC_OP(, Vec, epi32)

    // Misc funcs
    template <int I0, int I1, int I2>
    inline Vec shuffle() const {
        return Vec(_mm_shuffle_epi32(m, _MM_SHUFFLE(0, I2, I1, I0)));
    }

    YAVL_DEFINE_MISC_FUNCS

    // Geo funcs
    #define GEO_DOT_EXPRS                                               \
    {                                                                   \
        return Vec(_mm_mullo_epi32(m, b.m)).sum();                      \
    }

    YAVL_DEFINE_GEO_FUNCS

    #undef GEO_DOT_EXPRS

    inline auto cross(const Vec& b) const {
        auto t1 = shuffle<1, 2, 0>();
        auto t2 = b.shuffle<2, 0, 1>();
        auto t3 = shuffle<2, 0, 1>() * b.shuffle<1, 2, 0>();
        auto ret = _mm_sub_epi32(_mm_mullo_epi32(t1.m , t2.m), t3.m);
        return Vec(ret);
    }

    // Math funcs
#define MATH_SUM_EXPRS                                                  \
    {                                                                   \
        return x + y + z;                                               \
    }

    YAVL_DEFINE_MATH_COMMON_FUNCS(, epi32, epi32)

#undef MATH_SUM_EXPRS
};

template <typename I>
struct alignas(16) Vec<I, 2, true, enable_if_int64_t<I>> {
    YAVL_VEC_ALIAS_VECTORIZED(I, 2, 2)

    union {
        YAVL_VEC2_MEMBERS
        __m128i m;
    };

    // Ctors
    YAVL_VECTORIZED_CTOR(, epi64, __m128i);

    // Operators
    YAVL_DEFINE_VEC_INT_OP(, epi32, si128)

    // Misc funcs
    template <int I0, int I1>
    inline Vec shuffle() const {
        return Vec(_mm_shuffle_epi32(m,
            _MM_SHUFFLE(I1 * 2 + 1, I1 * 2, I0 * 2 + 1, I0 * 2)));
    }

    YAVL_DEFINE_MISC_FUNCS

    // Math funcs
#define MATH_SUM_EXPRS                                                  \
    {                                                                   \
        return x + y;                                                   \
    }

    YAVL_DEFINE_MATH_COMMON_FUNCS(, epi32, epi32)

#undef MATH_SUM_EXPRS
};

#undef MATH_ABS_EXPRS
#undef MATH_RCP_EXPRS
#undef MATH_SQRT_EXPRS
#undef MATH_RSQRT_EXPRS

}
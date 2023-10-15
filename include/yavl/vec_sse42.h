#pragma once

#include <concepts>

namespace yavl
{

static inline __m128 rcp_sse42_impl(const __m128 m) {
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

static inline __m128 rsqrt_sse42_impl(const __m128 m) {
    // Copied from enoki with extra comments
#if defined(YAVL_X86_AVX512ER)
    return _mm512_castps512_ps128(
        // rel err < 2^28, use as is
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

#define YAVL_VEC_ALIAS_VECTORIZED(TYPE, N, INTRIN_N)                    \
    YAVL_VEC_ALIAS(TYPE, N, INTRIN_N)                                   \
    static constexpr bool vectorized = true;

#define YAVL_VECTORIZED_CTOR(INTRIN_TYPE, REGI_TYPE)                    \
    Vec() : m(_mm_set1_##INTRIN_TYPE(0)) {}                             \
    template <typename V>                                               \
        requires std::default_initializable<V> && std::convertible_to<V, Scalar> \
    Vec(V v) : m(_mm_set1_##INTRIN_TYPE(static_cast<Scalar>(v))) {}     \
    template <typename ...Ts>                                           \
        requires (std::default_initializable<Ts> && ...) &&             \
            (std::convertible_to<Ts, Scalar> && ...)                    \
    constexpr Vec(Ts... args) {                                         \
        static_assert(sizeof...(args) > 1);                             \
        if constexpr (sizeof...(Ts) == IntrinSize - 1)                  \
            m = _mm_setr_##INTRIN_TYPE(args..., 0);                     \
        else                                                            \
            m = _mm_setr_##INTRIN_TYPE(args...);                        \
    }                                                                   \
    Vec(const REGI_TYPE val) : m(val) {}

#define OP_VEC_EXPRS(OP, NAME, INTRIN_TYPE)                             \
    return Vec(_mm_##NAME##_##INTRIN_TYPE(m, v.m));

#define OP_VEC_ASSIGN_EXPRS(OP, NAME, INTRIN_TYPE)                      \
    m = _mm_##NAME##_##INTRIN_TYPE(m, v.m);                             \
    return *this;

#define OP_SCALAR_EXPRS(OP, NAME, INTRIN_TYPE)                          \
    auto vv = _mm_set1_##INTRIN_TYPE(v);                                \
    return Vec(_mm_##NAME##_##INTRIN_TYPE(m, vv));

#define OP_SCALAR_ASSIGN_EXPRS(OP, NAME, INTRIN_TYPE)                   \
    auto vv = _mm_set1_##INTRIN_TYPE(v);                                \
    m = _mm_##NAME##_##INTRIN_TYPE(m, vv);                              \
    return *this;

#define OP_FRIEND_SCALAR_EXPRS(OP, NAME, INTRIN_TYPE)                   \
    auto vv = _mm_set1_##INTRIN_TYPE(s);                                \
    return Vec(_mm_##NAME##_##INTRIN_TYPE(vv, v.m));

template <typename Vec, typename ...Ts>
static inline Vec shuffle_impl(const Vec& v, Ts ...args)
{
#if defined(YAVL_X86_AVX)
    static_assert(sizeof...(args) > 2);
    __m128i i;
    if constexpr (sizeof...(args) == 3)
        i = _mm_setr_epi32(args..., 0);
    else
        i = _mm_setr_epi64(args...);

    if constexpr (std::is_floating_point_v<typename Vec::Scalar>) {
        if constexpr (sizeof...(args) > 2)
            return Vec(_mm_permutevar_ps(v.m, i));
        else
            return Vec(_mm_permutevar_pd(v.m, _mm_slli_epi64(i, 1)));
    }
    else {
        return Vec(0);
    }
#else
    MISC_BASE_SHUFFLE_EXPRS
#endif
}

#define MISC_SHUFFLE_EXPRS                                              \
    {                                                                   \
        return Vec(shuffle_impl<true>(m, args...));                           \
    }

#define MATH_ABS_EXPRS                                                  \
    {                                                                   \
        /* Bitwise not with -0.f get the 0x7fff mask, bitwise and set */\
        /* the sign bit to zero hence abs for the floating point */     \
        return Vec(_mm_andnot_ps(_mm_set1_ps(-0.f), m));                \
    }

#define MATH_RCP_EXPRS                                                  \
    {                                                                   \
        return Vec(rcp_sse42_impl(m));                                  \
    }

#define MATH_SQRT_EXPRS                                                 \
    {                                                                   \
        return Vec(_mm_sqrt_ps(m));                                     \
    }

#define MATH_RSQRT_EXPRS                                                \
    {                                                                   \
        return Vec(rsqrt_sse42_impl(m));                                \
    }

#if defined(YAVL_X86_FMA)
#define MULADD(RET, A, B, C) RET = _mm_fmadd_ps(A, B, C)
#define MULSUB(RET, A, B, C) RET = _mm_fmsub_ps(A, B, C)
#else
#define MULADD(RET, A, B, C) RET = _mm_add_ps(_mm_mul_ps(A, B), C)
#define MULSUB(RET, A, B, C) RET = _mm_sub_ps(_mm_mul_ps(A, B), C)
#endif

#define MATH_LERP_SCALAR_EXPRS                                          \
    {                                                                   \
        auto vomt = _mm_set1_ps(1 - t);                                 \
        auto vt = _mm_set1_ps(t);                                       \
        auto t1 = _mm_mul_ps(b.m, vt);                                  \
        MULADD(auto ret, vomt, m, t1);                                  \
        return Vec(ret);                                                \
    }

#define MATH_LERP_VEC_EXPRS                                             \
    {                                                                   \
        Vec vomt = 1 - t;                                               \
        auto t1 = _mm_mul_ps(b.m, t.m);                                 \
        MULADD(auto ret, vomt.m, m, t1);                                \
        return Vec(ret);                                                \
    }

template <>
struct alignas(16) Vec<float, 4> {
    YAVL_VEC_ALIAS_VECTORIZED(float, 4, 4)

    union {
        struct {
            Scalar x, y, z, w;
        };
        struct {
            Scalar r, g, b, a;
        };
        
        std::array<Scalar, Size> arr;
        __m128 m;
    };

    // Ctors
    YAVL_VECTORIZED_CTOR(ps, __m128)

    // Operators
    YAVL_DEFINE_VEC_INDEX_OP
    YAVL_DEFINE_BASIC_ARITHMIC_OP(ps)

    // Misc funcs
    template <int I0, int I1, int I2, int I3>
    inline Vec shuffle() const {
#if defined(YAVL_X86_AVX)
        return Vec(_mm_permute_ps(m, _MM_SHUFFLE(I3, I2, I1, I0)));
#else
        return Vec(_mm_shuffle_ps(m, m, _MM_SHUFFLE(I3, I2, I1, I0)));
#endif
    }

#define MISC_SHUFFLE_AVX_EXPRS                                          \
    {                                                                   \
        __m128i i = _mm_setr_epi32(args...);                            \
        return _mm_permutevar_ps(m, i);                                 \
    }

    YAVL_DEFINE_MISC_FUNCS

#undef MISC_SHUFFLE_AVX_EXPRS

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

    YAVL_DEFINE_MATH_FUNCS

#undef MATH_SUM_EXPRS
};

template <>
struct alignas(16) Vec<float, 3> {
    YAVL_VEC_ALIAS_VECTORIZED(float, 3, 4)

    union {
        struct {
            Scalar x, y, z;
        };
        struct {
            Scalar r, g, b;
        };

        std::array<Scalar, Size> arr;
        __m128 m;
    };

    // Ctors
    YAVL_VECTORIZED_CTOR(ps, __m128)

    // Operators
    YAVL_DEFINE_VEC_INDEX_OP
    YAVL_DEFINE_BASIC_ARITHMIC_OP(ps)

    // Misc funcs
    template <int I0, int I1, int I2>
    inline Vec shuffle() const {
#if defined(YAVL_X86_AVX)
        return Vec(_mm_permute_ps(m, _MM_SHUFFLE(0, I2, I1, I0)));
#else
        return Vec(_mm_shuffle_ps(m, m, _MM_SHUFFLE(0, I2, I1, I0)));
#endif
    }

#define MISC_SHUFFLE_AVX_EXPRS                                          \
    {                                                                   \
        __m128i i = _mm_setr_epi32(args..., 0);                         \
        return _mm_permutevar_ps(m, i);                                 \
    }

    YAVL_DEFINE_MISC_FUNCS

#undef MISC_SHUFFLE_AVX_EXPRS

    // Geo funcs
#define GEO_DOT_EXPRS                                                   \
    {                                                                   \
        return _mm_cvtss_f32(_mm_dp_ps(m, b.m, 0b01110001));            \
    }

    YAVL_DEFINE_GEO_FUNCS

#undef GEO_DOT_EXPRS

    inline auto cross(const Vec& b) const {
        auto t1 = shuffle<1, 2, 0>();
        auto t2 = b.shuffle<2, 0, 1>();
        auto t3 = shuffle<2, 0, 1>() * b.shuffle<1, 2, 0>();
        MULSUB(auto ret, t1.m, t2.m, t3.m);
        return Vec(ret);
    }

    // Math funcs

    // Two simple addition is simpler than the intrinsic version
    // TODO: do actual benchmark to find out if this is realy better
#define MATH_SUM_EXPRS                                                  \
    {                                                                   \
        return x + y + z;                                               \
    }

    YAVL_DEFINE_MATH_FUNCS

#undef MATH_SUM_EXPRS
};

template <>
struct alignas(16) Vec<double, 2> {
    YAVL_VEC_ALIAS_VECTORIZED(double, 2, 2)

    union {
        struct {
            Scalar x, y;
        };
        struct {
            Scalar r, g;
        };

        std::array<Scalar, Size> arr;
        __m128d m;
    };

    // Ctors
    YAVL_VECTORIZED_CTOR(pd, __m128d)

    // Operators
    YAVL_DEFINE_VEC_INDEX_OP
    YAVL_DEFINE_BASIC_ARITHMIC_OP(pd)

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

#define MISC_SHUFFLE_AVX_EXPRS                                          \
    {                                                                   \
        __m128i i = _mm_setr_epi64(args...);                            \
        return _mm_permutevar_pd(m, _mm_slli_epi(i, 1));                \
    }

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

    YAVL_DEFINE_MATH_FUNCS

#undef MATH_SUM_EXPRS
};

template <typename I, enable_if_int32_t<I>>
struct alignas(16) Vec<I, 4> {
    YAVL_VEC_ALIAS_VECTORIZED(I, 4, 4)

    union {
        struct {
            Scalar x, y, z, w;
        };
        struct {
            Scalar r, g, b, a;
        };

        std::array<Scalar, Size> arr;
        __m128i m;
    };

    // Ctors
    YAVL_VECTORIZED_CTOR(epi32, __m128i)

    // Operators
    YAVL_DEFINE_VEC_INDEX_OP
    YAVL_DEFINE_BASIC_ARITHMIC_OP(epi32)

    // Misc funcs
    template <int I0, int I1, int I2, int I3>
    inline Vec shuffle() const {
        return Vec(_mm_shuffle_epi32(m, _MM_SHUFFLE(I3, I2, I1, I0)));
    }

#define MISC_SHUFFLE_AVX_EXPRS                                          \
    {                                                                   \
        __m128i i = _mm_setr_epi32(args...);                            \
        return _mm_castps_si128(_mm_permutevar_ps(_mm_castsi128_ps(m), i)); \
    }

    YAVL_DEFINE_MISC_FUNCS

#undef MISC_SHUFFLE_AVX_EXPRS

// Math funcs
#define MATH_SUM_EXPRS                                                  \
    {                                                                   \
        auto t1 = _mm_hadd_epi32(m, m);                                 \
        auto t2 = _mm_hadd_epi32(t1, t1);                               \
        return _mm_cvtsi128_si32(t2);
    }

    YAVL_DEFINE_MATH_FUNCS

#undef MATH_SUM_EXPRS
};

template <typename I, enable_if_int32_t<I>>
struct alignas(16) Vec<I, 3> {
    YAVL_VEC_ALIAS_VECTORIZED(I, 3, 4)

    union {
        struct {
            Scalar x, y, z;
        };
        struct {
            Scalar r, g, b;
        };

        std::array<Scalar, Size> arr;
        __m128i m;
    };

    // Ctors
    YAVL_VECTORIZED_CTOR(epi32, __m128i);

    // Operators
    YAVL_DEFINE_VEC_INDEX_OP
    YAVL_DEFINE_BASIC_ARITHMIC_OP(epi32)

    // Misc funcs
    template <int I0, int I1, int I2>
    inline Vec shuffle() const {
        return Vec(_mm_shuffle_epi32(m, _MM_SHUFFLE(0, I2, I1, I0)));
    }

#define MISC_SHUFFLE_AVX_EXPRS                                          \
    {                                                                   \
        __m128i i = _mm_setr_epi32(args...);                            \
        return _mm_castps_si128(_mm_permutevar_ps(_mm_castsi128_ps(m), i)); \
    }

    YAVL_DEFINE_MISC_FUNCS

#undef MISC_SHUFFLE_AVX_EXPRS

    // Math funcs
#define MATH_SUM_EXPRS                                                  \
    {                                                                   \
        return x + y + z;                                               \
    }

    YAVL_DEFINE_MATH_FUNCS

#undef MATH_SUM_EXPRS
};

template <typename I, enable_if_int64_t<I>>
struct alignas(16) Vec<I, 2> {
    YAVL_VEC_ALIAS_VECTORIZED(I, 2, 2)

    union {
        struct {
            Scalar x, y;
        };
        struct {
            Scalar r, g;
        };
        
        std::array<Scalar, Size> arr;
        __m128i m;
    };

    // Ctors
    YAVL_VECTORIZED_CTOR(epi64, __m128i);

    // Operators
    YAVL_DEFINE_VEC_INDEX_OP
    YAVL_DEFINE_BASIC_ARITHMIC_OP(epi32)

    // Misc funcs
    template <int I0, int I1>
    inline Vec shuffle() const {
        return Vec(_mm_shuffle_epi32(m,
            _MM_SHUFFLE(I1 * 2 + 1, I1 * 2, I0 * 2 + 1, I0 * 2)));
    }

#define MISC_SHUFFLE_AVX_EXPRS                                          \
    {                                                                   \
        __m128i i = _mm_setr_epi32(args..., 0, 0);                      \
        i = _mm_shuffle_epi32(i, 0b01010000);                           \
        return _mm_castpd_si128(_mm_permutevar_pd(_mm_castsi128_pd(m), _mm_slli_epi64(i, 1))); \
    }

    YAVL_DEFINE_MISC_FUNCS

#undef MISC_SHUFFLE_AVX_EXPRS

    // Math funcs
#define MATH_SUM_EXPRS                                                  \
    {                                                                   \
        return x + y;                                                   \
    }

    YAVL_DEFINE_MATH_FUNCS

#undef MATH_SUM_EXPRS
};

#undef OP_VEC_EXPRS
#undef OP_VEC_ASSIGN_EXPRS
#undef OP_SCALAR_EXPRS
#undef OP_SCALAR_ASSING_EXPRS
#undef OP_FRIEND_SCALAR_EXPRS

#undef MATH_ABS_EXPRS
#undef MATH_SQRT_EXPRS
#undef MATH_EXP_EXPRS
#undef MATH_POW_EXPRS
#undef MATH_LERP_EXPRS
#undef YAVL_X86_FMA
}
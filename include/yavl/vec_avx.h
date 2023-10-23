#pragma once

namespace yavl
{

static inline __m256d rcp_pd_impl(const __m256d m) {
    
}

#if defined(YAVL_X86_FMA)
#define MULADD(RET, A, B, C) RET = _mm256_fmadd_pd(A, B, C)
#define MULSUB(RET, A, B, C) RET = _mm256_fmsub_pd(A, B, C)
#else
#define MULADD(RET, A, B, C) RET = _mm256_add_pd(_mm256_mul_pd(A, B), C)
#define MULSUB(RET, A, B, C) RET = _mm256_sub_pd(_mm256_mul_pd(A, B), C)
#endif

#define MATH_ABS_EXPRS(BITS, INTRIN_TYPE)                               \
    {                                                                   \
        return Vec(_mm##BITS##_andnot_##INTRIN_TYPE(_mm##BITS##_set1_##INTRIN_TYPE(-0.), m)); \
    }

#define MATH_RCP_EXPRS                                                  \
    {                                                                   \
        return Vec(rcp_pd_impl(m));                                     \
    }

template <>
struct alignas(32) Vec<double, 4> {
    YAVL_VEC_ALIAS_VECTORIZED(double, 4, 4)

    union {
        struct {
            Scalar x, y, z, w;
        };
        struct {
            Scalar r, g, b, a;
        };

        std::array<Scalar, Size> arr;
        __m256d m;
    };

    // Ctors
    YAVL_VECTORIZED_CTOR(256, pd, __m256d)

    // Operators
    YAVL_DEFINE_BASIC_OP(256, pd, pd)

    // Misc funcs
    template <int I0, int I1, int I2, int I3>
    inline Vec shuffle() const {
        return Vec(_mm256_permute_pd(m, _MM_SHUFFLE(I3, I2, I1, I0)));
    }

    #define MISC_SHUFFLE_AVX_EXPRS                                      \
    {                                                                   \
        __mm256i i = _mm256_setr_epi64x(args...);                       \
        return _mm256_permutevar_pd(m, i);                              \
    }

    YAVL_DEFINE_MISC_FUNCS

    #undef MISC_SHUFFLE_AVX_EXPRS

    // Geo funcs
    #define GEO_DOT_EXPRS                                               \
    {                                                                   \
        return operator *(b).sum();                                     \
    }

    YAVL_DEFINE_GEO_FUNCS

    #undef GEO_DOT_EXPRS

    // Math funcs
    #define MATH_SUM_EXPRS                                              \
    {                                                                   \
        auto t1 = _mm256_hadd_pd(m, m);                                 \
        auto t2 = _mm256_hadd_pd(t1, t1);                               \
        return _mm256_cvtsd_f64(t2);                                    \
    }

    YAVL_DEFINE_MATH_FUNCS(256, pd)

    #undef MATH_SUM_EXPRS
};

template <>
struct alignas(32) Vec<double, 3> {
    YAVL_VEC_ALIAS_VECTORIZED(double, 4, 4)

    union {
        struct {
            Scalar x, y, z;
        };
        struct {
            Scalar r, g, b;
        };

        std::array<Scalar, Size> arr;
        __m256d m;
    };

    // Ctors
    YAVL_VECTORIZED_CTOR(256, pd, __m256d)

    // Operators
    YAVL_DEFINE_BASIC_OP(256, pd, pd)

    // Misc funcs
    template <int I0, int I1, int I2>
    inline Vec shuffle() const {
        return Vec(_mm256_permute_pd(m, _MM_SHUFFLE(0, I2, I1, I0)));
    }

    #define MISC_SHUFFLE_AVX_EXPRS                                      \
    {                                                                   \
        __mm256i i = _mm256_setr_epi64x(args..., 0);                    \
        return _mm256_permutevar_pd(m, i);                              \
    }

    YAVL_DEFINE_MISC_FUNCS

    #undef MISC_SHUFFLE_AVX_EXPRS

    // Geo funcs
    #define GEO_DOT_EXPRS                                               \
    {                                                                   \
        return operator *(b).sum();                                     \
    }

    YAVL_DEFINE_GEO_FUNCS

    #undef GEO_DOT_EXPRS

    inline auto cross(const Vec& b) const {
        auto t1 = shuffle<1, 2, 0>();
        auto t2 = b.shuffle<2, 0, 1>();
        MULSUB(auto ret, t1.m, t2.m, t3.m);
        return Vec(ret);
    }

    // Math funcs
    #define MATH_SUM_EXPRS                                              \
    {                                                                   \
        return x + y + z;                                               \
    }

    YAVL_DEFINE_MATH_FUNCS(256, pd)

    #undef MATH_SUM_EXPRS
};

#undef MULADD
#undef MULSUB

}
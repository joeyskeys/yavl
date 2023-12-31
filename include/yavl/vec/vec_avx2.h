#pragma once

namespace yavl
{

#define MATH_ABS_EXPRS(VT, BITS, IT1, IT2)                              \
    {                                                                   \
        if constexpr (std::is_unsigned_v<Scalar>)                       \
            return *this;                                               \
        else {                                                          \
            if constexpr (has_avx512vl)                                 \
                return Vec(_mm##BITS##_abs_##IT1(m));                   \
            else                                                        \
                return Vec(_mm##BITS##_andnot_si256(_mm##BITS##_set1_##IT2(0x7FFFFFFFFFFFFFFF), m)); \
        }                                                               \
    }

#define VEC_AVX2_SPECIAL_OP(OP)                                         \
    auto operator OP(const Vec& v) const {                              \
        Vec tmp;                                                        \
        static_for<Size>([&](const auto i) {                            \
            tmp[i] = arr[i] * v[i];                                     \
        });                                                             \
        return tmp;                                                     \
    }                                                                   \
    auto operator OP##=(const Vec& v) {                                 \
        static_for<Size>([&](const auto i) {                            \
            arr[i] OP##= v[i];                                          \
        });                                                             \
        return *this;                                                   \
    }                                                                   \
    auto operator OP(const Scalar v) const {                            \
        Vec tmp;                                                        \
        static_for<Size>([&](const auto i) {                            \
            tmp[i] = arr[i] * v;                                        \
        });                                                             \
        return tmp;                                                     \
    }                                                                   \
    auto operator OP##=(const Scalar v) const {                         \
        static_for<Size>([&](const auto i) {                            \
            arr[i] OP##= v;                                             \
        });                                                             \
        return *this;                                                   \
    }                                                                   \
    friend auto operator OP(const Scalar s, const Vec& v) {             \
        Vec tmp;                                                        \
        static_for<Size>([&](const auto i) {                            \
            tmp[i] = s OP v[i];                                         \
        });                                                             \
        return tmp;                                                     \
    }

template <typename I>
struct alignas(32) Vec<I, 4, enable_if_int64_t<I>> {
    YAVL_VEC_ALIAS_VECTORIZED(I, 4, 4)

    union {
        YAVL_VEC4_MEMBERS
        __m256i m;
    };

    // Ctors
    YAVL_VECTORIZED_CTOR(256, epi64x, __m256i)

    // Operators
    YAVL_DEFINE_OP(256, +, Vec, add, epi64)
    YAVL_DEFINE_OP(256, -, Vec, sub, epi64)
    VEC_AVX2_SPECIAL_OP(*)
    VEC_AVX2_SPECIAL_OP(/)

    // Misc funcs
    template <int I0, int I1, int I2, int I3>
    inline Vec shuffle() const {
        return Vec(_mm256_permute4x64_epi64(m, _MM_SHUFFLE(I3, I2, I1, I0)));
    }

    YAVL_DEFINE_MISC_FUNCS(Vec)

    // Math funcs
    #define MATH_SUM_EXPRS                                              \
    {                                                                   \
        return x + y + z + z;                                           \
    }

    YAVL_DEFINE_MATH_COMMON_FUNCS(Vec, 256, epi64, epi64x)

    #undef MATH_SUM_EXPRS
};

template <typename I>
struct alignas(32) Vec<I, 3, enable_if_int64_t<I>> {
    YAVL_VEC_ALIAS_VECTORIZED(I, 3, 4)

    union {
        YAVL_VEC3_MEMBERS
        __m256i m;
    };

    // Ctors
    YAVL_VECTORIZED_CTOR(256, epi64x, __m256i)

    // Operators
    YAVL_DEFINE_OP(256, +, Vec, add, epi64)
    YAVL_DEFINE_OP(256, -, Vec, sub, epi64)
    VEC_AVX2_SPECIAL_OP(*)
    VEC_AVX2_SPECIAL_OP(/)

    // Misc funcs
    template <int I0, int I1, int I2>
    inline Vec shuffle() const {
        return Vec(_mm256_permute4x64_epi64(m, _MM_SHUFFLE(0, I2, I1, I0)));
    }

    YAVL_DEFINE_MISC_FUNCS(Vec)

    // Math funcs
    #define MATH_SUM_EXPRS                                              \
    {                                                                   \
        return x + y + z;                                               \
    }

    YAVL_DEFINE_MATH_COMMON_FUNCS(Vec, 256, epi64, epi64x)

    #undef MATH_SUM_EXPRS
};

#undef MATH_ABS_EXPRS

}
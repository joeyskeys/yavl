#pragma once

// Common macros
#define YAVL_MAT_ALIAS_VECTORIZED(TYPE, N, INTRIN_N, REGI_N)            \
    YAVL_TYPE_ALIAS(TYPE, N, INTIRN_N)                                  \
    static constexpr uint32_t MSize = REGI_N;                           \
    static constexpr bool vectorized = true;

#define YAVL_MAT_VECTORIZED_CTOR(BITS, IT, REGI_TYPE)                   \
    Mat() {                                                             \
        static_for<MSize>([&](const auto i) {                           \
            m[i] = _mm##BITS##_set1_##IT(static_cast<Scalar>(0));       \
        });                                                             \
    }                                                                   \
    template <typename V>                                               \
        requires std::default_initializable<V> && std::convertible_to<V, Scalar> \
    Mat(V v) {                                                          \
        static_for<MSize>([&](const auto i) {                           \
            m[i] = _mm##BITS##_set1_##IT(static_cast<Scalar>(v));       \
        });                                                             \
    }

#define MAT_MUL_SCALAR_EXPRS                                            \
    {                                                                   \
        Mat tmp;                                                        \
        auto vs = _mm##BITS##_set1_##IT(s);                             \
        static_for<MSize>([&](const auto i) {                           \
            tmp.m[i] = _mm##BITS##_mul_##IT(vs);                        \
        });                                                             \
        return tmp;                                                     \
    }

#define MAT_MUL_ASSIGN_SCALAR_EXPRS                                     \
    {                                                                   \
        auto vs = _mm##BITS##_set1_##IT(s);                             \
        static_for<MSize>([&](const auto i) {                           \
            m[i] = _mm##BITS##_mul_##IT(vs);                            \
        });                                                             \
        return *this;                                                   \
    }

#define MAT_MUL_VEC_EXPRS                                               \
    {                                                                   \
        Vec tmp;                                                        \
    }

// Cascaded including, using max bits intrinsic set available
#if defined(YAVL_X86_AVX512ER)
    #include <yavl/mat/mat_avx512.h>
#elif defined(YAVL_x86_AVX) && defined(YAVL_X86_AVX2)
    #include <yavl/mat/mat_avx.h>
    #include <yavl/mat/mat_avx2.h>
#elif defined(YAVL_X86_SSE42)
    #include <yavl/mat/mat_sse42.h>
#endif

namespace yavl
{

// Mat type aliasing

}
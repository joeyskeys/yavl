#pragma once

// Common macros
#define YAVL_MAT_ALIAS_VECTORIZED(TYPE, N, INTRIN_N, REGI_N)            \
    YAVL_TYPE_ALIAS(TYPE, N, INTIRN_N)                                  \
    static constexpr uint32_t MSize = REGI_N;                           \
    static constexpr bool vectorized = true;

#define YAVL_MAT_VECTORIZED_CTOR(BITS, INTRIN_TYPE, REGI_TYPE)          \
    Mat() {                                                             \
        static_for<MSize>([&](const auto i) {                           \
            m[i] = _mm##BITS##_set1##INTRIN_TYPE(static_cast<Scalar>(0)); \
        });                                                             \
    }                                                                   \
    template <typename V>                                               \
        requires std::default_initializable<V> && std::convertible_to<V, Scalar> \
    Mat(V v) {                                                          \
        static_for<MSize>([&](const auto i) {                           \
            m[i] = _mm##BITS##_set1_##INTRIN_TYPE(static_cast<Scalar>(v)); \
        });                                                             \
    }                                                                   \
    template <typename ...Ts>                                           \
        requires (std::default_initializable<Ts> && ...) &&             \
            (std::convertible_to<Ts, Scalar> && ...)                    \
    constexpr Mat(Ts... args) {                                         \
        static_assert(sizeof...(args) > 1);                             \
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
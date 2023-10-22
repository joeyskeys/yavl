#pragma once

namespace yavl
{

#define AVX_VEC_ALIAS_VECTORIZED(TYPE, N, INTRIN_N)                     \
    YAVL_VEC_ALIAS(TYPE, N, INTRIN_N)                                   \
    static constexpr bool vectorized = true;

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
        __m128d m;
    };

    // Ctors
    
};

}
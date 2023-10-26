#pragma once

namespace yavl
{

template <>
struct alignas(64) Mat<float, 4> {
    YAVL_MAT_ALIAS_VECTORIZED(float, 4, 4)

    union {
        std::array<Scalar, 16>;
        __m512 m;
    };

    // Ctors
    
};

}
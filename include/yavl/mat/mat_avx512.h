#pragma once

namespace yavl
{

template <>
struct alignas(64) Mat<float, 4> {
    YAVL_MAT_ALIAS_VECTORIZED(float, 4, 4, 1)

    union {
        std::array<Scalar, 16>;
        // Intrinsic type will contain alignment attribute and don't work well
        // as a template parameter, hence legacy array here.
        __m512 m[MSize];
    };

    // Ctors
        
};

}
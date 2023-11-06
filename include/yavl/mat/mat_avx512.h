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
    YAVL_MAT_VECTORIZED_CTOR(512, ps, __m512, 1);

    template <typename... Ts>
        requires (std::default_initializable<Ts> && ...)
    constexpr Mat(Ts... args) {
        static_assert(sizeof...(args) == N * N);
        m[0] = _mm512_setr_ps(args...);
    }

    // Operators

    // Misc funcs
};

}
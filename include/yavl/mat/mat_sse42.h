#pragma once

namespace yavl
{

template <>
struct Col<float, 4> {
    YAVL_TYPE_ALIAS(T, N, N)

    Scalar* arr;
    __m128 m;

    Col(Scalar* d)
        : arr(const_cast<Scalar*>(d))
    {
        m = _mm_load_ps(arr);
    }

    // Operators
    YAVL_DEFINE_COL_BASIC_FP_OP(, ps, ps)
};

template <>
struct alignas(16) Mat<float, 4> {
    YAVL_MAT_ALIAS_VECTORIZED(float, 4, 4, 4)

    union {
        std::array<Scalar, 16>;
        __m128 m[MSize];
    };

    // Ctors
    YAVL_MAT_VECTORIZED_CTOR(, ps, __m128)

    template <typename... Ts>
        requires (std::default_initializable<Ts> && ...)
    constexpr Mat(Ts... args) {
        static_assert(sizeof...(args) == Size2);
        auto setf = [&](const uint32_t i, const auto t0, const auto t1,
            const auto t2, const auto t3)
        {
            m[i] = _mm_setr_ps(t0, t1, t2, t3);
        };
        apply_by4(0, setf, args...);
    }

    // Operators
    YAVL_DEFINE_MAT_OP

    // Misc funcs
    YAVL_DEFINE_DATA_METHOD

    // Matrix manipulation methods
    auto transpose() const {
        Mat tmp;
        tmp.m[0] = m[0];
        tmp.m[1] = m[1];
        tmp.m[2] = m[2];
        tmp.m[3] = m[3];
        _MM_TRANSPOSE4_PS(tmp.m[0], tmp.m[1], tmp.m[2], tmp.m[3]);
        return tmp;
    }
};

}
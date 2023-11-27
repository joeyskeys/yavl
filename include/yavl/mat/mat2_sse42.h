#pragma once

namespace yavl
{

// Mat2 implementation for 32 bit types is special since even for avx
// implementation we only need a single __m128/__m128i to hold all
// elements needed for the matrix, so code here will be included for
// different instruction set implementations

#define MAT_MUL_VEC_EXPRS                                               \
{                                                                       \
    if constexpr (std::is_floating_point_v<Scalar>)                     \
        return detail::mat2_mul_vec_f32(*this, v);                      \
    else                                                                \
        return detail::mat2_mul_vec_i32(*this, v);                      \
}

#define MAT_MUL_COL_EXPRS                                               \
{                                                                       \
    if constexpr (std::is_floating_point_v<Scalar>)                     \
        return detail::mat2_mul_vec_f32(*this, Vec<Scalar, Size>{v[0], v[1]}); \
    else                                                                \
        return detail::mat2_mul_vec_i32(*this, Vec<Scalar, Size>{v[0], v[1]}); \
}

#define MAT_MUL_MAT_EXPRS                                               \
{                                                                       \
    Mat tmp;                                                            \
    auto tmpv4a = Vec<Scalar, 4>(m[0]);                                 \
    auto tmpv4b = Vec<Scalar, 4>(mat.m[0]);                             \
    auto v1 = tmpv4a * tmpv4b.shuffle<0, 0, 1, 1>();                    \
    auto v2 = tmpv4a * tmpv4b.shuffle<2, 2, 3, 3>();                    \
    tmp.m[0] = _mm_hadd_ps(v1.shuffle<0, 2, 1, 3>(), v2.shuffle<0, 2, 1, 3>()); \
    return tmp;
}

template <>
struct alignas(16) Mat<float, 2> {
    YAVL_MAT_ALIAS_VECTORIZED(float, 2, 4, 1)

    YAVL_DEFINE_MAT_UNION(__m128)

    // Ctors
    YAVL_MAT_VECTORIZED_CTOR(, ps, __m128)

    template <typename... Ts>
        requires (std::default_initializable<Ts> && ...)
    constexpr Mat(Ts... args) {
        static_assert(sizeof...(args) == Size2);
        m[0] = _mm_setr_ps(args...);
    }

    // Operators
    YAVL_DEFINE_MAT_OP(, ps)

    // Misc funcs
    YAVL_DEFINE_DATA_METHOD

    // Matrix manipulation methods
    auto transpose() const {
        Mat tmp;
        tmp.m[0] = Vec<Scalar, 4>(m[0]).shuffle<0, 2, 1, 3>().m;
        return tmp;
    }
};

#undef MAT_MUL_VEC_EXPRS
#undef MAT_MUL_COL_EXPRS
#undef MAT_MUL_MAT_EXPRS

}
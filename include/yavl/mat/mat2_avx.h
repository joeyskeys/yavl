#pragma once

namespace yavl
{

#define MAT_MUL_VEC_EXPRS                                               \
{                                                                       \
    return detail::mat2_mul_vec_f64(*this, v);                          \
}

#define MAT_MUL_COL_EXPRS                                               \
{                                                                       \
    return detail::mat2_mul_vec_f64(*this, Vec<Scalar, Size>{v[0], v[1]}); \
}

#define MAT_MUL_MAT_EXPRS                                               \
{                                                                       \
    Mat tmp;                                                            \
    auto tmpv4a = Vec<Scalar, 4>(m[0]);                                 \
    auto tmpv4b = Vec<Scalar, 4>(mat.m[0]);                             \
    auto v1 = tmpv4a * tmpv4b.shuffle<0, 0, 1, 1>();                    \
    auto v2 = tmpv4a * tmpv4b.shuffle<2, 2, 3, 3>();                    \
    tmp.m[0] = _mm256_hadd_pd(v1.shuffle<0, 2, 1, 3>(), v2.shuffle<0, 2, 1, 3>()); \
    tmp.m[0] = _mm256_permute4x64_pd(tmp.m[0], 0b11011000);             \
    return tmp;                                                         \
}

template <>
struct alignas(32) Mat<double, 2> {
    YAVL_MAT_ALIAS_VECTORIZED(double, 2, 4, 1)

    YAVL_DEFINE_MAT_UNION(__m256d)

    // Ctors
    YAVL_MAT_VECTORIZED_CTOR(256, pd, __m256d)

    template <typename... Ts>
        requires(std::default_initializable<Ts> && ...)
    constexpr Mat(Ts... args) {
        static_assert(sizeof...(args) == Size2);
        m[0] = _mm256_setr_pd(args...);
    }

    // Operators
    YAVL_DEFINE_MAT_OP(256, pd, mul)

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
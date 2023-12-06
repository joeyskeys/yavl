#pragma once

namespace yavl
{

#define _MM_TRANSPOSE4_PD(col0, col1, col2, col3)                       \
do {                                                                    \
    __m256d tmp3, tmp2, tmp1, tmp0;                                     \
    tmp0 = _mm256_unpacklo_pd((col0), (col1));                          \
    tmp2 = _mm256_unpacklo_pd((col2), (col3));                          \
    tmp1 = _mm256_unpackhi_pd((col0), (col1));                          \
    tmp3 = _mm256_unpackhi_pd((col2), (col3));                          \
    (col0) = _mm256_permute2f128_pd(tmp0, tmp2, 0x20);                  \
    (col1) = _mm256_permute2f128_pd(tmp1, tmp3, 0x20);                  \
    (col2) = _mm256_permute2f128_pd(tmp0, tmp2, 0x31);                  \
    (col3) = _mm256_permute2f128_pd(tmp1, tmp3, 0x31);                  \
} while (0)

#define MAT_MUL_VEC_EXPRS                                               \
{                                                                       \
    return detail::avx_mat_mul_vec_impl(*this, v);                      \
}

#define MAT_MUL_COL_EXPRS                                               \
{                                                                       \
    if constexpr (Size == 2)                                            \
        return detail::avx_mat_mul_vec_impl(*this, Vec<Scalar, Size>{v[0], v[1]}); \
    else                                                                \
        return detail::avx_mat_mul_vec_impl(*this, Vec<Scalar, Size>(v.m)); \
}

#define MAT_MUL_MAT_EXPRS                                               \
{                                                                       \
    Mat tmp;                                                            \
    __m256 m1 = _mm256_set1_ps(0);                                      \
    __m256 m2 = _mm256_set1_ps(0);                                      \
    __m128 col[4];                                                      \
    col[0] = _mm256_extractf128_ps(m[0], 0);                            \
    col[1] = _mm256_extractf128_ps(m[0], 1);                            \
    col[2] = _mm256_extractf128_ps(m[1], 0);                            \
    col[3] = _mm256_extractf128_ps(m[1], 1);                            \
    static_for<Size>([&](const auto i) {                                \
        __m128 cola = col[i];                                           \
        __m256 va = _mm256_broadcast_ps(&cola);                         \
        static_for<MSize>([&](const auto j) {                           \
            auto v1 = mat.arr[(j << 1) * Size + i];                     \
            auto v2 = mat.arr[((j << 1) + 1) * Size + i];               \
            auto vb = _mm256_setr_ps(v1, v1, v1, v1, v2, v2, v2, v2);   \
            tmp.m[j] = _mm256_fmadd_ps(va, vb, tmp.m[j]);               \
        });                                                             \
    });                                                                 \
    return tmp;                                                         \
}

template <>
struct alignas(32) Mat<float, 4> {
    YAVL_MAT_ALIAS_VECTORIZED(float, 4, 8, 2)

    YAVL_DEFINE_MAT_UNION(__m256)

    // Ctors
    YAVL_MAT_VECTORIZED_CTOR(256, ps, __m256)

    template <typename... Ts>
        requires (std::default_initializable<Ts> && ...)
    constexpr Mat(Ts... args) {
        static_assert(sizeof...(args) == Size2);
        auto setf = [&](const uint32_t i, const auto t0, const auto t1,
            const auto t2, const auto t3, const auto t4, const auto t5,
            const auto t6, const auto t7)
        {
            m[i] = _mm256_setr_ps(t0, t1, t2, t3, t4, t5, t6, t7);
        };
        apply_by8(0, setf, args...);
    }

    // Operators
    YAVL_DEFINE_MAT_OP(256, ps, mul)

    // Misc funcs
    YAVL_DEFINE_DATA_METHOD

    // Matrix manipulation methods
    auto transpose() const {
        Mat tmp;
        auto m02 = _mm256_permute2f128_ps(m[0], m[1], 0b00100000);
        auto m13 = _mm256_permute2f128_ps(m[0], m[1], 0b00110001);
        auto tmp0 = _mm256_unpacklo_ps(m02, m13);
        auto tmp1 = _mm256_unpackhi_ps(m02, m13);
        auto mask = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
        tmp.m[0] = _mm256_permutevar8x32_ps(tmp0, mask);
        tmp.m[1] = _mm256_permutevar8x32_ps(tmp1, mask);
        return tmp;
    }
};

#undef MAT_MUL_MAT_EXPRS

// Code here is just same as code for Mat<float, 4> sse impl
#define MAT_MUL_MAT_EXPRS                                               \
{                                                                       \
    Mat tmp;                                                            \
    static_for<Size>([&](const auto i) {                                \
        static_for<Size>([&](const auto j) {                            \
            auto bij = _mm256_set1_pd(mat[i][j]);                       \
            tmp.m[i] = _mm256_fmadd_pd(m[j], bij, tmp.m[i]);            \
        });                                                             \
    });                                                                 \
    return tmp;                                                         \
}

template <>
struct alignas(32) Mat<double, 4> {
    YAVL_MAT_ALIAS_VECTORIZED(double, 4, 8, 4)

    YAVL_DEFINE_MAT_UNION(__m256d)

    // Ctors
    YAVL_MAT_VECTORIZED_CTOR(256, pd, __m256d)

    template <typename... Ts>
        requires (std::default_initializable<Ts> && ...)
    constexpr Mat(Ts... args) {
        static_assert(sizeof...(args) == Size2);
        auto setf = [&](const uint32_t i, const auto t0, const auto t1,
            const auto t2, const auto t3)
        {
            m[i] = _mm256_setr_pd(t0, t1, t2, t3);
        };
        apply_by4(0, setf, args...);
    }

    // Operators
    YAVL_DEFINE_MAT_OP(256, pd, mul);

    // Misc funcs
    YAVL_DEFINE_DATA_METHOD

    // Matrix manipulation methods
    auto transpose() const {
        Mat tmp;
        tmp.m[0] = m[0];
        tmp.m[1] = m[1];
        tmp.m[2] = m[2];
        tmp.m[3] = m[3];
        _MM_TRANSPOSE4_PD(tmp.m[0], tmp.m[1], tmp.m[2], tmp.m[3]);
        return tmp;
    }
};

template <>
struct alignas(32) Mat<double, 3> {
    YAVL_MAT_ALIAS_VECTORIZED(double, 3, 4, 3);

    YAVL_DEFINE_MAT_UNION(__m256d)

    // Ctors
    YAVL_MAT_VECTORIZED_CTOR(256, pd, __m256d)

    template <typename... Ts>
        requires (std::default_initializable<Ts> && ...)
    constexpr Mat(Ts... args) {
        static_assert(sizeof...(args) == Size2);
        auto setf = [&](const uint32_t i, const auto t0, const auto t1,
            const auto t2)
        {
            m[i] = _mm256_setr_pd(t0, t1, t2, 0);
        };
        apply_by3(0, setf, args...);
    }

    // Operators
    YAVL_DEFINE_MAT3_INDEX_OP
    YAVL_DEFINE_MAT_MUL_OP(256, pd, mul)

    // Misc funcs
    YAVL_DEFINE_DATA_METHOD

    // Matrix manipulation methods
    auto transpose() const {
        Mat<Scalar, 4> tmp4;
        tmp4.m[0] = m[0];
        tmp4.m[1] = m[1];
        tmp4.m[2] = m[2];
        tmp4 = tmp4.transpose();
        Mat tmp;
        tmp.m[0] = tmp4.m[0];
        tmp.m[1] = tmp4.m[1];
        tmp.m[2] = tmp4.m[2];
        return tmp;
    }
};

#undef MAT_MUL_VEC_EXPRS
#undef MAT_MUL_COL_EXPRS
#undef MAT_MUL_MAT_EXPRS
#undef _MM_TRANSPOSE4_PD

} // namespace yavl
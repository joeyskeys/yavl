#pragma once

namespace yavl
{

#define MAT_MUL_VEC_EXPRS                                               \
{                                                                       \
    return sse42_mat_mul_vec_impl(*this, v);                            \
}

#define MAT_MUL_COL_EXPRS                                               \
{                                                                       \
    if constexpr (Size == 2)                                            \
        return sse42_mat_mul_vec_impl(*this, Vec<Scalar, Size>{v[0], v[1]}); \
    else                                                                \
        return sse42_mat_mul_vec_impl(*this, Vec<Scalar, Size>(v.m));   \
}

#define MAT_MUL_MAT_EXPRS                                               \
{                                                                       \
    Mat tmp;                                                            \
    static_for<Size>([&](const auto i) {                                \
        static_for<Size>([&](const auto j) {                            \
            auto bij = _mm_set1_ps(mat[i][j]);                          \
            tmp.m[i] = _mm_fmadd_ps(m[j], bij, tmp.m[i]);               \
        });                                                             \
    });                                                                 \
    return tmp;                                                         \
}

template <>
struct alignas(16) Mat<float, 4> {
    YAVL_MAT_ALIAS_VECTORIZED(float, 4, 4, 4)

    YAVL_DEFINE_MAT_UNION(__m128)

    // Ctors
    YAVL_MAT_VECTORIZED_CTOR(, ps, __m128)
    YAVL_MAT_CTOR_BY4(, ps)

    // Operators
    YAVL_DEFINE_MAT_OP(, ps, mul)

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

template <>
struct alignas(16) Mat<float, 3> {
    YAVL_MAT_ALIAS_VECTORIZED(float, 3, 4, 3)

    YAVL_DEFINE_MAT_UNION(__m128)
    
    // Ctors
    YAVL_MAT_VECTORIZED_CTOR(, ps, __m128)
    YAVL_MAT_CTOR_BY3(, ps)

    // Operators
    YAVL_DEFINE_MAT3_INDEX_OP
    YAVL_DEFINE_MAT_MUL_OP(, ps, mul)

    // Misc funcs
    YAVL_DEFINE_DATA_METHOD

    // Matrix manipulation methods
    auto transpose() const {
        Mat<Scalar, 4> tmp4;
        tmp4.m[0] = m[0];
        tmp4.m[1] = m[1];
        tmp4.m[2] = m[2];
        _MM_TRANSPOSE4_PS(tmp4.m[0], tmp4.m[1], tmp4.m[2], tmp4.m[3]);
        Mat tmp;
        tmp.m[0] = tmp4.m[0];
        tmp.m[1] = tmp4.m[1];
        tmp.m[2] = tmp4.m[2];
        return tmp;
    }
};

#undef MAT_MUL_MAT_EXPRS

#define MAT_MUL_MAT_EXPRS                                               \
{                                                                       \
    Mat tmp;                                                            \
    static_for<Size>([&](const auto i) {                                \
        static_for<Size>([&](const auto j) {                            \
            auto bij = _mm_set1_epi32(mat[i][j]);                       \
            tmp.m[i] = _mm_add_epi32(_mm_mullo_epi32(m[j], bij), tmp.m[i]); \
        });                                                             \
    });                                                                 \
    return tmp;                                                         \
}

template <typename I>
struct alignas(16) Mat<I, 4, true, enable_if_int32_t<I>> {
    YAVL_MAT_ALIAS_VECTORIZED(I, 4, 4)

    YAVL_DEFINE_MAT_UNION(__m128i)

    // Ctors
    YAVL_MAT_VECTORIZED_CTOR(, epi32, __m128i)
    YAVL_MAT_CTOR_BY4(, epi32)

    // Operators
    YAVL_DEFINE_MAT_OP(, epi32, mullo)

    // Misc funcs
    YAVL_DEFINE_DATA_METHOD

    // Matrix manipulation methods
    auto transpose() const {
        Mat tmp;
        __m128* mf[4];
        mf[0] = &tmp.m[0];
        mf[1] = &tmp.m[1];
        mf[2] = &tmp.m[2];
        mf[3] = &tmp.m[3];
        _MM_TRANSPOSE4_PS(*mf[0], *mf[1], *mf[2], *mf[3]);
        return tmp;
    }
};

template <typename I>
struct alignas(16) Mat<I, 3, true, enable_if_int32_t<I>> {
    YAVL_MAT_ALIAS_VECTORIZED(I, 4, 3)

    YAVL_DEFINE_MAT_UNION(__m128i)

    // Ctors
    YAVL_MAT_VECTORIZED_CTOR(, epi32, __m128i)
    YAVL_MAT_CTOR_BY3(, epi32)

    // Operators
    YAVL_DEFINE_MAT_OP(, epi32, mullo)

    // Misc funcs
    YAVL_DEFINE_DATA_METHOD

    // Matrix manipulation methods
    auto transpose() const {
        Mat tmp;
        __m128* mf[4];
        __m128 extra_m;
        mf[0] = &tmp.m[0];
        mf[1] = &tmp.m[1];
        mf[2] = &tmp.m[2];
        mf[3] = &extra_m;
        _MM_TRANSPOSE4_PS(*mf[0], *mf[1], *mf[2], *mf[3]);
        return tmp;
    }
};

#undef MAT_MUL_VEC_EXPRS
#undef MAT_MUL_COL_EXPRS
#undef MAT_MUL_MAT_EXPRS

} // namespace yavl
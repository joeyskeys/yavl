#pragma once

namespace yavl
{

template <typename T, uint32_t N>
static inline Vec<T, N> sse42_mat_mul_vec_32_impl(const Mat<T, N>& mat, const Vec<T, N>& vec) {
    // Check comments in avx512 impl
    Vec<T, N> tmp;
    if constexpr (N == 2) {
        // TODO : integeters
        tmp = mat2_mul_vec_f32(mat, vec);
    }
    else {
        static_for<Mat<T, N>::MSize>([&](const auto i) {
            if constexpr (std::is_floating_point_v<T>) {
                auto vm = _mm_set1_ps(vec[i]);
                tmp.m = _mm_fmadd_ps(mat.m[i], vm, tmp.m);
            }
            else {
                auto vm = _mm_set1_ps(vec[i]);
                tmp.m = _mm_add_epi32(tmp.m, _mm_mul_epi32(mat.m[i], vm));
            }
        });
    }
    return tmp;
}

#define MAT_MUL_VEC_EXPRS                                               \
{                                                                       \
    return sse42_mat_mul_vec_32_impl(*this, v);                         \
}

#define MAT_MUL_COL_EXPRS                                               \
{                                                                       \
    if constexpr (Size == 2)                                            \
        return sse42_mat_mul_vec_32_impl(*this, Vec<Scalar, Size>{v[0], v[1]}); \
    else                                                                \
        return sse42_mat_mul_vec_32_impl(*this, Vec<Scalar, Size>(v.m)); \
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
    YAVL_DEFINE_MAT_OP(, ps)

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

    template <typename... Ts>
        requires (std::default_initializable<Ts> && ...)
    constexpr Mat(Ts... args) {
        static_assert(sizeof...(args) == Size2);
        auto setf = [&](const uint32_t i, const auto t0, const auto t1,
            const auto t2)
        {
            m[i] = _mm_setr_ps(t0, t1, t2, 0);
        };
        apply_by3(0, setf, args...);
    }

    // Operators
    YAVL_DEFINE_MAT_OP(, ps)

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

#undef MAT_MUL_VEC_EXPRS
#undef MAT_MUL_COL_EXPRS
#undef MAT_MUL_MAT_EXPRS

} // namespace yavl
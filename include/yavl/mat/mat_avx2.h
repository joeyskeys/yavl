#pragma once

namespace yavl
{

#define MAT_MUL_VEC_EXPRS                                               \
{                                                                       \
    return avx_mat_mul_vec_impl(*this, v);                              \
}

#define MAT_MUL_COL_EXPRS                                               \
{                                                                       \
    if constexpr (Size == 2)                                            \
        return avx_mat_mul_vec_impl(*this, Vec<Scalar, Size>{v[0], v[1]}); \
    else                                                                \
        return avx_mat_mul_vec_impl(*this, Vec<Scalar, Size>(v.m));     \
}

#define MAT_MUL_MAT_EXPRS                                               \
{                                                                       \
    Mat tmp;                                                            \
    static_for<MSize>([&](const auto i) {                               \
        __m256i m1 = _mm256_set1_epi32(0);                              \
        static_for<MSize>([&](const auto j) {                           \
            auto v1 = mat.arr[(i << 1) * Size + (j << 1)];              \
            auto v2 = mat.arr[(i << 1) * Size + (j << 1 + 1)];          \
            auto bij = _mm256_setr_epi32(v1, v1, v1, v1, v2, v2, v2, v2); \
            m1 = _mm256_add_epi32(_mm256_mullo_epi32(m[j], bij), m1);   \
        });                                                             \
        auto m1_flip = _mm256_permute2x128_si256(m1, m1, 1);            \
        m1 = _mm256_add_epi32(m1, m1_flip);                             \
        __m256i m2 = _mm256_set1_ps(0);                                 \
        static_for<MSize>([&](const auto j) {                           \
            auto v1 = mat.arr[((i << 1) + 1) * Size + (j << 1)];        \
            auto v2 = mat.arr[((i << 1) + 1) * Size + (j << 1 + 1)];    \
            auto bij = _mm256_setr_epi32(v1, v1, v1, v1, v2, v2, v2, v2); \
            m2 = _mm256_add_epi32(_mm256_mullo_epi32(m[j], bij), m2);   \
        });                                                             \
        auto m2_flip = _mm256_permute2x128_si256(m2, m2, 1);            \
        m2 = _mm256_add_epi32(m2, m2_flip);                             \
        tmp.m[i] = _mm256_permute2x128_si256(m1, m2, 0b00100000);       \
    });                                                                 \
    return tmp;                                                         \
}

template <typename I>
struct alignas(32) Mat<I, 4, true, enable_if_int32_t<I>> {
    YAVL_MAT_ALIAS_VECTORIZED(I, 4, 8, 2)

    YAVL_DEFINE_MAT_UNION(__m256i)

    // Ctors
    YAVL_MAT_VECTORIZED_CTOR(256, epi32, __m256)

    template <typename... Ts>
        requires (std::default_initializable<Ts> && ...)
    constexpr Mat(Ts... args) {
        static_assert(sizeof...(args) == Size2);
        auto seti = [&](const uint32_t i, const auto t0, const auto t1,
            const auto t2, const auto t3, const auto t4, const auto t5,
            const auto t6, const auto t7)
        {
            m[i] = _mm256_setr_epi32(t0, t1, t2, t3, t4, t5, t6, t7);
        };
        apply_by8(0, seti, args...);
    }

    // Operators
    YAVL_DEFINE_MAT_OP(256, epi32, mullo)

    // Misc funcs
    YAVL_DEFINE_DATA_METHOD

    // Matrix manipulation methods
    auto transpose() const {
        Mat tmp;
        auto m02 = _mm256_permute2f128_si256(m[0], m[1], 0b00100000);
        auto m13 = _mm256_permute2f128_si256(m[0], m[1], 0b00110001);
        auto tmp0 = _mm256_unpacklo_epi32(m02, m13);
        auto tmp1 = _mm256_unpackhi_epi32(m02, m13);
        constexpr auto mask = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
        tmp.m[0] = _mm256_permute8x32_epi32(tmp0, mask);
        tmp.m[1] = _mm256_permute8x32_epi32(tmp1, mask);
    }
};

// Leave as a placeholder for now, use case is scarce
/*
template <typename I>
struct alignas(32) Mat<I, 4, true, enable_if_int64_v<T>> {
    YAVL_MAT_ALIAS_VECTORIZED(I, 4, 8, 4)

    YAVL_DEFINE_MAT_UNION(__m256i)

    // Ctors
    //YAVL_MAT_VECTORIZED_CTOR(256, epi64, __m256i)
};

template <typename I>
struct alignas(32) Mat<I, 3, true, enable_if_int64_v<T>> {

};
*/

#undef MAT_MUL_VEC_EXPRS
#undef MAT_MUL_COL_EXPRS
#undef MAT_MUL_MAT_EXPRS

} // namespace yavl
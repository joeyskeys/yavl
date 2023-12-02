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
}

template <typename I>
struct alignas(32) Mat<I, 4, true, enable_if_int32_t<I>> {
    YAVL_MAT_ALIAS_VECTORIZED(I, 4, 8, 2)

    YAVL_DEFINE_MAT_UNION(__m256i)

    // Ctors
    YAVL_MAT_VECTORIZED_CTOR(256, epi32, __m256)

    template <typename... Ts>
        requires(std::default_initializable<Ts> && ...)
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
    YAVL_DEFINE_MAT_OP(256, epi32)

    // Misc funcs
    YAVL_DEFINE_DATA_METHOD

    // Matrix manipulation methods
    auto transpose() const {

    }
};

template <typename I>
struct alignas(32) Mat<I, 3, true, enable_if_int32_t<T>> {
    // Same as impl in avx, write everything manually
    using Scalar = T;
    static constexpr uint32_t Size = 3;
    static constexpr uint32_t IntrinSize = 4; // Problematic
    static constexpr uint32_t Size2 = Size * Size;
    static constexpr uint32_t MSize = 2; // Problematic
    static constexpr bool vectorized = true;

    union {
        std::array<Scalar, 12> arr;
        struct {
            __m256i m1;
            __m128i m2;
        };
    };
};

template <typename I>
struct alignas(32) Mat<I, 4, true, enable_if_int64_v<T>> {

};

template <typename I>
struct alignas(32) Mat<I, 3, true, enable_if_int64_v<T>> {

};

} // namespace yavl
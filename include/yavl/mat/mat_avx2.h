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
        __m256 m1 = _mm256_set1_epi32(0);                               \
        static_for<MSize>([&](const auto j) {                           \
            auto v1 = mat.arr[(i << 1) * Size + (j << 1)];              \
            auto v2 = mat.arr[(i << 1) * Size + (j << 1 + 1)];          \
            auto bij = _mm256_setr_ps(v1, v1, v1, v1, v2, v2, v2, v2);  \
            m1 = _mm256_add_epi32(_mm256_mullo_epi32(m[j], bij), m1);   \
        });                                                             \
        auto m1_flip = _mm256_permute2x128_si256(m1, m1, 1);            \
        m1 = _mm256_add_ps(m1, m1_flip);                                \
        __m256 m2 = _mm256_set1_ps(0);                                  \
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

    // Ctors
    Mat() {
        m1 = _mm256_set1_epi32(static_cast<Scalar>(0));
        m2 = _mm_set1_epi32(static_cast<Scalar>(0));
    }

    template <typename V>
        requires std::default_initializable<V> && std::convertible_to<V, Scalar>
    Mat(V v) {
        m1 = _mm256_set1_epi32(static_cast<Scalar>(v));
        m2 = _mm_set1_epi32(static_cast<Scalar>(v));
    }

    constexpr Mat(const Scalar t0, const Scalar t1, const Scalar t2,
        const Scalar t3, const Scalar t4, const Scalar t5, const Scalar t6,
        const Scalar t7, const Scalar t8)
    {
        m1 = _mm256_setr_epi32(t0, t1, t2, 0, t3, t4, t5, 0);
        m2 = _mm_setr_epi32(t6, t7, t8, 0);
    }

    // Operators
    YAVL_DEFINE_MAT_INDEX_OP

    auto operator *(const Scalar s) const {
        Mat tmp;
        auto vm1 = _mm256_set1_epi32(s);
        auto vm2 = _mm_set1_epi32(s);
        tmp.m1 = _mm256_mullo_epi32(m1, vm1);
        tmp.m2 = _mm_mullo_epi32(m2, vm2);
        return tmp;
    }

    auto operator *=(const Scalar s) const {
        auto vm1 = _mm256_set1_epi32(s);
        auto vm2 = _mm_set1_epi32(s);
        m1 = _mm256_mullo_epi32(m1, vm1);
        m2 = _mm_mullo_epi32(m2, vm2);
        return *this;
    }

    auto operator *(const Vec<Scalar, Size>& v) const {
        constexpr auto vm1 = _mm256_setr_epi32(v[0], v[0], v[0], v[0], v[1],
            v[1], v[1], v[1]);
        constexpr auto vm2 = _mm256_set1_epi32(v[2]);
        vm1 = _mm256_mullo_epi32(m1, vm1);
        vm2 = _mm_mullo_epi32(m2, vm2);

        auto vm1_flip = _mm256_permute2x128_si256(vm1);
        Vec<Scalar, Size> tmp;
        auto vm1_hsum = _mm256_hadd_epi32(vm1, vm1_flip);
        tmp.m = _mm_add_epi32(vm2, _mm256_extracti128_si256(vm1_hsum, 0));
        return tmp;
    }

    auto operator *(const Col<Scalar, Size>& v) const {
        auto vec = Vec<Scalar, Size>{v.m};
        return operator *(vec);
    }

    auto operator *(const Mat& mat) const {
        __m128i col[3];
        col[0] = _mm256_extracti128_si256(mat.m1, 0);
        col[1] = _mm256_extracti128_si256(mat.m1, 1);
        col[2] = mat.m2;
        static_for<Size>([&](const auto i) {
            col[i] = operator *(Vec<Scalar, Size>{col[i]});
        });
        Mat tmp;
        tmp.m1 = _mm256_insertf128_epi32(tmp.m1, col[0], 0);
        tmp.m1 = _mm256_insertf128_epi32(tmp.m1, col[1], 1);
        tmp.m2 = col[3];
        return tmp;
    }

    auto operator *=(const Mat& mat) {
        Mat tmp = *this * mat;
        *this = tmp;
        return *this;
    }

    // Misc funcs
    YAVL_DEFINE_DATA_METHOD

    // Matrix manipulation methods
    auto transpose() const {
        Mat<Scalar, 4> mat4;
        memcpy(mat4.data(), data(), 12 * sizeof(Scalar));
        mat4 = mat4.transpose();
        memcpy(data(), mat4.data(), 12 * sizeof(Scalar));
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
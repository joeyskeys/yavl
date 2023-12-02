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
    (col1) = _mm256_permute2f128_pd(tmp0, tmp2, 0x31);                  \
    (col2) = _mm256_permute2f128_pd(tmp1, tmp3, 0x20);                  \
    (col3) = _mm256_permute2f128_pd(tmp1, tmp3, 0x31);                  \
} while (0)

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
        __m256 m1 = _mm256_set1_ps(0);                                  \
        static_for<MSize>([&](const auto j) {                           \
            auto v1 = mat.arr[(i << 1) * Size + (j << 1)];              \
            auto v2 = mat.arr[(i << 1) * Size + (j << 1 + 1)];          \
            auto bij = _mm256_setr_ps(v1, v1, v1, v1, v2, v2, v2, v2);  \
            m1 = _mm256_fmadd_ps(m[j], bij, m1);                        \
        });                                                             \
        auto m1_flip = _mm256_permute2f128_ps(m1, m1, 1);               \
        m1 = _mm256_add_ps(m1, m1_flip);                                \
        __m256 m2 = _mm256_set1_ps(0);                                  \
        static_for<MSize>([&](const auto j) {                           \
            auto v1 = mat.arr[((i << 1) + 1) * Size + (j << 1)];        \
            auto v2 = mat.arr[((i << 1) + 1) * Size + (j << 1 + 1)];    \
            auto bij = _mm256_setr_ps(v1, v1, v1, v1, v2, v2, v2, v2);  \
            m2 = _mm256_fmadd_ps(m[j], bij, m2);                        \
        });                                                             \
        auto m2_flip = _mm256_permute2f128_ps(m2, m2, 1);               \
        m2 = _mm256_add_ps(m2, m2_flip);                                \
        tmp.m[i] = _mm256_permute2f128_ps(m1, m2, 0b00100000);          \
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
    YAVL_DEFINE_MAT_OP(256, ps)

    // Misc funcs
    YAVL_DEFINE_DATA_METHOD

    // Matrix manipulation methods
    auto transpose() const {
        Mat tmp;
        auto m02 = _mm256_permute2f128_ps(m[0], m[1], 0b00100000);
        auto m13 = _mm256_permute2f128_ps(m[0], m[1], 0b00110001);
        auto tmp0 = _mm256_unpacklo_ps(m02, m13);
        auto tmp1 = _mm256_unpackhi_ps(m02, m13);
        constexpr auto mask = _mm256_setr_epi32(0, 1, 4, 5, 2, 3, 6, 7);
        tmp.m[0] = _mm256_permute8x32_ps(tmp0, mask);
        tmp.m[1] = _mm256_permute8x32_ps(tmp1, mask);
    }
}

template <>
struct alignas(32) Mat<float, 3> {
    // This one is kinda special, write everything without most macros
    using Scalar = float;
    static constexpr uint32_t Size = 3;
    static constexpr uint32_t IntrinSize = 4; // Problematic
    static constexpr uint32_t Size2 = Size * Size;
    static constexpr uint32_t MSize = 2; // Problematic
    static constexpr bool vectorized = true;

    union {
        std::array<Scalar, 12> arr;
        struct {
            __m256 m1;
            __m128 m2;
        };
    };

    // Ctors
    Mat() {
        m1 = _mm256_set1_ps(static_cast<Scalar>(0));
        m2 = _mm_set1_ps(static_cast<Scalar>(0));
    }

    template <typename V>
        requires std::default_initializable<V> && std::convertible_to<V, Scalar>
    Mat(V v) {
        m1 = _mm256_set1_ps(static_cast<Scalar>(v));
        m2 = _mm_set1_ps(static_cast<Scalar>(v));
    }

    constexpr Mat(const Scalar t0, const Scalar t1, const Scalar t2,
        const Scalar t3, const Scalar t4, const Scalar t5, const Scalar t6,
        const Scalar t7, const Scalar t8)
    {
        m1 = _mm256_setr_ps(t0, t1, t2, 0, t3, t4, t5, 0);
        m2 = _mm_setr_ps(t6, t7, t8, 0);
    }

    // Operators
    YAVL_DEFINE_MAT_INDEX_OP

    auto operator *(const Scalar s) const {
        Mat tmp;
        auto vm1 = _mm256_set1_ps(s);
        auto vm2 = _mm_set1_ps(s);
        tmp.m1 = _mm256_mul_ps(m1, vm1);
        tmp.m2 = _mm_mul_ps(m2, vm2);
        return tmp;
    }

    auto operator *=(const Scalar s) {
        auto vm1 = _mm256_set1_ps(s);
        auto vm2 = _mm_set1_ps(s);
        m1 = _mm256_mul_ps(m1, vm1);
        m2 = _mm_mul_ps(m2, vm2);
        return *this;
    }

    auto operator *(const Vec<Scalar, Size>& v) const {
        constexpr auto vm1 = _mm256_setr_ps(v[0], v[0], v[0], v[0], v[1], v[1],
            v[1], v[1]);
        constexpr auto vm2 = _mm256_set1_ps(v[2]);
        vm1 = _mm256_mul_ps(m1, vm1);
        vm2 = _mm_mul_ps(m2, vm2);

        auto vm1_flip = _mm256_permute2f128_ps(vm1);
        Vec<Scalar, Size> tmp;
        auto vm1_hsum = _mm256_hadd_ps(vm1, vm1_flip);
        tmp.m = _mm_add_ps(vm2, _mm256_extractf128_ps(vm1_hsum, 0));
        return tmp;
    }

    auto operator *(const Col<Scalar, Size>& v) const {
        auto vec = Vec<Scalar, Size>{v.m};
        return operator *(vec);
    }

    auto operator *(const Mat& mat) const {
        __m128 col[3];
        col[0] = _mm256_extractf128_ps(mat.m1, 0);
        col[1] = _mm256_extractf128_ps(mat.m1, 1);
        col[2] = mat.m2;
        static_for<Size>([&](const auto i) {
            col[i] = operator *(Vec<Scalar, Size>{col[i]});
        });
        Mat tmp;
        tmp.m1 = _mm256_insertf128_ps(tmp.m1, col[0], 0);
        tmp.m1 = _mm256_insertf128_ps(tmp.m1, col[1], 1);
        tmp.m2 = col[3];
        return tmp;

        /*
        // Or this impl, need to do some benchmarking
        Mat<Scalar, 4> m1, m2;
        memcpy(m1.data(), data(), 12 * sizeof(float));
        memcpy(m2.data(), mat.data(), 12 * sizeof(float));
        m1 *= m2;
        Mat tmp;
        memcpy(tmp.data(), m1.data(), 12 * sizeof(float));
        return tmp;
        */
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
    YAVL_DEFINE_MAT_OP(256, pd);

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
    YAVL_MAT_ALIAS_VECTORIZED(double, 3, 8, 3);

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
    YAVL_DEFINE_MAT_OP(256, pd)

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

} // namespace yavl
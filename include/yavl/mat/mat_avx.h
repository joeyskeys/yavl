#pragma once

#include <yavl/mat/mat2_sse42.h>

namespace yavl
{

template <typename T, uint32_t N>
static inline Vec<T, N> avx_mat_mul_vec_32_impl(const Mat<T, N>& mat, const Vec<T, N>& vec) {
    // Check comments in avx512 impl
    Vec<T, N> tmp;
    if constexpr (N == 2) {
        tmp = mat2_mul_vec_f32(mat, vec);
    }
    else {
        static_for<Mat<T, N>::MSize>([&](const auto i) {
            auto v1 = vec[i << 1];
            auto v2 = vec[i << 1 + 1];
            if constexpr (std::is_floating_point_v<T>) {
                auto vm = _mm256_set1_ps(0);
                auto b = _mm256_setr_ps(v1, v1, v1, v1, v2, v2, v2, v2);
                vm = _mm256_fmadd_ps(mat.m[i], b, vm);
                auto vm_flip = _mm256_permute2f128_ps(vm, vm, 1);
                tmp.m = _mm256_extractf128_ps(_mm256_add_ps(vm, vm_flip), 0);
            }
            /*
            else {
                auto vm256 = _mm256_set1_epi32(0);
                auto vm = _mm256_setr_epi32(v1, v1, v1, v1, v2, v2, v2, v2);
                vm256 = _mm256_add_epi32(vm256, _mm256_mullo_epi32(mat.m[i], vm));
                vm256 = _mm256_permutevar_epi32(vm256, idx256);
                tmp.m = _mm256_extract
            }
            */
        });
    }
    return tmp;
}

#define MAT_MUL_VEC_EXPRS                                               \
{
    return avx_mat_mul_vec_32_impl(*this, v);
}

#define MAT_MUL_COL_EXPRS                                               \
{                                                                       \
    if constexpr (Size == 2)                                            \
        return avx_mat_mul_vec_32_impl(*this, Vec<Scalar, Size>{v[0], v[1]}); \
    else                                                                \
        return avx_mat_mul_vec_32_impl(*this, Vec<Scalar, Size>(v.m));  \
}

#define MAT_MUL_MAT_EXPRS                                               \
{                                                                       \
    Mat tmp;                                                            \
    static_for<MSize>([&](const auto i) {                               \
        __m256 m1 = _mm256_set1_ps(0);                                  \
        static_for<MSize>([&](const auto j) {                           \
            auto v1 = mat.arr[(i << 1) * Size + (j << 1)];        \
            auto v2 = mat.arr[(i << 1) * Size + (j << 1 + 1)];    \
            auto bij = _mm256_setr_ps(v1, v1, v1, v1, v2, v2, v2, v2);  \
            m1 = _mm256_fmadd_ps(m[j], bij, m1);                        \
        });                                                             \
        auto m1_flip = _mm256_permute2f128_ps(m1, m1, 1);               \
        m1 = _mm256_add_ps(m1, m1_flip);                                \
        __m256 m2 = _mm256_set1_ps(0);                                  \
        static_for<MSize>([&](const auto j) {                           \
            auto v1 = mat.arr[((i << 1) + 1) * Size + (j << 1)];  \
            auto v2 = mat.arr[((i << 1) + 1) * Size + (j << 1 + 1)]; \
            auto bij = _mm256_setr_ps(v1, v1, v1, v1, v2, v2, v2, v2);  \
            m2 = _mm256_fmadd_ps(m[j], bij, m2);                        \
        });                                                             \
        auto m2_flip = _mm256_permute2f128_ps(m2, m2, 1);               \
        m2 = _mm256_add_ps(m2, m2_flip);                                \
        tmp.m[i] = _mm256_permute2f128(m1, m2, 0b00100000);             \
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
        tmp.m[0] = _mm256_permutevar_ps(tmp0, mask);
        tmp.m[1] = _mm256_permutevar_ps(tmp1, mask);
    }
}

#undef MAT_MUL_VEC_EXPRS
#undef MAT_MUL_COL_EXPRS
#undef MAT_MUL_MAT_EXPRS

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
        requires std::default_initializable<V> && std::convertible_to<V, Scalar> \
    Mat(V v) {
        m1 = _mm256_set1_ps(static_cast<Scalar>(v));
        m2 = _mm_set1_ps(static_cast<Scalar>(v));
    }

    constexpr Mat(const Scalar& t0, const Scalar& t1, const Scalar& t2,
        const Scalar& t3, const Scalar& t4, const Scalar& t5, const Scalar& t6,
        const Scalar& t7, const Scalar& t8)
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
        col[3] = mat.m2;
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
        memcpy(mat4.data(), data(), 12 * sizeof(float));
        mat4 = mat4.transpose();
        memcpy(data(), mat4.data(), 12 * sizeof(float));
    }
};

template <>
struct alignas(32) Mat<double, 4> {
    YAVL_MAT_ALIAS_VECTORIZED(float, 4, 8, 4)

    YAVL_DEFINE_MAT_UNION(__m256)

    // Ctors
    YAVL_MAT_VECTORIZED_CTOR(256, pd, __m128)

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
        // TODO
    }
};

template <>
struct alignas(32) Mat<double, 3> {

};

template <>
struct alignas(32) Mat<double, 2> {

};

}
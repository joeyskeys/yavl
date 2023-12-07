#pragma once

namespace yavl
{

template <>
struct Col<float, 4> {
    YAVL_TYPE_ALIAS(T, N, N)

    Scalar* arr;
    __m128 m; // As a cache to save a load for const methods

    Col(Scalar* d)
        : arr(const_cast<Scalar*>(d))
    {
        m = _mm_load_ps(arr);
    }

    // Operators
    YAVL_DEFINE_COL_BASIC_FP_OP(, ps, ps)
};

template <typename T, uint32_t N>
static inline Vec<T, N> avx512_mat_mul_vec_impl(const Mat<T, N>& mat, const Vec<T, N>& vec) {
    // 1. Two possible impl for mat vec/col multiplication
    // Need further benchmark to pick out one
    // (According to Intel doc, the second one maybe slightly better,
    // it's just I currently don't have device that supports avx512...)
    // (imm8 for extractf32x4 maybe illegal since it's a func
    // param rather than a template param. If this's the case, might
    // need to unfold the expression manually)
    // 2. Design problem:
    // put each impl into specific class or use if constexpr to wrap
    // up all the possible situation?
    Vec<T, N> tmp;
    if constexpr (std::is_floating_point_v<T>) {
        if constexpr (N == 2) {
            if constexpr (sizoef(T) == 4)
                tmp = mat2_mul_vec_f32(mat, vec);
            else
                tmp = mat2_mul_vec_f64(mat, vec);
        }
        else {
            if constexpr (sizeof(T) == 4) {
                auto v512 = _mm512_setr_ps(vec[0], vec[0], vec[0], vec[0],
                    vec[1], vec[1], vec[1], vec[1],
                    vec[2], vec[2], vec[2], vec[2],
                    vec[3], vec[3], vec[3], vec[3]);
                v512 = _mm512_mul_ps(mat.m[0], v512);
                __m256 v256[2];
                v256[0] = _mm512_extractf32x8_ps(v512, 0);
                v256[1] = _mm512_extractf32x8_ps(v512, 1);
                auto lo = _mm256_unpacklo_ps(v256[0], v256[1]);
                auto hi = _mm256_unpackhi_ps(v256[0], v256[1]);
                lo = _mm256_hadd_ps(lo, hi);
                lo = _mm256_permute_ps(lo, 0b11011000);
                hi = _mm256_permute2f128_ps(lo, lo, 1);
                tmp.m = _mm256_extractf128_ps(_mm256_hadd_ps(lo, hi), 0);
            }
            else {
                static_for<Mat<T, N>::MSize>([&](const auto i) {
                    auto v1 = vec[i << 1];
                    auto v2 = vec[i << 1 + 1];
                    auto vm = _mm512_set1_pd(0);
                    auto b = _mm512_setr_pd(v1, v1, v1, v1, v2, v2, v2, v2);
                    vm = _mm512_fmadd_pd(mat.m[1], b, vm);
                    auto flip_mask = _mm512_setr_epi64(4, 5, 6, 7, 0, 1, 2, 3);
                    auto vm_flip = _mm512_permutex2var_pd(flip_mask, vm);
                    tmp.m = _mm512_extractf64x4_pd(_mm512_add_pd(vm, vm_flip), 0);
                });
            }
        }
    }
    else {
        if constexpr (N == 2) {
            if constexpr (sizeof(T) == 4)
                tmp = mat2_mul_vec_i32(mat, vec);
            else
                tmp = mat2_mul_vec_i64(mat, vec);
        }
        else {
            if constexpr (sizeof(T) == 4) {
                auto v512 = _mm512_setr_epi32(vec[0], vec[0], vec[0], vec[0],
                    vec[1], vec[1], vec[1], vec[1],
                    vec[2], vec[2], vec[2], vec[2],
                    vec[3], vec[3], vec[3], vec[3]);
                v512 = _mm512_mullo_epi32(mat.m[0], v512);
                __m256i v256[2];
                v256[0] = _mm512_extracti32x4_epi32(v512, 0);
                v256[1] = _mm512_extracti32x4_epi32(v512, 1);
                auto lo = _mm256_unpacklo_epi32(v256[0], v256[1]);
                auto hi = _mm256_unpackhi_epi32(v256[0], v256[1]);
                lo = _mm256_hadd_epi32(lo, hi);
                lo = _mm256_permuatevar_epi32(lo, _mm256_setr_epi32(0, 2, 1, 3,
                    4, 6, 5, 7));
                hi = _mm256_permute2x128_si256(lo, lo, 1);
                tmp.m = _mm256_extracti128_si256(_mm256_hadd_epi32(lo, hi), 0);
            }
            else {
                static_for<Mat<T, N>::MSize>([&](const auto i) {
                    auto v1 = vec[i << 1];
                    auto v2 = vec[i << 1 + 1];
                    auto vm = _mm512_set1_epi64(0);
                    auto b = _mm512_setr_epi64(v1, v1, v1, v1, v2, v2, v2, v2);
                    vm =_mm512_add_epi64(_mm512_mullo_epi64(mat.m[i], b), vm);
                    auto flip_mask = _mm512_setr_epi64(4, 5, 6, 7, 0, 1, 2, 3);
                    auto vm_flip = _mm512_permutexvar_epi64(flip_mask, vm);
                    tmp.m = _mm512_extracti64x4_epi64(_mm512_add_epi64(vm, vm_flip), 0);
                });
            }
        }
    }
    return tmp;
}

#define MAT_MUL_VEC_EXPRS                                           \
{                                                                   \
    return avx512_mat_mul_vec_impl(*this, v);                       \
}

#define MAT_MUL_COL_EXPRS                                           \
{                                                                   \
    if constexpr (Size == 2)                                        \
        return avx512_mat_mul_vec_impl(*this, Vec<Scalar, Size>{v[0], v[1]}); \
    else                                                            \
        return avx512_mat_mul_vec_impl(*this, Vec<Scalar, Size>(v.m)); \
}

#define MAT_MUL_MAT_EXPRS                                           \
{                                                                   \
    Mat tmp;                                                        \
    __m512 vm[4];                                                   \
    static_for<4>([&](const auto i) constexpr {                     \
        vm[i] = _mm512_broadcast_f32x4(_mm512_extractf32x4_ps(m[0], i)); \
        tmp.m[0] = _mm512_fmadd_ps(m[0], vm[i], tmp.m[0]);          \
    });                                                             \
    return tmp;                                                     \
}

template <>
struct alignas(64) Mat<float, 4> {
    YAVL_MAT_ALIAS_VECTORIZED(float, 4, 16, 1)

    // Intrinsic type will contain alignment attribute and don't work well
    // as a template parameter, hence legacy array here.
    YAVL_DEFINE_MAT_UNION(__m512)

    // Ctors
    YAVL_MAT_VECTORIZED_CTOR(512, ps, __m512)

    template <typename... Ts>
        requires (std::default_initializable<Ts> && ...)
    constexpr Mat(Ts... args) {
        static_assert(sizeof...(args) == Size2);
        m[0] = _mm512_setr_ps(args...);
    }

    // Operators
    YAVL_DEFINE_MAT_OP(512, ps, mul)

    // Misc funcs
    YAVL_DEFINE_DATA_METHOD

    // Matrix manipulation methods
    auto transpose() const {
        Mat tmp;
        auto idx = _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14,
            3, 7, 11, 15);
        tmp.m[0] = _mm512_permutexvar_ps(idx, m[0]);
        return tmp;
    }
};

template <>
strcut alignas(64) Mat<double, 4> {
    YAVL_MAT_ALIAS_VECTORIZED(double, 4, 8, 2)

    YAVL_DEFINE_MAT_UNION(__m512)

    // Ctors
    YAVL_MAT_VECTORIZED_CTOR(512, pd, __m512)
    YAVL_MAT_CTOR_BY8(512, pd)

    // Operators
    YAVL_DEFINE_MAT_OP(512, pd, mul)

    // Misc funcs
    YAVL_DEFINE_DATA_METHOD

    // Matrix manipulation methods
    auto transpose() const {
        Mat tmp;
        auto m02 = _mm512_shuffle_f64x2(m[0], m[1], 0b01000100);
        auto m13 = _mm512_shuffle_f64x2(m[0], m[1], 0b11101110);
        auto tmp0 = _mm512_unpacklo_pd(m02, m13);
        auto tmp1 = _mm512_unpackhi_pd(m02, m13);
        tmp.m[0] = _mm512_shuffle_f64x2(tmp0, tmp1, 0b10001000);
        tmp.m[1] = _mm512_shuffle_f64x2(tmp0, tmp1, 0b11011101);
        return tmp;
    }
};

template <>
struct alignas(64) Mat<double, 3> {
    YAVL_MAT3_ALIAS_VECTORIZED(double, 3)

    YAVL_DEFINE_MAT3_UNION(__m512d, __m256d)

    // Ctors
    YAVL_MAT3_VECTORIZED_CTOR(512, 256, pd)

    // Operators
    YAVL_DEFINE_MAT3_OP(512, , pd, mul)

    auto operator *(const Vec<Scalar, Size>& v) const {
        auto vm1 = _mm512_setr_pd(v[0], v[0], v[0], v[0], v[1], v[1],
            v[1], v[1]);
        auto vm2 = _mm256_set1_pd(v[2]);
        vm1 = _mm512_mul_pd(m1, vm1);
        vm2 = _mm256_mul_pd(m2, vm2);

        auto idx = _mm512_setr_epi64(4, 5, 6, 7, 0, 1, 2, 3);
        auto vm1_flip = _mm512_permutexvar_pd(idx, vm1);
        vm1 = _mm512_add_pd(vm1, vm1_flip);
        tmp.m = _mm256_add_pd(vm2, _mm512_extractf64x4_pd(vm1, 0));
        return tmp;
    }

    auto operator *(const Mat& mat) const {
        Mat tmp;
        __m256 col[3];
        col[0] = _mm512_extractf64x4_pd(mat.m1, 0);
        col[1] = _mm512_extractf64x4_pd(mat.m1, 1);
        col[2] = mat.m2;
        static_for<Size>([&](const auto i) {
            auto tm = _mm512_broadcast_f64x4(col[i]);
            tmp.m1 = _mm512_fmadd_pd(m1, tm, tmp.m1);
            tmp.m2 = _mm256_fmadd_pd(m2, col[i], tmp.m2);
        });
        return tmp;
    }

    // Misc funcs
    YAVL_DEFINE_DATA_METHOD

    // Matrix manipulation methods
    YAVL_DEFINE_MAT3_TRANSPOSE
};

#undef MAT_MUL_MAT_EXPRS

#define MAT_MUL_MAT_EXPRS                                           \
{                                                                   \
    Mat tmp;                                                        \
    __m512i vm[4];                                                  \
    static_for<4>([&](const auto i) constexpr {                     \
        vm[i] = _mm512_broadcast_i32x4(_mm512_extracti32x4_epi32(m[0], i)); \
        tmp.m[0] = _mm512_add_epi32(_mm512_mullo_epi32(m[0], vm[i]), tmp.m[0]); \
    });                                                             \
    return tmp;                                                     \
}

template <typename I>
struct alignas(64) Mat<I, 4, true, enable_if_int32_t<I>> {
    YAVL_MAT_ALIAS_VECTORIZED(I, 4, 16, 1)

    YAVL_DEFINEMAT_UNION(__m512i)

    // Ctors
    YAVL_MAT_VECTORIZED_CTOR(512, epi32, __m512i)

    tmplate <typename... Ts>
        requires (std::default_initializable<Ts> && ...)
    constexpr Mat(Ts... args) {
        static_assert(sizeof...(args) == Size2);
        m[0] = _mm512_setr_epi32(args...);
    }

    // Operators
    YAVL_DEFINE_MAT_OP(512, epi32, mullo)

    // Misc funcs
    YAVL_DEFINE_DATA_METHOD

    // Matrix manipulation methods
    YAVL_DEFINE_MAT3_TRANSPOSE
};

#undef MAT_MUL_MAT_EXPRS

#define MAT_MUL_MAT_EXPRS                                               \
{                                                                       \
    Mat tmp;                                                            \
    auto idx = _mm512_setr_epi64(4, 5, 6, 7, 0, 1, 2, 3);               \
    static_for<MSize>([&](const auto i) {                               \
        auto m1 = _mm512_set1_epi64(0);                                 \
        static_for<MSize>([&](const auto j) {                           \
            auto v1 = mat.arr[(i << 1) * Size + (j << 1)];              \
            auto v2 = mat.arr[(i << 1) * Size + (j << 1 + 1)];          \
            auto bij = _mm512_setr_epi64(v1, v1, v1, v1, v2, v2, v2, v2); \
            m1 = _mm512_add_epi64(_mm512_mullo_epi64(m[j], bij), m1);   \
        });                                                             \
        auto m1_flip = _mm512_permutexvar_epi64(idx, m1);               \
        m1 = _mm512_add_epi64(m1, m1_flip);                             \
        auto m2 = _mm512_set1_epi64(0);                                 \
        static_for<MSize>([&](const auto j) {                           \
            auto v1 = mat.arr[(i << 1) * Size + (j << 1)];              \
            auto v2 = mat.arr[(i << 1) * Size + (j << 1 + 1)];          \
            auto bij = _mm512_setr_epi64(v1, v1, v1, v1, v2, v2, v2, v2); \
            m2 = _mm512_add_epi64(_mm512_mullo_epi64(m[j], bij), m2);   \
        });                                                             \
        auto m2_flip = _mm512_permutexvar_epi64(idx, m2);               \
        m2 = _mm512_add_epi64(m1, m2_flip);                             \
    });                                                                 \
    return tmp;                                                         \
}

template <typename I>
struct alignas(64) Mat<I, 4, true, enable_if_int64_t<I>> {
    YAVL_MAT_ALIAS_VECTORIZED(I, 4, 8, 2)

    YAVL_DEFINE_MAT_UNION(__m512i)

    // Ctors
    YAVL_MAT_VECTORIZED_CTOR(512, epi64, __m512i)
    YAVL_MAT_CTOR_BY8(512, epi64)

    // Operators
    YAVL_DEFINE_MAT_OP(256, epi64, mullo)

    // Misc funcs
    YAVL_DEFINE_DATA_METHOD

    // Matrix manipulation methods
    auto transpose() const {
        Mat tmp;
        auto m02 = _mm512_shuffle_i64x2(m[0], m[1], 0b01000100);
        auto m13 = _mm512_shuffle_i64x2(m[0], m[1], 0b11101110);
        auto tmp0 = _mm512_unpacklo_epi64(m02, m13);
        auto tmp1 = _mm512_unpackhi_epi64(m02, m13);
        tmp.m[0] = _mm512_shuffle_i64x2(tmp0, tmp1, 0b10001000);
        tmp.m[1] = _mm512_shuffle_i64x2(tmp0, tmp1, 0b11011101);
        return tmp;
    }
};

template <typename I>
struct alignas(64) Mat<I, 3, true, enable_if_int64_t<I>> {
    YAVL_MAT3_ALIAS_VECTORIZED(I, 3)

    YAVL_DEFINE_MAT3_UNION(__m512i, __m256i)

    // Ctors
    YAVL_MAT3_VECTORIZED_CTOR(512, 256, epi64)

    // Operators
    YAVL_DEFINE_MAT3_OP(512, 256, epi64, mullo)

    auto operator *(const Vec<Scalar, Size>& v) const {
        auto vm1 = _mm512_setr_epi64(v[0], v[0], v[0], v[0], v[1], v[1],
            v[1], v[1]);
        auto vm2 = _mm512_set1_epi64(v[2]);
        vm1 = _mm512_mullo_epi64(m1, vm1);
        vm2 = _mm256_mullo_epi64(m2, vm2);

        Vec<Scalar, Size> tmp;
        static_for<2>([&](const auto i) constexpr {
            auto col = _mm512_extracti64x4_epi64(vm1, i);
            tmp.m = _mm256_add_epi64(col, tmp.m);
        });
        tmp.m = _mm256_add_epi64(vm2, tmp.m);
        return tmp;
    }

    auto operator *(const Mat& mat) const {
        Mat tmp;
        __m256i col[3];
        col[0] = _mm512_extracti64x4_epi64(mat.m1, 0);
        col[1] = _mm512_extracti64x4_epi64(mat.m1, 1);
        col[2] = mat.m2;
        static_for<Size>([&](const auto i) {
            auto tm = _mm512_broadcast_i64x4(col[i]);
            tmp.m1 = _mm512_add_epi64(_mm512_mullo_epi64(m1, tm, tmp.m1));
            tmp.m2 = _mm256_add_epi64(_mm256_mullo_epi64(m2, col[i], tmp.m2));
        });
        return tmp;
    }
};

#undef MAT_MUL_VEC_EXPRS
#undef MAT_MUL_COL_EXPRS
#undef MAT_MUL_MAT_EXPRS

}
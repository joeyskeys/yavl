#pragma once

namespace yavl
{

// Common macros
#define YAVL_MAT_ALIAS_VECTORIZED(TYPE, N, INTRIN_N, REGI_N)            \
    YAVL_MAT_ALIAS(TYPE, N, INTRIN_N)                                   \
    static constexpr uint32_t MSize = REGI_N;                           \
    static constexpr bool vectorized = true;

#define YAVL_DEFINE_MAT_UNION(IT)                                       \
    union {                                                             \
        std::array<Scalar, IntrinSize * MSize> arr;                     \
        IT m[MSize];                                                    \
    };

#define YAVL_MAT_VECTORIZED_CTOR(BITS, IT, REGI_TYPE)                   \
    Mat() {                                                             \
        static_for<MSize>([&](const auto i) {                           \
            m[i] = _mm##BITS##_set1_##IT(static_cast<Scalar>(0));       \
        });                                                             \
    }                                                                   \
    template <typename V>                                               \
        requires std::default_initializable<V> && std::convertible_to<V, Scalar> \
    Mat(V v) {                                                          \
        static_for<MSize>([&](const auto i) {                           \
            m[i] = _mm##BITS##_set1_##IT(static_cast<Scalar>(v));       \
        });                                                             \
    }

#define MAT_MUL_SCALAR_EXPRS(BITS, IT)                                  \
{                                                                       \
    Mat tmp;                                                            \
    auto vs = _mm##BITS##_set1_##IT(s);                                 \
    static_for<MSize>([&](const auto i) {                               \
        tmp.m[i] = _mm##BITS##_mul_##IT(m[i], vs);                      \
    });                                                                 \
    return tmp;                                                         \
}

#define MAT_MUL_ASSIGN_SCALAR_EXPRS(BITS, IT)                           \
{                                                                       \
    auto vs = _mm##BITS##_set1_##IT(s);                                 \
    static_for<MSize>([&](const auto i) {                               \
        m[i] = _mm##BITS##_mul_##IT(m[i], vs);                          \
    });                                                                 \
    return *this;                                                       \
}

#define OP_VEC_EXPRS(BITS, OP, AT, NAME, IT)                            \
{                                                                       \
    return Vec<Scalar, Size>(_mm##BITS##_##NAME##_##IT(m, v.m));        \
}

#define OP_VEC_ASSIGN_EXPRS(BITS, OP, AT, NAME, IT)                     \
{                                                                       \
    auto vv = _mm##BITS##_##NAME##_##IT(m, v.m);                        \
    m = _mm##BITS##_##NAME##_##IT(m, vv);                               \
    _mm##BITS##_store_##IT(arr, m);                                     \
}

#define OP_SCALAR_EXPRS(BITS, OP, NAME, IT)                             \
{                                                                       \
    auto vv = _mm##BITS##_set1_##IT(v);                                 \
    return Vec<Scalar, Size>(_mm##BITS##_##NAME##_##IT(m, vv));         \
}

#define OP_SCALAR_ASSIGN_EXPRS(BITS, OP, NAME, IT)                      \
{                                                                       \
    auto vv = _mm##BITS##_set1_##IT(v);                                 \
    m = _mm##BITS##_##NAME##_##IT(m, vv);                               \
    _mm##BITS##_store_##IT(arr, m);                                     \
    return *this;                                                       \
}

#define OP_FRIEND_SCALAR_EXPRS(BITS, OP, AT, NAME, IT)                  \
{                                                                       \
    auto vv = _mm##BITS##_set1_##IT(s);                                 \
    return Vec<Scalar, Size>(_mm##BITS##_##NAME##_##IT(vv, v.m));       \
}

#define YAVL_MAT3_ALIAS_VECTORIZED(TYPE, N)                             \
    using Scalar = TYPE;                                                \
    static constexpr uint32_t Size = 3;                                 \
    static constexpr uint32_t Size2 = 9;                                \
    static constexpr bool vectorized = true;


#define YAVL_DEFINE_MAT3_UNION(IT1, IT2)                                \
    union {                                                             \
        std::array<Scalar, 12> arr;                                     \
        struct {                                                        \
            IT1 m1;                                                     \
            IT2 m2;                                                     \
        };                                                              \
    };

#define YAVL_MAT3_VECTORIZED_CTOR(BITS1, BITS2, IT)                     \
    Mat() {                                                             \
        m1 = _mm##BITS1##_set1_##IT(static_cast<Scalar>(0));            \
        m2 = _mm##BITS2##_set1_##IT(static_cast<Scalar>(0));            \
    }                                                                   \
    template <typename V>                                               \
        requires std::default_initializable<V> && std::convertible_to<V, Scalar> \    Mat(V v) {
        m1 = _mm##BITS1##_set1_##IT(static_cast<Scalar>(v));            \
        m2 = _mm##BITS2##_set1_##IT(static_cast<Scalar>(v));            \
    }                                                                   \
    constexpr Mat(const Scalar t0, const Scalar t1, const Scalar t2,    \
        const Scalar t3, const Scalar t4, const Scalar t5,              \
        const Scalar t6, const Scalar t7, const Scalar t8)              \
    {                                                                   \
        m1 = _mm##BITS1##_setr_##IT(t0, t1, t2, 0, t3, t4, t5, 0);      \
        m2 = _mm##BITS2##_setr_##IT(t6, t7, t8, 0);                     \
    }

#define MAT3_MUL_SCALAR_EXPRS(BITS1, BITS2, IT, MUL)                    \
{                                                                       \
    Mat tmp;                                                            \
    auto vm1 = _mm##BITS1##_set1_##IT(s);                               \
    auto vm2 = _mm##BITS2##_set1_##IT(s);                               \
    tmp.m1 = _mm##BITS1##_##MUL##_##IT(m1, vm1);                        \
    tmp.m2 = _mm##BITS2##_##MUL##_##IT(m2, vm2);                        \
    return tmp;                                                         \
}

#define MAT3_MUL_ASSIGN_SCALAR_EXPRS(BITS1, BITS2, IT, MUL)             \
{                                                                       \
    auto vm1 = _mm##BITS1##_set1_##IT(s);                               \
    auto vm2 = _mm##BITS2##_set1_##IT(s);                               \
    m1 = _mm##BITS1##_##MUL##_##IT(m1, vm1);                            \
    m2 = _mm##BITS2##_##MUL##_##IT(m2, vm2);                            \
    return *this;                                                       \
}

#define YAVL_DEFINE_MAT3_COMMON_OP(BITS1, BITS2, IT, MUL)               \
    YAVL_DEFINE_MAT_INDEX_OP                                            \
    auto operator *(const Scalar s) const {                             \
        MAT3_MUL_SCALAR_EXPRS(BITS1, BITS2, IT, MUL)                    \
    }                                                                   \
    auto operator *=(const Scalar s) const {                            \
        MAT3_MUL_SCALAR_EXPRS(BITS1, BITS2, IT, MUL)                    \
    }                                                                   \
    auto operator *(const Col<Scalar, Size>& v) const {                 \
        auto vec = Vec<Scalar, Size>{v.m};                              \
        return operator *(vec);                                         \
    }                                                                   \
    auto operator *=(const Mat& mat) const {                            \
        Mat tmp = *this * mat;                                          \
        *this = tmp;                                                    \
        return *this;                                                   \
    }

#define YAVL_DEFINE_MAT3_TRANSPOSE                                      \
    auto transpose() const {                                            \
        Mat<Scalar, 4> mat4;                                            \
        memcpy(mat4.data(), data(), 12 * sizeof(Scalar));               \
        mat4 = mat4.transpose();                                        \
        memcpy(data(), mat4.data(), 12 * sizeof(Scalar));               \
    }

// Common methods
namespace detail
{

// TODO: find out whether a shuffle and a hadd is better than two adds
#define MAT2_MUL_VEC_COMMON_EXPRS(TYPE, HADD)                           \
    auto tmpv4a = Vec<TYPE, 4>(mat.m[0]);                               \
    auto tmpv4b = Vec<TYPE, 4>(vec[0], vec[0], vec[1], vec[1]);         \
    auto vm = (tmpv4a * tmpv4b).shuffle<0, 2, 1, 3>().m;                \
    tmpv4a = Vec<TYPE, 4>(HADD(vm, vm));                                \
    return Vec<TYPE, 2>(tmpv4a[0], tmpv4a[1]);

static inline Vec<float, 2> mat2_mul_vec_f32(const Mat<float, 2>& mat,
    const Vec<float, 2>& vec)
{
    MAT2_MUL_VEC_COMMON_EXPRS(float, _mm_hadd_ps)
}

template <typename T>
    requires yavl::is_int32_v<T>
static inline Vec<T, 2> mat2_mul_vec_i32(const Mat<T, 2>& mat,
    const Vec<T, 2>& vec)
{
    MAT2_MUL_VEC_COMMON_EXPRS(T, _mm_hadd_epi32)
}

static inline Vec<double, 2> mat2_mul_vec_f64(const Mat<double, 2>& mat,
    const Vec<double, 2>& vec)
{
    MAT2_MUL_VEC_COMMON_EXPRS(double, _mm256_hadd_pd)
}

template <typename T>
    requires yavl::is_int64_v<T>
static inline Vec<T, 2> mat2_mul_vec_i64(const Mat<T, 2>& mat,
    const Vec<T, 2>& vec)
{
    auto tmpv4a = Vec<T, 4>(mat.m[0]);
    auto tmpv4b = Vec<T, 4>(vec[0], vec[0], vec[1], vec[1]);
    auto vm = Vec<T, 4>((tmpv4a * tmpv4b).shuffle<0, 2, 1, 3>().m);
    return Vec<T, 2>(vm[0] + vm[1], vm[2] + vm[3]);
}

template <typename T>
    requires yavl::is_int64_v<T>
static inline Vec<T, 2> mat2_mul_vec_i64(const Mat<T, 2>& mat,
    const Vec<T, 2>& vec)
{
    return Vec<T, 2>();
}

template <typename T, uint32_t N>
static inline Vec<T, N> sse42_mat_mul_vec_impl(const Mat<T, N>& mat, const Vec<T, N>& vec) {
    // Check comments in avx512 impl
    Vec<T, N> tmp;
    if constexpr (N == 2) {
        if constexpr (std::is_floating_point_v<T>) {
            tmp = mat2_mul_vec_f32(mat, vec);
        }
        else {
            tmp = mat2_mul_vec_i32(mat, vec);
        }
    }
    else {
        static_for<Mat<T, N>::MSize>([&](const auto i) {
            if constexpr (std::is_floating_point_v<T>) {
                auto vm = _mm_set1_ps(vec[i]);
                // If we're using this impl, it mean FMA not available
                //tmp.m = _mm_fmadd_ps(mat.m[i], vm, tmp.m);
                tmp.m = _mm_add_ps(_mm_mul_ps(mat.m[i], vm), tmp.m);
            }
            else {
                auto vm = _mm_set1_ps(vec[i]);
                tmp.m = _mm_add_epi32(tmp.m, _mm_mul_epi32(mat.m[i], vm));
            }
        });
    }
    return tmp;
}

template <typename T, uint32_t N>
static inline Vec<T, N> avx_mat_mul_vec_impl(const Mat<T, N>& mat, const Vec<T, N>& vec) {
    // Check comments in avx512 impl
    Vec<T, N> tmp;
    if constexpr (N == 2) {
        if constexpr (std::is_floating_point_v<T>) {
            if constexpr (sizeof(T) == 4)
                tmp = mat2_mul_vec_f32(mat, vec);
            else
                tmp = mat2_mul_vec_f64(mat, vec);
        }
        else {
            if constexpr (sizeof(T) == 4)
                tmp = mat2_mul_vec_i32(mat, vec);
            else
                tmp = mat2_mul_vec_i64(mat, vec);
        }
    }
    else {
        if constexpr (sizoef(T) == 4) {
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
                else {
                    auto vm = _mm256_set1_epi32(0);
                    auto b = _mm256_setr_epi32(v1, v1, v1, v1, v2, v2, v2, v2);
                    vm = _mm256_add_epi32(vm256, _mm256_mullo_epi32(mat.m[i], vm));
                    auto vm_flip = _mm256_permute2x128_si256(vm, vm, 1);
                    tmp.m = _mm256_extracti128_si256(_mm256_add_epi32(vm, vm_flip), 0);
                }
            });
        }
        else {
            static_for<Mat<T, N>::MSize>([&](const auto i) {
                if constexpr (std::is_floating_point_v<T>) {
                    auto vm = _mm256_set1_pd(vec[i]);
                    tmp.m = _mm256_fmadd_pd(mat.m[i], vm, tmp.m);
                }
                else {
                    auto vm = _mm256_set1_pd(vec[i]);
                    tmp.m = _mm256_add_epi64(tmp.m, _mm256_mul_epi64(mat.m[i], vm));
                }
            });
        }
    }
    return tmp;
}

} // namespace detail

template <>
struct Col<float, 4> {
    YAVL_TYPE_ALIAS(float, 4, 4)

    Scalar* arr;
    __m128 m;

    Col(const Scalar* d)
        : arr(const_cast<Scalar*>(d))
    {
        m = _mm_load_ps(arr);
    }

    // Miscs
    YAVL_DEFINE_COL_MISC_FUNCS

    // Operators
    YAVL_DEFINE_COL_BASIC_FP_OP(, ps, ps)
};

template <>
struct Col<float, 3> {
    YAVL_TYPE_ALIAS(float, 3, 4)

    Scalar* arr;
    __m128 m;

    Col(const Scalar* d)
        : arr(const_cast<Scalar*>(d))
    {
        m = _mm_setr_ps(arr[0], arr[1], arr[2], 0);
    }

    // Miscs
    YAVL_DEFINE_COL_MISC_FUNCS

    // Operators
    YAVL_DEFINE_COL_BASIC_FP_OP(, ps, ps)
};

// Include possible common impl
#if defined(YAVL_X86_SSE42)
    #include <yavl/mat/mat2_sse42.h>
#endif

#if defined(YAVL_X86_AVX)
    #include <yavl/mat/mat2_avx.h>
#endif

// Cascaded including, using max bits intrinsic set available
#if defined(YAVL_X86_AVX512ER) && !defined(YAVL_FORCE_SSE_MAT) && !defined(YAVL_FORCE_AVX_MAT)
    #include <yavl/mat/mat_avx512.h>
    #include <yavl/mat/mat3_avx.h>
    #include <yavl/mat/mat3_avx2.h>
#elif defined(YAVL_X86_AVX) && defined(YAVL_X86_AVX2) && !defined(YAVL_FORCE_SSE_MAT)
    #include <yavl/mat/mat_avx.h>
    #include <yavl/mat/mat_avx2.h>
    #include <yavl/mat/mat3_avx.h>
    #include <yavl/mat/mat3_avx2.h>
#elif defined(YAVL_X86_SSE42)
    #include <yavl/mat/mat_sse42.h>
#endif

#undef MAT_MUL_SCAlAR_EXPRS
#undef MAT_MUL_ASSIGN_SCALAR_EXPRS
#undef OP_VEC_EXPRS
#undef OP_VEC_ASSIGN_EXPRS
#undef OP_SCALAR_EXPRS
#undef OP_SCALAR_ASSIGN_EXPRS
#undef OP_FRIEND_SCALAR_EXPRS

} // namespace yavl
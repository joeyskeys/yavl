#pragma once

// Common macros
#define YAVL_MAT_ALIAS_VECTORIZED(TYPE, N, INTRIN_N, REGI_N)            \
    YAVL_MAT_ALIAS(TYPE, N, INTRIN_N)                                   \
    static constexpr uint32_t MSize = REGI_N;                           \
    static constexpr bool vectorized = true;

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
    {                                                                   \
        Mat tmp;                                                        \
        auto vs = _mm##BITS##_set1_##IT(s);                             \
        static_for<MSize>([&](const auto i) {                           \
            tmp.m[i] = _mm##BITS##_mul_##IT(m[i], vs);                  \
        });                                                             \
        return tmp;                                                     \
    }

#define MAT_MUL_ASSIGN_SCALAR_EXPRS(BITS, IT)                           \
    {                                                                   \
        auto vs = _mm##BITS##_set1_##IT(s);                             \
        static_for<MSize>([&](const auto i) {                           \
            m[i] = _mm##BITS##_mul_##IT(m[i], vs);                      \
        });                                                             \
        return *this;                                                   \
    }

#define OP_VEC_EXPRS(BITS, OP, AT, NAME, IT)                            \
    {                                                                   \
        return Vec<Scalar, Size>(_mm##BITS##_##NAME##_##IT(m, v.m));    \
    }

#define OP_VEC_ASSIGN_EXPRS(BITS, OP, AT, NAME, IT)                     \
    {                                                                   \
        auto vv = _mm##BITS##_##NAME##_##IT(m, v.m);                    \
        m = _mm##BITS##_##NAME##_##IT(m, vv);                           \
        _mm##BITS##_store_##IT(arr, m);                                 \
    }

#define OP_SCALAR_EXPRS(BITS, OP, NAME, IT)                             \
    {                                                                   \
        auto vv = _mm##BITS##_set1_##IT(v);                             \
        return Vec<Scalar, Size>(_mm##BITS##_##NAME##_##IT(m, vv));     \
    }

#define OP_SCALAR_ASSIGN_EXPRS(BITS, OP, NAME, IT)                      \
    {                                                                   \
        auto vv = _mm##BITS##_set1_##IT(v);                             \
        m = _mm##BITS##_##NAME##_##IT(m, vv);                           \
        _mm##BITS##_store_##IT(arr, m);                                 \
        return *this;                                                   \
    }

#define OP_FRIEND_SCALAR_EXPRS(BITS, OP, AT, NAME, IT)                  \
    {                                                                   \
        auto vv = _mm##BITS##_set1_##IT(s);                             \
        return Vec<Scalar, Size>(_mm##BITS##_##NAME##_##IT(vv, v.m));   \
    }

// Common methods
namespace detail
{

static inline Vec<float, 2> mat2_mul_vec_f32(const Mat<float, 2>& mat,
    const Vec<float, 2>& vec)
{
    auto tmpv4a = Vec<float, 4>(mat.m[0]);
    auto tmpv4b = Vec<float, 4>(vec[0], vec[0], vec[1], vec[1]);
    auto vm = (tmpv4a * tmpv4b).shuffle<0, 2, 1, 3>().m;
    tmpv4a = Vec<float, 4>(_mm_hadd_ps(vm, vm));
    return Vec<float, 2>(tmpv4a[0], tmpv4a[1]);
}

template <typename T>
    requires yavl::is_int32_v<T>
static inline Vec<T, 2> mat2_mul_vec_i32(const Mat<float, 2>& mat,
    const Vec<T, 2>& vec)
{
    // TODO: Blabla
    return Vec<T, 2>();
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

// Cascaded including, using max bits intrinsic set available
#if defined(YAVL_X86_AVX512ER) && !defined(YAVL_FORCE_SSE_MAT) && !defined(YAVL_FORCE_AVX_MAT)
    #include <yavl/mat/mat_avx512.h>
#elif defined(YAVL_X86_AVX) && defined(YAVL_X86_AVX2) && !defined(YAVL_FORCE_SSE_MAT)
    #include <yavl/mat/mat_avx.h>
    #include <yavl/mat/mat_avx2.h>
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
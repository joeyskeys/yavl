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

// Cascaded including, using max bits intrinsic set available
#if defined(YAVL_X86_AVX512ER)
    #include <yavl/mat/mat_avx512.h>
#elif defined(YAVL_x86_AVX) && defined(YAVL_X86_AVX2)
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
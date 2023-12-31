#pragma once

// Common macros
#define YAVL_VEC4_MEMBERS                                               \
    struct {                                                            \
        Scalar x, y, z, w;                                              \
    };                                                                  \
    struct {                                                            \
        Scalar r, g, b, a;                                              \
    };                                                                  \
    std::array<Scalar, Size> arr;

#define YAVL_VEC3_MEMBERS                                               \
    struct {                                                            \
        Scalar x, y, z;                                                 \
    };                                                                  \
    struct {                                                            \
        Scalar r, g, b;                                                 \
    };                                                                  \
    std::array<Scalar, Size> arr;

#define YAVL_VEC2_MEMBERS                                               \
    struct {                                                            \
        Scalar x, y;                                                    \
    };                                                                  \
    struct {                                                            \
        Scalar r, g;                                                    \
    };                                                                  \
    std::array<Scalar, Size> arr;

#define YAVL_VEC_ALIAS_VECTORIZED(TYPE, N, INTRIN_N)                    \
    YAVL_TYPE_ALIAS(TYPE, N, INTRIN_N)                                  \
    static constexpr bool vectorized = true;

#define YAVL_VECTORIZED_CTOR(BITS, IT, REGI_TYPE)                       \
    Vec() : m(_mm##BITS##_set1_##IT(static_cast<Scalar>(0))) {}         \
    template <typename V>                                               \
        requires std::default_initializable<V> && std::convertible_to<V, Scalar> \
    Vec(V v) : m(_mm##BITS##_set1_##IT(static_cast<Scalar>(v))) {}      \
    template <typename ...Ts>                                           \
        requires (std::default_initializable<Ts> && ...) &&             \
            (std::convertible_to<Ts, Scalar> && ...)                    \
    constexpr Vec(Ts... args) {                                         \
        static_assert(sizeof...(args) > 1);                             \
        if constexpr (sizeof...(Ts) == IntrinSize - 1)                  \
            m = _mm##BITS##_setr_##IT(args..., 0);                      \
        else                                                            \
            m = _mm##BITS##_setr_##IT(args...);                         \
    }                                                                   \
    Vec(const REGI_TYPE val) : m(val) {}

#define COPY_ASSIGN_EXPRS(BITS, IT)                                     \
    {                                                                   \
        _mm##BITS##_store_##IT(arr.data(), b.m);                        \
        return *this;                                                   \
    }

#define OP_VEC_EXPRS(BITS, OP, AT, NAME, IT)                            \
    return Vec(_mm##BITS##_##NAME##_##IT(m, v.m));

#define OP_VEC_ASSIGN_EXPRS(BITS, OP, AT, NAME, IT)                     \
    m = _mm##BITS##_##NAME##_##IT(m, v.m);                              \
    return *this;

#define OP_SCALAR_EXPRS(BITS, OP, AT, NAME, IT)                         \
    auto vv = _mm##BITS##_set1_##IT(v);                                 \
    return Vec(_mm##BITS##_##NAME##_##IT(m, vv));

#define OP_SCALAR_ASSIGN_EXPRS(BITS, OP, AT, NAME, IT)                  \
    {                                                                   \
        auto vv = _mm##BITS##_set1_##IT(v);                             \
        m = _mm##BITS##_##NAME##_##IT(m, vv);                           \
        return *this;                                                   \
    }

#define OP_FRIEND_SCALAR_EXPRS(BITS, OP, AT, NAME, IT)                  \
    auto vv = _mm##BITS##_set1_##IT(s);                                 \
    return Vec(_mm##BITS##_##NAME##_##IT(vv, v.m));

#if defined(YAVL_X86_FMA)
#define MULADD(BITS, IT, A, B, C) _mm##BITS##_fmadd_##IT(A, B, C)
#define MULSUB(BITS, IT, A, B, C) _mm##BITS##_fmsub_##IT(A, B, C)
#else
#define MULADD(BITS, IT, A, B, C) _mm##BITS##_add_##IT(_mm##BITS##_mul_##IT(A, B), C)
#define MULSUB(BITS, IT, A, B, C) _mm##BITS##_sub_##IT(_mm##BITS##_mul_##IT(A, B), C)
#endif

#define MATH_LERP_SCALAR_EXPRS(BITS, IT)                                \
    {                                                                   \
        auto vomt = _mm##BITS##_set1_##IT(1 - t);                       \
        auto vt = _mm##BITS##_set1_##IT(t);                             \
        auto t1 = _mm##BITS##_mul_##IT(b.m, vt);                        \
        auto ret = MULADD(BITS, IT, vomt, m, t1);                       \
        return Vec(ret);                                                \
    }

#define MATH_LERP_VEC_EXPRS(BITS, IT)                                   \
    {                                                                   \
        Vec vomt = 1 - t;                                               \
        auto t1 = _mm##BITS##_mul_##IT(b.m, t.m);                       \
        auto ret = MULADD(BITS, IT, vomt.m, m, t1);                     \
        return Vec(ret);                                                \
    }

// Common methods
namespace detail
{

template <typename Vec, typename ...Ts>
inline Vec shuffle_impl(const Vec& v, Ts ...args) {
    static_assert(sizeof...(Ts) == Vec::Size);
    static_assert(sizeof...(Ts) > 1 && sizeof...(Ts) < 5);

    #if defined(YAVL_X86_AVX)
        __m128i i;
        if constexpr (sizeof...(args) == 4)
            i = _mm_setr_epi32(args...);
        else if constexpr (sizeof...(args) == 3)
            i = _mm_setr_epi32(args..., 0);
        else if constexpr (sizeof...(args) == 2) {
            i = _mm_setr_epi32(args..., 0, 0);
            i = _mm_shuffle_epi32(i, 0b11011100);
        }

        if constexpr (std::is_floating_point_v<typename Vec::Scalar>) {
            if constexpr (sizeof...(args) > 2 && yavl::is_float_v<typename Vec::Scalar>)
                return Vec(_mm_permutevar_ps(v.m, i));
            else if constexpr (sizeof...(args) > 2) {
                // _mm256_permuatevar_pd don't work like expected
                base_shuffle_impl(v, args...);
            }
            else
                return Vec(_mm_permutevar_pd(v.m, _mm_slli_epi64(i, 1)));
        }
        else {
            if constexpr (sizeof...(args) > 2)
                return Vec(_mm_castps_si128(_mm_permutevar_ps(_mm_castsi128_ps(v.m), i)));
            else
                return Vec(_mm_castpd_si128(_mm_permutevar_pd(_mm_castsi128_pd(v.m),
                    _mm_slli_epi64(i, 1))));
        }
    #else
        base_shuffle_impl(v, args...);
    #endif
}

}

#define MISC_SHUFFLE_EXPRS                                              \
    {                                                                   \
        return Vec(detail::shuffle_impl(*this, args...));               \
    }

#define YAVL_DEFINE_CROSS_FUNC(BITS, IT)                                \
    inline auto cross(const Vec& b) const {                             \
        auto t1 = shuffle<1, 2, 0>();                                   \
        auto t2 = b.shuffle<2, 0, 1>();                                 \
        auto t3 = shuffle<2, 0, 1>() * b.shuffle<1, 2, 0>();            \
        auto ret = MULSUB(BITS, IT, t1.m, t2.m, t3.m);                  \
        return Vec(ret);                                                \
    }

#if defined(YAVL_X86_SSE42)
    #include <yavl/vec/vec_sse42.h>
#endif

#if defined(YAVL_X86_AVX)
    #include <yavl/vec/vec_avx.h>
#endif

#if defined(YAVL_X86_AVX2)
    #include <yavl/vec/vec_avx2.h>
#endif

#undef COPY_ASSIGN_EXPRS
#undef OP_VEC_EXPRS
#undef OP_VEC_ASSIGN_EXPRS
#undef OP_SCALAR_EXPRS
#undef OP_SCALAR_ASSIGN_EXPRS
#undef OP_FRIEND_SCALAR_EXPRS
#undef MISC_SHUFFLE_EXPRS
#undef MATH_LERP_SCALAR_EXPRS
#undef MATH_LERP_VEC_EXPRS
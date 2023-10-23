#pragma once

// Common macros
#define YAVL_VEC_ALIAS_VECTORIZED(TYPE, N, INTRIN_N)                    \
    YAVL_VEC_ALIAS(TYPE, N, INTRIN_N)                                   \
    static constexpr bool vectorized = true;

#define YAVL_VECTORIZED_CTOR(BITS, INTRIN_TYPE, REGI_TYPE)              \
    Vec() : m(_mm##BITS##_set1_##INTRIN_TYPE(static_cast<Scalar>(0))) {} \
    template <typename V>                                               \
        requires std::default_initializable<V> && std::convertible_to<V, Scalar> \
    Vec(V v) : m(_mm##BITS##_set1_##INTRIN_TYPE(static_cast<Scalar>(v))) {} \
    template <typename ...Ts>                                           \
        requires (std::default_initializable<Ts> && ...) &&             \
            (std::convertible_to<Ts, Scalar> && ...)                    \
    constexpr Vec(Ts... args) {                                         \
        static_assert(sizeof...(args) > 1);                             \
        if constexpr (sizeof...(Ts) == IntrinSize - 1)                  \
            m = _mm##BITS##_setr_##INTRIN_TYPE(args..., 0);             \
        else                                                            \
            m = _mm##BITS##_setr_##INTRIN_TYPE(args...);                \
    }                                                                   \
    Vec(const REGI_TYPE val) : m(val) {}

#define COPY_ASSIGN_EXPRS(BITS, INTRIN_TYPE)                            \
    {                                                                   \
        _mm##BITS##_store_##INTRIN_TYPE(arr.data(), b.m);               \
        return *this;                                                   \
    }

#define OP_VEC_EXPRS(BITS, OP, NAME, INTRIN_TYPE)                       \
    return Vec(_mm##BITS##_##NAME##_##INTRIN_TYPE(m, v.m));

#define OP_VEC_ASSIGN_EXPRS(BITS, OP, NAME, INTRIN_TYPE)                \
    m = _mm##BITS##_##NAME##_##INTRIN_TYPE(m, v.m);                     \
    return *this;

#define OP_SCALAR_EXPRS(BITS, OP, NAME, INTRIN_TYPE)                    \
    auto vv = _mm##BITS##_set1_##INTRIN_TYPE(v);                        \
    return Vec(_mm##BITS##_##NAME##_##INTRIN_TYPE(m, vv));

#define OP_SCALAR_ASSIGN_EXPRS(BITS, OP, NAME, INTRIN_TYPE)             \
    auto vv = _mm##BITS##_set1_##INTRIN_TYPE(v);                        \
    m = _mm##BITS##_##NAME##_##INTRIN_TYPE(m, vv);                      \
    return *this;

#define OP_FRIEND_SCALAR_EXPRS(BITS, OP, NAME, INTRIN_TYPE)             \
    auto vv = _mm##BITS##_set1_##INTRIN_TYPE(s);                        \
    return Vec(_mm##BITS##_##NAME##_##INTRIN_TYPE(vv, v.m));

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
                if constexpr (sizeof...(args) > 2)
                    return Vec(_mm_permutevar_ps(v.m, i));
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
            MISC_BASE_SHUFFLE_EXPRS
        #endif
    }
}

#define MISC_SHUFFLE_EXPRS                                              \
    {                                                                   \
        return Vec(detail::shuffle_impl(*this, args...));               \
    }

#if defined(YAVL_X86_SSE42)
#   include <yavl/vec_sse42.h>
#endif

#if defined(YAVL_X86_AVX)
#   include <yavl/vec_avx.h>
#endif

namespace yavl
{

template <typename T>
using Vec2 = Vec<T, 2>;

template <typename T>
using _Vec2 = Vec<T, 2, false>;

template <typename T>
using Vec3 = Vec<T, 3>;

template <typename T>
using _Vec3 = Vec<T, 3, false>;

template <typename T>
using Vec4 = Vec<T, 4>;

template <typename T>
using _Vec4 = Vec<T, 4, false>;

template <typename T>
using Vec2f = Vec2<float>;

template <typename T>
using Vec2d = Vec2<double>;

template <typename T>
using Vec2i = Vec2<int>;

template <typename T>
using Vec2u = Vec2<uint32_t>;

template <typename T>
using Vec3f = Vec3<float>;

template <typename T>
using Vec3d = Vec3<double>;

template <typename T>
using Vec3i = Vec3<int>;

template <typename T>
using Vec3u = Vec3<uint32_t>;

template <typename T>
using Vec4f = Vec4<float>;

template <typename T>
using Vec4d = Vec4<double>;

template <typename T>
using Vec4i = Vec4<int>;

template <typename T>
using Vec4u = Vec4<uint32_t>;

template <typename T>
using _Vec2f = _Vec2<float>;

template <typename T>
using _Vec2d = _Vec2<double>;

template <typename T>
using _Vec2i = _Vec2<int>;

template <typename T>
using _Vec2u = _Vec2<uint32_t>;

template <typename T>
using _Vec3f = _Vec3<float>;

template <typename T>
using _Vec3d = _Vec3<double>;

template <typename T>
using _Vec3i = _Vec3<int>;

template <typename T>
using _Vec3u = _Vec3<uint32_t>;

template <typename T>
using _Vec4f = _Vec4<float>;

template <typename T>
using _Vec4d = _Vec4<double>;

template <typename T>
using _Vec4i = _Vec4<int>;

template <typename T>
using _Vec4u = _Vec4<uint32_t>;
}

#undef MISC_SHUFFLE_EXPRS
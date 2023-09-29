#pragma once

#include <concepts>

namespace yavl
{

template <>
struct alignas(16) Vec<float, 4> {
    YAVL_VEC_ALIAS(float, 4)

    static constexpr bool vectorized = true;

    union {
        struct {
            Scalar x, y, z, w;
        };
        struct {
            Scalar r, g, b, a;
        };
        
        std::array<Scalar, Size> arr;
        __m128 m;
    };

    // Ctors
    Vec() : m(_mm_set1_ps(0)) {}

    template <typename V>
        requires std::default_initializable<V> && std::convertible_to<V, Scalar>
    Vec(V v) : m(_mm_set1_ps(static_cast<Scalar>(v))) {}

    template <typename ...Ts>
        requires (std::default_initializable<Ts> && ...) && (std::convertible_to<Ts, Scalar> && ...)
    Vec(Ts... args) {
            m = _mm_setr_ps(args...);
    }

    Vec(const __m128 val) : m(val) {}

    // Operators
#define VEC_EXPRS(OP, NAME)                                             \
    return Vec(_mm_##NAME##_ps(m, v.m));

#define VEC_ASSIGN_EXPRS(OP, NAME)                                      \
    m = _mm_##NAME##_ps(m, v.m);                                        \
    return *this;

#define SCALAR_EXPRS(OP, NAME)                                          \
    auto vv = _mm_set1_ps(v);                                           \
    return Vec(_mm_##NAME##_ps(m, vv));

#define SCALAR_ASSIGN_EXPRS(OP, NAME)                                   \
    auto vv = _mm_set1_ps(v);                                           \
    m = _mm_##NAME##_ps(m, vv);                                         \
    return *this;

#define FRIEND_SCALAR_EXPRS(OP, NAME)                                   \
    auto vv = _mm_set1_ps(s);                                           \
    return Vec(_mm_##NAME##_ps(vv, v.m));

    YAVL_DEFINE_OP(+, add)
    YAVL_DEFINE_OP(-, sub)
    YAVL_DEFINE_OP(*, mul)
    YAVL_DEFINE_FRIEND_OP(*, mul)
    YAVL_DEFINE_OP(/, div)
    YAVL_DEFINE_FRIEND_OP(/, div)

#undef VEC_EXPRS
#undef VEC_ASSIGN_EXPRS
#undef SCALAR_EXPRS
#undef SCALAR_ASSING_EXPRS
#undef FRIEND_SCALAR_EXPRS
};

}
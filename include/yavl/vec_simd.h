#pragma once

// Common macros
#define YAVL_VEC_ALIAS_VECTORIZED(TYPE, N, INTRIN_N)                    \
    YAVL_VEC_ALIAS(TYPE, N, INTRIN_N)                                   \
    static constexpr bool vectorized = true;

#define YAVL_VECTORIZED_CTOR(INTRIN_TYPE, REGI_TYPE)                    \
    Vec() : m(_mm_set1_##INTRIN_TYPE(static_cast<Scalar>(0))) {}        \
    template <typename V>                                               \
        requires std::default_initializable<V> && std::convertible_to<V, Scalar> \
    Vec(V v) : m(_mm_set1_##INTRIN_TYPE(static_cast<Scalar>(v))) {}     \
    template <typename ...Ts>                                           \
        requires (std::default_initializable<Ts> && ...) &&             \
            (std::convertible_to<Ts, Scalar> && ...)                    \
    constexpr Vec(Ts... args) {                                         \
        static_assert(sizeof...(args) > 1);                             \
        if constexpr (sizeof...(Ts) == IntrinSize - 1)                  \
            m = _mm_setr_##INTRIN_TYPE(args..., 0);                     \
        else                                                            \
            m = _mm_setr_##INTRIN_TYPE(args...);                        \
    }                                                                   \
    Vec(const REGI_TYPE val) : m(val) {}

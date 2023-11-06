#pragma once

#include <concepts>
#include <stdint.h>
#include <type_traits>
#include <utility>

namespace yavl
{

// Macros
#define YAVL_TYPE_ALIAS(TYPE, N, INTRIN_N)                              \
    using Scalar = std::decay_t<TYPE>;                                  \
    static constexpr uint32_t Size = N;                                 \
    static constexpr uint32_t IntrinSize = INTRIN_N;                    \
    static constexpr uint32_t BitSize = sizeof(Scalar) * N * 8;

// Templates
template <int... indices, typename Func>
inline void static_for(const Func& func, std::integer_sequence<int, indices...> sequence) {
    (func(indices), ...);
}

template <int N, typename Func>
inline void static_for(const Func& func) {
    static_for(func, std::make_integer_sequence<int, N>{});
}

template <int N, typename Func, typename... Ts>
struct SplitHelper {
    static inline void even_split(const Func& func, Ts... args) {
        static_assert(sizeof...(Ts) % N == 0);
        constexpr auto div = sizeof...(Ts) / N;
        SplitHelper<div, Func, Ts...>::even_split(func, args...);
    }
};

template <typename Func, typename T1, typename T2, typename... Ts>
struct SplitHelper<2, Func, T1, T2, Ts...> {
    static inline void even_split(const Func& func, const T1& t1,
        const T2& t2, Ts... args)
    {
        func(t1, t2);
        if constexpr (sizeof...(Ts) >= 2)
            SplitHelper<2, Func, Ts...>::even_split(func, args...);
    }
};

template <typename Func, typename T1, typename T2, typename T3,
    typename... Ts>
struct SplitHelper<3, Func, T1, T2, T3, Ts...> {
    static inline void even_split(const Func& func, const T1& t1,
        const T2& t2, const T3& t3, Ts... args)
    {
        func(t1, t2, t3);
        if constexpr (sizeof...(Ts) >= 3)
            SplitHelper<3, Func, Ts...>::even_split(func, args...);
    }
};

template <typename Func, typename T1, typename T2,
    typename T3, typename T4, typename... Ts>
struct SplitHelper<4, Func, T1, T2, T3, T4, Ts...> {
    static inline void even_split(const Func& func, const T1& t1,
        const T2& t2, const T3& t3, const T4& t4, Ts... args)
    {
        func(t1, t2, t3, t4);
        if constexpr (sizeof...(Ts) >= 4)
            SplitHelper<4, Func, Ts...>::even_split(func, args...);
    }
};

// Traits
template <typename T>
struct is_float {
    static constexpr bool value = false;
};

template <>
struct is_float<float> {
    static constexpr bool value = true;
};

template <typename T>
inline constexpr bool is_float_v = is_float<T>::value;

template <typename T>
struct is_double {
    static constexpr bool value = false;
};

template <>
struct is_double<double> {
    static constexpr bool value = true;
};

template <typename T>
constexpr bool is_double_v = is_double<T>::value;

template <typename T>
using is_int32 = std::bool_constant<std::is_integral_v<T> && sizeof(T) == 4>;

template <typename T>
constexpr bool is_int32_v = is_int32<T>::value;

template <typename T>
using is_int64 = std::bool_constant<std::is_integral_v<T> && sizeof(T) == 8>;

template <typename T>
constexpr bool is_int64_v = is_int64<T>::value;

template <typename T>
using enable_if_int32_t = std::enable_if_t<is_int32_v<T>>;

template <typename T>
using enable_if_int64_t = std::enable_if_t<is_int64_v<T>>;

// Types
struct empty_t {};

// Variables
template <typename T>
constexpr T epsilon = static_cast<T>(1e-6f);

}
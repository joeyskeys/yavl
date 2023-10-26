#pragma once

#include <concepts>
#include <stdint.h>
#include <type_traits>

namespace yavl
{

// Macros
#define YAVL_TYPE_ALIAS(TYPE, N, INTRIN_N)                              \
    using Scalar = std::decay_t<TYPE>;                                  \
    static constexpr uint32_t Size = N;                                 \
    static constexpr uint32_t IntrinSize = INTRIN_N;                    \
    static constexpr uint32_t BitSize = sizeof(Scalar) * N * 8;

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

}
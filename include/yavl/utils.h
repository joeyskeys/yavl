#pragma once

#include <concepts>
#include <stdint.h>
#include <tuple>
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

#define YAVL_DEFINE_VEC_OP(BITS, OP, AT, NAME, IT)                      \
    auto operator OP(const AT& v) const {                               \
        OP_VEC_EXPRS(BITS, OP, AT, NAME, IT)                            \
    }                                                                   \
    auto operator OP##=(const AT& v) {                                  \
        OP_VEC_ASSIGN_EXPRS(BITS, OP, AT, NAME, IT)                     \
    }

#define YAVL_DEFINE_SCALAR_OP(BITS, OP, NAME, IT)                       \
    auto operator OP(const Scalar v) const {                            \
        OP_SCALAR_EXPRS(BITS, OP, NAME, IT)                             \
    }                                                                   \
    auto operator OP##=(const Scalar v) {                               \
        OP_SCALAR_ASSIGN_EXPRS(BITS, OP, NAME, IT)                      \
    }

#define YAVL_DEFINE_FRIEND_OP(BITS, OP, AT, NAME, IT)                   \
    friend auto operator OP(const Scalar s, const AT& v) {              \
        OP_FRIEND_SCALAR_EXPRS(BITS, OP, AT, NAME, IT)                  \
    }

#define YAVL_DEFINE_OP(BITS, OP, AT, NAME, IT)                          \
    YAVL_DEFINE_VEC_OP(BITS, OP, AT, NAME, IT)                          \
    YAVL_DEFINE_SCALAR_OP(BITS, OP, NAME, IT)                           \
    YAVL_DEFINE_FRIEND_OP(BITS, OP, AT, NAME, IT)

#define YAVL_DEFINE_BASIC_FP_ARITHMIC_OP(BITS, AT, IT)                  \
    YAVL_DEFINE_OP(BITS, +, AT, add, IT)                                \
    YAVL_DEFINE_OP(BITS, -, AT, sub, IT)                                \
    YAVL_DEFINE_OP(BITS, *, AT, mul, IT)                                \
    YAVL_DEFINE_OP(BITS, /, AT, div, IT)

#define YAVL_DEFINE_BASIC_INT_ARITHMIC_OP(BITS, AT, IT)                 \
    YAVL_DEFINE_OP(BITS, +, AT, add, IT)                                \
    YAVL_DEFINE_OP(BITS, -, AT, sub, IT)                                \
    YAVL_DEFINE_OP(BITS, *, AT, mul, IT)

#define YAVL_DEFINE_VEC_INDEX_OP                                        \
    Scalar& operator [](const uint32_t i) {                             \
        assert(i < Size);                                               \
        return arr[i];                                                  \
    }                                                                   \
    const Scalar& operator [](const uint32_t i) const {                 \
        assert(i < Size);                                               \
        return arr[i];                                                  \
    }

#define YAVL_DEFINE_COPY_ASSIGN_OP(BITS, AT, IT)                        \
    Vec& operator =(const AT& b) {                                      \
        COPY_ASSIGN_EXPRS(BITS, IT)                                     \
    }

#define YAVL_DEFINE_VEC_BASIC_MISC_OP(BITS, IT)                         \
    YAVL_DEFINE_VEC_INDEX_OP                                            \
    YAVL_DEFINE_COPY_ASSIGN_OP(BITS, Vec, IT)

#define YAVL_DEFINE_VEC_BASIC_FP_OP(BITS, IT, CMD_SUFFIX)               \
    YAVL_DEFINE_VEC_BASIC_MISC_OP(BITS, CMD_SUFFIX)                     \
    YAVL_DEFINE_BASIC_FP_ARITHMIC_OP(BITS, Vec, IT)

#define YAVL_DEFINE_VEC_BASIC_INT_OP(BITS, IT, CMD_SUFFIX)              \
    YAVL_DEFINE_VEC_BASIC_MISC_OP(BITS, CMD_SUFFIX)                     \
    YAVL_DEFINE_BASIC_INT_ARITHMIC_OP(BITS, Vec, IT)

// Templates
template <int... indices, typename Func>
inline void static_for(const Func& func, std::integer_sequence<int, indices...> sequence) {
    (func(indices), ...);
}

template <int N, typename Func>
inline void static_for(const Func& func) {
    static_for(func, std::make_integer_sequence<int, N>{});
}

template <int N, typename... Ts>
inline auto head(Ts... args) {
    return std::tuple();
}

template <int N, typename T, typename... Ts>
inline auto head(T t, Ts... args) {
    if constexpr (N > 0)
        return std::tuple_cat(std::make_tuple(t), head<N - 1>(args...));
    else
        return std::tuple();
}

template <int N, typename T, typename... Ts>
inline auto skip(T t, Ts... args) {
    if constexpr (N > 0)
        return skip<N - 1>(args...);
    else
        return std::tuple_cat(std::make_tuple(t), std::make_tuple(args...));
}

template <int N, typename... Ts>
inline auto tail(Ts... args) {
    static_assert(N <= sizeof...(Ts));
    return skip<sizeof...(Ts) - N>(args...);
}

/*
struct dispatch {
    template <int N>
    void operator()(const auto& func, auto... args) {
        static_assert(sizeof...(args) > 0 && sizeof...(args) % N == 0);
        std::apply(func, head<N>(args...));
        if constexpr (sizeof...(args) > 0 && sizeof...(args) >= N + N) {
            std::apply(dispatch, std::tuple_cat(
                std::make_tuple(func), tail<sizeof...(args) - N>(args...)));
        }
    }
};
*/

template <typename F, typename T0, typename T1, typename T2, typename T3, 
    typename... Ts>
inline void apply_by4(const uint32_t i, F&& f, T0&& t0, T1&& t1, T2&& t2,
    T3&& t3, Ts&&... args)
{
    f(i, std::forward(t0), std::forward(t1), std::forward(t2), std::forward(t3));
    apply_by4(i + 1, std::forward(f), args...);
}

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
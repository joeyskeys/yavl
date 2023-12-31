#pragma once

#include <array>
#include <cassert>
#include <concepts>
#include <numeric>
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

#define YAVL_DEFINE_VEC_OP(BITS, OP, AT, NAME, IT)                      \
    auto operator OP(const AT<Scalar, Size>& v) const {                 \
        OP_VEC_EXPRS(BITS, OP, AT, NAME, IT)                            \
    }                                                                   \
    auto operator OP##=(const AT<Scalar, Size>& v) {                    \
        OP_VEC_ASSIGN_EXPRS(BITS, OP, AT, NAME, IT)                     \
    }

#define YAVL_DEFINE_SCALAR_OP(BITS, OP, AT, NAME, IT)                   \
    auto operator OP(const Scalar v) const {                            \
        OP_SCALAR_EXPRS(BITS, OP, AT, NAME, IT)                         \
    }                                                                   \
    auto operator OP##=(const Scalar v) {                               \
        OP_SCALAR_ASSIGN_EXPRS(BITS, OP, AT, NAME, IT)                  \
    }

#define YAVL_DEFINE_FRIEND_OP(BITS, OP, AT, NAME, IT)                   \
    friend auto operator OP(const Scalar s, const AT<Scalar, Size>& v) { \
        OP_FRIEND_SCALAR_EXPRS(BITS, OP, AT, NAME, IT)                  \
    }

#define YAVL_DEFINE_OP(BITS, OP, AT, NAME, IT)                          \
    YAVL_DEFINE_VEC_OP(BITS, OP, AT, NAME, IT)                          \
    YAVL_DEFINE_SCALAR_OP(BITS, OP, AT, NAME, IT)                       \

#define YAVL_DEFINE_OP_WITH_FRIEND(BITS, OP, AT, NAME, IT)              \
    YAVL_DEFINE_OP(BITS, OP, AT, NAME, IT)                              \
    YAVL_DEFINE_FRIEND_OP(BITS, OP, AT, NAME, IT)

#define YAVL_DEFINE_FP_ARITHMIC_OP(BITS, AT, IT)                        \
    YAVL_DEFINE_OP(BITS, +, AT, add, IT)                                \
    YAVL_DEFINE_OP(BITS, -, AT, sub, IT)                                \
    YAVL_DEFINE_OP(BITS, *, AT, mul, IT)                                \
    YAVL_DEFINE_OP(BITS, /, AT, div, IT)

#define YAVL_DEFINE_FP_ARITHMIC_OP_WITH_FRIEND(BITS, AT, IT)            \
    YAVL_DEFINE_OP_WITH_FRIEND(BITS, +, AT, add, IT)                    \
    YAVL_DEFINE_OP_WITH_FRIEND(BITS, -, AT, sub, IT)                    \
    YAVL_DEFINE_OP_WITH_FRIEND(BITS, *, AT, mul, IT)                    \
    YAVL_DEFINE_OP_WITH_FRIEND(BITS, /, AT, div, IT)

#define YAVL_DEFINE_INT_ARITHMIC_OP(BITS, AT, IT)                       \
    YAVL_DEFINE_OP(BITS, +, AT, add, IT)                                \
    YAVL_DEFINE_OP(BITS, -, AT, sub, IT)                                \
    YAVL_DEFINE_OP(BITS, *, AT, mullo, IT)

#define YAVL_DEFINE_INT_ARITHMIC_OP_WITH_FRIEND(BITS, AT, IT)           \
    YAVL_DEFINE_OP_WITH_FRIEND(BITS, +, AT, add, IT)                    \
    YAVL_DEFINE_OP_WITH_FRIEND(BITS, -, AT, sub, IT)                    \
    YAVL_DEFINE_OP_WITH_FRIEND(BITS, *, AT, mullo, IT)

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
    auto& operator =(const AT<Scalar, Size>& b) {                       \
        COPY_ASSIGN_EXPRS(BITS, IT)                                     \
    }

#define YAVL_DEFINE_VEC_MISC_OP(VT, BITS, IT)                           \
    YAVL_DEFINE_VEC_INDEX_OP                                            \
    YAVL_DEFINE_COPY_ASSIGN_OP(BITS, VT, IT)

#define YAVL_DEFINE_VEC_FP_OP(VT, BITS, IT, CMD_SUFFIX)                 \
    YAVL_DEFINE_VEC_MISC_OP(VT, BITS, CMD_SUFFIX)                       \
    YAVL_DEFINE_FP_ARITHMIC_OP_WITH_FRIEND(BITS, VT, IT)

#define YAVL_DEFINE_VEC_INT_OP(VT, BITS, IT, CMD_SUFFIX)                \
    YAVL_DEFINE_VEC_MISC_OP(VT, BITS, CMD_SUFFIX)                       \
    YAVL_DEFINE_INT_ARITHMIC_OP_WITH_FRIEND(BITS, VT, IT)

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

template <typename F, typename... Ts>
inline void apply_by3(const uint32_t i, const F& f, Ts&&... args) {
    return;
}

template <typename F, typename T0, typename T1, typename T2, typename... Ts>
inline void apply_by3(const uint32_t i, const F& f, T0&& t0, T1&& t1, T2&& t2,
    Ts&&... args)
{
    f(i, std::forward<T0>(t0), std::forward<T1>(t1), std::forward<T2>(t2));
    apply_by3(i + 1, f, args...);
}

template <typename F, typename... Ts>
inline void apply_by4(const uint32_t i, const F& f, Ts&&... args) {
    return;
}

template <typename F, typename T0, typename T1, typename T2, typename T3, 
    typename... Ts>
inline void apply_by4(const uint32_t i, const F& f, T0&& t0, T1&& t1, T2&& t2,
    T3&& t3, Ts&&... args)
{
    f(i, std::forward<T0>(t0), std::forward<T1>(t1), std::forward<T2>(t2),
        std::forward<T3>(t3));
    apply_by4(i + 1, f, args...);
}

template <typename F, typename... Ts>
inline void apply_by8(const uint32_t i, const F& f, Ts&&... args) {
    return;
}

template <typename F, typename T0, typename T1, typename T2, typename T3,
    typename T4, typename T5, typename T6, typename T7, typename... Ts>
inline void apply_by8(const uint32_t i, const F& f, T0&& t0, T1&& t1, T2&& t2,
    T3&& t3, T4&& t4, T5&& t5, T6&& t6, T7&& t7, Ts&&... args)
{
    f(i, std::forward<T0>(t0), std::forward<T1>(t1), std::forward<T2>(t2),
        std::forward<T3>(t3), std::forward<T4>(t4), std::forward<T5>(t5),
        std::forward<T6>(t6), std::forward<T7>(t7));
    apply_by8(i + 1, f, args...);
}

// https://gist.github.com/jjsullivan5196/a83f99263cda755edc257f9c50b53470
template <typename T, T Start, T End, T Step = 1, T... Ints>
struct integer_range : std::conditional<
    std::integral_constant<bool, Start >= End>::value,
    std::integer_sequence<T, Ints..., End>,
    integer_range<T, Start + Step, End, Step, Ints..., Start>
>::type {};

template <typename T, T Start, T End, T Step = 1>
constexpr auto integer_range_v = integer_range<T, Start, End, Step>();

template <typename T, T Size, T... Seq>
constexpr auto array_from_integer_range(std::integer_sequence<T, Seq...>) {
    return std::array<T, Size> { Seq... };
}

template <typename T, T Start, T End, T Step = 1, T Size = (((End - Start) / Step) + 1)>
constexpr auto integer_range_array = array_from_integer_range<T, Size>(integer_range_v<T, Start, End, Step>);

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
using enable_if_int32_t = std::enable_if_t<is_int32_v<T>, int>;

template <typename T>
using enable_if_int64_t = std::enable_if_t<is_int64_v<T>, int>;

// Types
struct empty_t {};

// Variables
template <typename T>
constexpr T epsilon = static_cast<T>(1e-6f);

// Functions
void escape(void *p) {
    asm volatile("" : : "g"(p) : "memory");
}

void clobber() {
    asm volatile("" : : : "memory");
}

}
#pragma once

#include <algorithm>
#include <cmath>
#include <stdint.h>
#include <type_traits>

#include <yavl/intrin.h>
#include <yavl/traits.h>
#include <yavl/utils.h>

namespace yavl
{

#define YAVL_VEC_ALIAS(TYPE, N, INTRIN_N)                               \
    using Scalar = std::decay_t<TYPE>;                                  \
    static constexpr uint32_t Size = N;                                 \
    static constexpr uint32_t IntrinSize = INTRIN_N;                    \
    static constexpr uint32_t BitSize = sizeof(Scalar) * N * 8;

#define YAVL_VEC_OPTIONAL_ALIAS(TYPE, N)                                \
    using Z = std::conditional_t<(N > 2), TYPE, empty_t>;               \
    using W = std::conditional_t<(N > 3), TYPE, empty_t>;               \
    using B = std::conditional_t<(N > 2), TYPE, empty_t>;               \
    using A = std::conditional_t<(N > 3), TYPE, empty_t>;

#define YAVL_DEFINE_VEC_OP(OP, NAME, INTRIN_TYPE)                       \
    auto operator OP(const Vec& v) const {                              \
        OP_VEC_EXPRS(OP, NAME, INTRIN_TYPE)                             \
    }                                                                   \
    auto operator OP##=(const Vec& v) {                                 \
        OP_VEC_ASSIGN_EXPRS(OP, NAME, INTRIN_TYPE)                      \
    }

#define YAVL_DEFINE_SCALAR_OP(OP, NAME, INTRIN_TYPE)                    \
    auto operator OP(const Scalar v) const {                            \
        OP_SCALAR_EXPRS(OP, NAME, INTRIN_TYPE)                          \
    }                                                                   \
    auto operator OP##=(const Scalar v) {                               \
        OP_SCALAR_ASSIGN_EXPRS(OP, NAME, INTRIN_TYPE)                   \
    }

#define YAVL_DEFINE_FRIEND_OP(OP, NAME, INTRIN_TYPE)                    \
    friend auto operator OP(const Scalar s, const Vec& v) {             \
        OP_FRIEND_SCALAR_EXPRS(OP, NAME, INTRIN_TYPE)                   \
    }

#define YAVL_DEFINE_OP(OP, NAME, INTRIN_TYPE)                           \
    YAVL_DEFINE_VEC_OP(OP, NAME, INTRIN_TYPE)                           \
    YAVL_DEFINE_SCALAR_OP(OP, NAME, INTRIN_TYPE)                        \
    YAVL_DEFINE_FRIEND_OP(OP, NAME, INTRIN_TYPE)

#define YAVL_DEFINE_BASIC_ARITHMIC_OP(INTRIN_TYPE)                      \
    YAVL_DEFINE_OP(+, add, INTRIN_TYPE)                                 \
    YAVL_DEFINE_OP(-, sub, INTRIN_TYPE)                                 \
    YAVL_DEFINE_OP(*, mul, INTRIN_TYPE)                                 \
    YAVL_DEFINE_OP(/, div, INTRIN_TYPE)


#define YAVL_DEFINE_VEC_INDEX_OP                                        \
    Scalar& operator [](const uint32_t i) {                             \
        assert(i < Size);                                               \
        return arr[i];                                                  \
    }                                                                   \
    const Scalar& operator [](const uint32_t i) const {                 \
        assert(i < Size);                                               \
        return arr[i];                                                  \
    }

template <typename T, uint32_t N, bool enable_vec=true, typename = int>
struct Vec {
    YAVL_VEC_ALIAS(T, N, N)
    YAVL_VEC_OPTIONAL_ALIAS(Scalar, N)

    static constexpr bool vectorized = false;

    union {
        struct {
            Scalar x;
            Scalar y;

            [[no_unique_address]] Z z;
            [[no_unique_address]] W w;
        };
        struct {
            Scalar r;
            Scalar g;

            [[no_unique_address]] B b;
            [[no_unique_address]] A a;
        };

        std::array<Scalar, Size> arr;
    };

    // Ctors
    constexpr Vec() {
        arr.fill(static_cast<T>(0));
    }

    template <typename ...Ts>
    constexpr Vec(Ts... args) {
        static_assert(sizeof...(args) == 1 || sizeof...(args) == Size);
        if constexpr (sizeof...(args) == 1)
            arr.fill(static_cast<Scalar>(args)...);
        else
            arr = { static_cast<Scalar>(args)... };
    }

    Vec(const Vec&) = default;
    Vec(Vec&&) = default;

    // Operators
    YAVL_DEFINE_VEC_INDEX_OP

#define OP_VEC_EXPRS(OP, NAME, INTRIN_TYPE)                             \
    Vec tmp;                                                            \
    for (int i = 0; i < Size; ++i)                                      \
        tmp[i] = arr[i] OP v[i];                                        \
    return tmp;

#define OP_VEC_ASSIGN_EXPRS(OP, NAME, INTRIN_TYPE)                      \
    for (int i = 0; i < Size; ++i)                                      \
        arr[i] OP##= v[i];                                              \
    return *this;

#define OP_SCALAR_EXPRS(OP, NAME, INTRIN_TYPE)                          \
    Vec tmp;                                                            \
    for (int i = 0; i < Size; ++i)                                      \
        tmp[i] = arr[i] OP v;                                           \
    return tmp;

#define OP_SCALAR_ASSIGN_EXPRS(OP, NAME, INTRIN_TYPE)                   \
    for (int i = 0; i < Size; ++i)                                      \
        arr[i] OP##= v;                                                 \
    return *this;

#define OP_FRIEND_SCALAR_EXPRS(OP, NAME, INTRIN_TYPE)                   \
    Vec tmp;                                                            \
    for (int i = 0; i < Size; ++i)                                      \
        tmp[i] = s OP v[i];                                             \
    return tmp;

    YAVL_DEFINE_BASIC_ARITHMIC_OP( )

#undef OP_VEC_EXPRS
#undef OP_VEC_ASSIGN_EXPRS
#undef OP_SCALAR_EXPRS
#undef OP_SCALAR_ASSIGN_EXPRS
#undef OP_FRIEND_SCALAR_EXPRS

    // Misc
#define MISC_SHUFFLE_FUNC                                               \
    template <typename ...Ts>                                           \
        requires (std::default_initializable<Ts> && ...) &&             \
            (std::convertible_to<Ts, Scalar> && ...)                    \
    inline Vec shuffle(Ts... args) const {                    \
        MISC_SHUFFLE_EXPRS                                              \
    }

#define MISC_BASE_SHUFFLE_EXPRS                                              \
    {                                                                   \
        static_assert(sizeof...(args) == Size);                         \
        Vec tmp;                                                        \
        ([&] <std::size_t... Is>(std::index_sequence<Is...>, auto&&... as) { \
            auto impl = [&] <typename A>(std::size_t i, A&& a) {        \
                tmp[i] = arr[static_cast<uint32_t>(a)];                 \
            };                                                          \
            (impl(Is, std::forward<decltype(as)>(as)), ...);            \
        }                                                               \
        (std::index_sequence_for<Ts...>{}, std::forward<Ts>(args)...)); \
        return tmp;                                                     \
    }

#define MISC_SHUFFLE_EXPRS MISC_BASE_SHUFFLE_EXPRS

#define YAVL_DEFINE_MISC_FUNCS                                          \
    MISC_SHUFFLE_FUNC

    YAVL_DEFINE_MISC_FUNCS

#undef MISC_SHUFFLE_EXPRS

    // Geometry
#define GEO_DOT_FUNC                                                    \
    inline Scalar dot(const Vec& b) const {                             \
        GEO_DOT_EXPRS                                                   \
    }

#define GEO_DOT_EXPRS                                                   \
    {                                                                   \
        return this->operator*(b).sum();                                \
    }

#define YAVL_DEFINE_GEO_FUNCS                                           \
    GEO_DOT_FUNC

    YAVL_DEFINE_GEO_FUNCS

#undef GEO_DOT_EXPRS

    // Math
#define MATH_LENGTH_SQUARED_FUNC                                        \
    inline auto length_squared() const {                                \
        return dot(*this);                                              \
    }

#define MATH_LENGTH_FUNC                                                \
    inline auto length() const {                                        \
        return std::sqrt(length_squared());                             \
    }

#define MATH_NORMALIZE_FUNC                                             \
    inline Vec& normalize() {                                           \
        Scalar rcp = 1. / length();                                     \
        *this *= rcp;                                                   \
        return *this;                                                   \
    }

#define MATH_NORMALIZED_FUNC                                            \
    inline auto normalized() const {                                    \
        Scalar rcp = 1. / length();                                     \
        return *this * rcp;                                             \
    }

#define MATH_ABS_FUNC                                                   \
    inline auto abs() const {                                           \
        MATH_ABS_EXPRS                                                  \
    }

#define MATH_ABS_EXPRS                                                  \
    {                                                                   \
        Vec tmp;                                                        \
        for (int i = 0; i < Size; ++i)                                  \
            tmp[i] = std::abs(arr[i]);                                  \
        return tmp;                                                     \
    }

#define MATH_SUM_FUNC                                                   \
    inline Scalar sum() const {                                         \
        MATH_SUM_EXPRS                                                  \
    }

#define MATH_SUM_EXPRS                                                  \
    {                                                                   \
        return std::accumulate(arr.begin(), arr.end(), Scalar{0});      \
    }

#define MATH_SQUARE_FUNC                                                \
    inline auto square() const {                                        \
        return *this * *this;                                           \
    }

#define MATH_RCP_FUNC                                                   \
    inline auto rcp() const {                                           \
        MATH_RCP_EXPRS                                                  \
    }
#define MATH_RCP_EXPRS                                                  \
    {                                                                   \
        return Vec(1 / *this);                                          \
    }

#define MATH_SQRT_FUNC                                                  \
    inline auto sqrt() const {                                          \
        MATH_SQRT_EXPRS                                                 \
    }

#define MATH_SQRT_EXPRS                                                 \
    {                                                                   \
        Vec tmp;                                                        \
        for (int i = 0; i < Size; ++i)                                  \
            tmp[i] = std::sqrt(arr[i]);                                 \
        return tmp;                                                     \
    }

#define MATH_RSQRT_FUNC                                                 \
    inline auto rsqrt() const {                                         \
        MATH_RSQRT_EXPRS                                                \
    }

#define MATH_RSQRT_EXPRS                                                \
    {                                                                   \
        return 1. / length();                                           \
    }

/*
// Put these functions into a seperated file
#define MATH_EXP_FUNC                                                   \
    inline auto exp() const {                                           \
        MATH_EXP_EXPRS                                                  \
    }

#define MATH_EXP_EXPRS                                                  \
    {                                                                   \
        Vec tmp;                                                        \
        for (int i = 0; i < Size; ++i)                                  \
            tmp[i] = std::exp(arr[i]);                                  \
        return tmp;                                                     \
    }

#define MATH_POW_FUNC                                                   \
    inline auto pow(const Scalar beta) const {                          \
        MATH_POW_EXPRS                                                  \
    }

#define MATH_POW_EXPRS                                                  \
    {                                                                   \
        Vec tmp;                                                        \
        for (int i = 0; i < N; ++i)                                     \
            tmp[i] = std::pow(arr[i]);                                  \
        return tmp;                                                     \
    }
*/

#define MATH_LERP_FUNC                                                  \
    inline auto lerp(const Vec& b, const Scalar t) const {              \
        MATH_LERP_SCALAR_EXPRS                                          \
    }                                                                   \
    inline auto lerp(const Vec& b, const Vec& t) const {                \
        MATH_LERP_VEC_EXPRS                                             \
    }

#define MATH_LERP_SCALAR_EXPRS                                          \
    {                                                                   \
        return this->operator*(1 - t) + b * t;                          \
    }

#define MATH_LERP_VEC_EXPRS MATH_LERP_SCALAR_EXPRS

#define YAVL_DEFINE_MATH_FUNCS                                          \
    MATH_LENGTH_SQUARED_FUNC                                            \
    MATH_LENGTH_FUNC                                                    \
    MATH_NORMALIZE_FUNC                                                 \
    MATH_ABS_FUNC                                                       \
    MATH_SUM_FUNC                                                       \
    MATH_SQUARE_FUNC                                                    \
    MATH_RCP_FUNC                                                       \
    MATH_SQRT_FUNC                                                      \
    MATH_RSQRT_FUNC                                                     \
    MATH_LERP_FUNC

    YAVL_DEFINE_MATH_FUNCS

#undef MATH_LENGTH_SQUARED_EXPRS
#undef MATH_LENGTH_EXPRS
#undef MATH_NORMALIZE_EXPRS
#undef MATH_NORMALIZED_EXPRS
#undef MATH_ABS_EXPRS
#undef MATH_SUM_EXPRS
#undef MATH_SQUARE_EXPRS
#undef MATH_RCP_EXPRS
#undef MATH_SQRT_EXPRS
#undef MATH_RSQRT_EXPRS
#undef MATH_LERP_SCALAR_EXPRS
#undef MATH_LERP_VEC_EXPRS
};

}

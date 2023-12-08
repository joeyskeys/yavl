#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>

namespace yavl
{

#define YAVL_VEC_OPTIONAL_ALIAS(TYPE, N)                                \
    using Z = std::conditional_t<(N > 2), TYPE, empty_t>;               \
    using W = std::conditional_t<(N > 3), TYPE, empty_t>;               \
    using B = std::conditional_t<(N > 2), TYPE, empty_t>;               \
    using A = std::conditional_t<(N > 3), TYPE, empty_t>;

template <typename Vec, typename ...Ts>
inline Vec base_shuffle_impl(const Vec& v, Ts ...args) {
    static_assert(sizeof...(args) == Vec::Size);
    Vec tmp;
    ([&] <std::size_t... Is>(std::index_sequence<Is...>, auto&&... as) {
        auto impl = [&] <typename A>(std::size_t i, A&& a) {
            tmp[i] = v.arr[static_cast<uint32_t>(a)];
        };
        (impl(Is, std::forward<decltype(as)>(as)), ...);
    }
    (std::index_sequence_for<Ts...>{}, std::forward<Ts>(args)...));
    return tmp;
}

template <typename T, uint32_t N, typename = int>
struct Vec {
    YAVL_TYPE_ALIAS(T, N, N)
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
        arr.fill(static_cast<Scalar>(0));
    }

    template <typename ...Ts>
    constexpr Vec(Ts... args) {
        static_assert(sizeof...(args) == 1 || sizeof...(args) == Size);
        if constexpr (sizeof...(args) == 1)
            arr.fill( static_cast<Scalar>(args)... );
        else
            arr = { static_cast<Scalar>(args)... };
    }

    Vec(const Vec&) = default;
    Vec(Vec&&) = default;

    // Operators
#define COPY_ASSIGN_EXPRS(BITS, IT)                                     \
    {                                                                   \
        std::memcpy(arr.data(), b.arr.data(), sizeof(Scalar) * Size);   \
        return *this;                                                   \
    }

#define OP_VEC_EXPRS(BITS, OP, AT, NAME, IT)                            \
    {                                                                   \
        AT tmp;                                                         \
        static_for<Size>([&](const auto i) {                            \
            tmp[i] = arr[i] OP v[i];                                    \
        });                                                             \
        return tmp;                                                     \
    }

#define OP_VEC_ASSIGN_EXPRS(BITS, OP, AT, NAME, IT)                     \
    {                                                                   \
        static_for<Size>([&](const auto i) {                            \
            arr[i] OP##= v[i];                                          \
        });                                                             \
        return *this;                                                   \
    }

#define OP_SCALAR_EXPRS(BITS, OP, NAME, IT)                             \
    {                                                                   \
        Vec tmp;                                                        \
        static_for<Size>([&](const auto i) {                            \
            tmp[i] = arr[i] OP v;                                       \
        });                                                             \
        return tmp;                                                     \
    }

#define OP_SCALAR_ASSIGN_EXPRS(BITS, OP, NAME, IT)                      \
    {                                                                   \
        static_for<Size>([&](const auto i) {                            \
            arr[i] OP##= v;                                             \
        });                                                             \
        return *this;                                                   \
    }

#define OP_FRIEND_SCALAR_EXPRS(BITS, OP, AT, NAME, IT)                  \
    {                                                                   \
        AT tmp;                                                         \
        static_for<Size>([&](const auto i) {                            \
            tmp[i] = s OP v[i];                                         \
        });                                                             \
        return tmp;                                                     \
    }

    YAVL_DEFINE_VEC_FP_OP( , , )

#undef COPY_ASSIGN_EXPRS
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
    inline Vec shuffle(Ts... args) const {                              \
        return base_shuffle_impl(*this, args...);                       \
    }

#define YAVL_DEFINE_MISC_FUNCS                                          \
    inline auto* data() {                                               \
        return arr.data();                                              \
    }                                                                   \
    inline auto* data() const {                                         \
        return arr.data();                                              \
    }                                                                   \
    MISC_SHUFFLE_FUNC

    template <int ...Is>
    inline Vec shuffle() const {
        static_assert(sizeof...(Is) == Size);                           \
        std::array<int, sizeof...(Is)> indice{ Is... };                 \
        Vec tmp;                                                        \
        static_for<Size>([&](const auto i) {                            \
            tmp.arr[i] = arr[indice[i]];                                \
        });                                                             \
        return tmp;                                                     \
    }

    YAVL_DEFINE_MISC_FUNCS

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

    inline auto cross(const Vec& b) const {
        static_assert(Vec::Size > 1 && Vec::Size < 4);
        if constexpr (Vec::Size == 2) {
            return x * b.y - y * b.x;
        }
        else {
            return Vec(y * b.z - z * b.y, z * b.x - x * b.z,
                x * b.y - y * b.x);
        }
    }

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

#define MATH_ABS_FUNC(BITS, IT1, IT2)                                   \
    inline auto abs() const {                                           \
        MATH_ABS_EXPRS(BITS, IT1, IT2)                                  \
    }

#define MATH_ABS_EXPRS(BITS, IT1, IT2)                                  \
    {                                                                   \
        Vec tmp;                                                        \
        static_for<Size>([&](const auto i) {                            \
            tmp[i] = std::abs(arr[i]);                                  \
        });                                                             \
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
        static_for<Size>([&](const auto i) {                            \
            tmp[i] = std::sqrt(arr[i]);                                 \
        });                                                             \
        return tmp;                                                     \
    }

#define MATH_RSQRT_FUNC                                                 \
    inline auto rsqrt() const {                                         \
        MATH_RSQRT_EXPRS                                                \
    }

#define MATH_RSQRT_EXPRS                                                \
    {                                                                   \
        return 1. / sqrt();                                             \
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

#define MATH_LERP_FUNC(BITS, IT)                                        \
    inline auto lerp(const Vec& b, const Scalar t) const {              \
        MATH_LERP_SCALAR_EXPRS(BITS, IT)                                \
    }                                                                   \
    inline auto lerp(const Vec& b, const Vec& t) const {                \
        MATH_LERP_VEC_EXPRS(BITS, IT)                                   \
    }

#define MATH_LERP_SCALAR_EXPRS(BITS, IT)                                \
    {                                                                   \
        return this->operator*(1 - t) + b * t;                          \
    }

#define MATH_LERP_VEC_EXPRS(BITS, IT) MATH_LERP_SCALAR_EXPRS(BITS, IT)

#define YAVL_DEFINE_MATH_COMMON_FUNCS(BITS, IT1, IT2)                   \
    MATH_LENGTH_SQUARED_FUNC                                            \
    MATH_LENGTH_FUNC                                                    \
    MATH_ABS_FUNC(BITS, IT1, IT2)                                       \
    MATH_SUM_FUNC                                                       \
    MATH_SQUARE_FUNC

#define YAVL_DEFINE_MATH_FP_FUNCS(BITS, IT)                             \
    MATH_NORMALIZE_FUNC                                                 \
    MATH_RCP_FUNC                                                       \
    MATH_SQRT_FUNC                                                      \
    MATH_RSQRT_FUNC                                                     \
    MATH_LERP_FUNC(BITS, IT)

#define YAVL_DEFINE_MATH_FUNCS(BITS, IT1, IT2)                          \
    YAVL_DEFINE_MATH_COMMON_FUNCS(BITS, IT1, IT2)                       \
    YAVL_DEFINE_MATH_FP_FUNCS(BITS, IT1)

    YAVL_DEFINE_MATH_FUNCS(, ,)

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

// Vec type aliasing
template <typename T>
using Vec2 = Vec<T, 2>;

template <typename T>
using Vec3 = Vec<T, 3>;

template <typename T>
using Vec4 = Vec<T, 4>;

using Vec2f = Vec2<float>;
using Vec2d = Vec2<double>;
using Vec2i = Vec2<int>;
using Vec2u = Vec2<uint32_t>;

using Vec3f = Vec3<float>;
using Vec3d = Vec3<double>;
using Vec3i = Vec3<int>;
using Vec3u = Vec3<uint32_t>;

using Vec4f = Vec4<float>;
using Vec4d = Vec4<double>;
using Vec4i = Vec4<int>;
using Vec4u = Vec4<uint32_t>;

/*
using _Vec2f = _Vec2<float>;
using _Vec2d = _Vec2<double>;
using _Vec2i = _Vec2<int>;
using _Vec2u = _Vec2<uint32_t>;

using _Vec3f = _Vec3<float>;
using _Vec3d = _Vec3<double>;
using _Vec3i = _Vec3<int>;
using _Vec3u = _Vec3<uint32_t>;

using _Vec4f = _Vec4<float>;
using _Vec4d = _Vec4<double>;
using _Vec4i = _Vec4<int>;
using _Vec4u = _Vec4<uint32_t>;
*/

}

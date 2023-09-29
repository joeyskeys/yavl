#pragma once

#include <stdint.h>

#include <yavl/intrin.h>
#include <yavl/traits.h>
#include <yavl/utils.h>

namespace yavl
{

#define YAVL_VEC_ALIAS(TYPE, N)                                         \
    using Scalar = std::decay_t<TYPE>;                                  \
    static constexpr uint32_t Size = N;                                 \
    static constexpr uint32_t BitSize = sizeof(Scalar) * N * 8;

#define YAVL_VEC_OPTIONAL_ALIAS(TYPE, N)                                \
    using Z = std::conditional_t<(N > 2), TYPE, empty_t>;               \
    using W = std::conditional_t<(N > 3), TYPE, empty_t>;               \
    using B = std::conditional_t<(N > 2), TYPE, empty_t>;               \
    using A = std::conditional_t<(N > 3), TYPE, empty_t>;

#define YAVL_DEFINE_VEC_OP(OP, NAME)                                    \
    auto operator OP(const Vec& v) const {                              \
        VEC_EXPRS(OP, NAME)                                             \
    }                                                                   \
    auto operator OP##=(const Vec& v) {                                 \
        VEC_ASSIGN_EXPRS(OP, NAME)                                      \
    }

#define YAVL_DEFINE_SCALAR_OP(OP, NAME)                                 \
    auto operator OP(const Scalar v) const {                            \
        SCALAR_EXPRS(OP, NAME)                                          \
    }                                                                   \
    auto operator OP##=(const Scalar v) {                               \
        SCALAR_ASSIGN_EXPRS(OP, NAME)                                   \
    }

#define YAVL_DEFINE_OP(OP, NAME)                                        \
    YAVL_DEFINE_VEC_OP(OP, NAME)                                        \
    YAVL_DEFINE_SCALAR_OP(OP, NAME)

#define YAVL_DEFINE_FRIEND_OP(OP, NAME)                                 \
    friend auto operator OP(const Scalar s, const Vec& v) {             \
        FRIEND_SCALAR_EXPRS(OP, NAME)                                   \
    }

template <typename T, uint32_t N>
struct Vec {
    YAVL_VEC_ALIAS(T, N)
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
    Vec() {
        arr.fill(static_cast<T>(0));
    }

    template <typename ...Ts>
    Vec(Ts... args) {
        static_assert(sizeof...(args) == 1 || sizeof...(args) == Size);
        if constexpr (sizeof...(args) == 1)
            arr.fill(static_cast<Scalar>(args)...);
        else
            arr = { static_cast<Scalar>(args)... };
    }

    Vec(const Vec&) = default;
    Vec(Vec&&) = default;

    // Operators
#define VEC_EXPRS(OP, NAME)                                             \
    Vec tmp;                                                            \
    for (int i = 0; i < Size; ++i)                                      \
        tmp.arr[i] = arr[i] OP v.arr[i];                                \
    return tmp;

#define VEC_ASSIGN_EXPRS(OP, NAME)                                      \
    for (int i = 0; i < Size; ++i)                                      \
        arr[i] OP##= v.arr[i];                                          \
    return *this;

#define SCALAR_EXPRS(OP, NAME)                                          \
    Vec tmp;                                                            \
    for (int i = 0; i < Size; ++i)                                      \
        tmp.arr[i] = arr[i] OP v;                                       \
    return tmp;

#define SCALAR_ASSIGN_EXPRS(OP, NAME)                                   \
    for (int i = 0; i < Size; ++i)                                      \
        arr[i] OP##= v;                                                 \
    return *this;

#define FRIEND_SCALAR_EXPRS(OP, NAME)                                   \
    Vec tmp;                                                            \
    for (int i = 0; i < Size; ++i)                                      \
        tmp.arr[i] = s OP v.arr[i];                                     \
    return tmp;

    YAVL_DEFINE_OP(+,)
    YAVL_DEFINE_OP(-,)
    YAVL_DEFINE_OP(*,)
    YAVL_DEFINE_FRIEND_OP(*,)
    YAVL_DEFINE_OP(/,)
    YAVL_DEFINE_FRIEND_OP(/,)

#undef VEC_EXPRS
#undef VEC_ASSIGN_EXPRS
#undef SCALAR_EXPRS
#undef SCALAR_ASSIGN_EXPRS
#undef FRIEND_SCALAR_EXPRS
};

}
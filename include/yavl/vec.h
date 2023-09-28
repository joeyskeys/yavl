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
};

}
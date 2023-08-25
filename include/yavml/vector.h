#pragma once

#include <stdint.h>

#include <yavml/intrin.h>
#include <yavml/traits.h>
#include <yavml/utils.h>

namespace yavml
{

template <typename T, uint32_t N>
class Vec {
public:
    using Scalar = T;

    static constexpr uint32_t TypeSize = sizeof(T) * 8; // in bits
    static constexpr uint32_t CompSize = N;
    // 4 members to store a 3 component vector
    static constexpr uint32_t RealSize = next_even<N>();
    static constexpr uint32_t AlignasSize = TypeSize * RealSize / 8;

    using Z = std::conditional_t<(N > 2), T, empty_t>;
    using W = std::conditional_t<(N > 3), T, empty_t>;
    using VT = intrinsic_type_t<T, TypeSize * RealSize>;

    union {
        struct alignas(AlignasSize) {
            T x;
            T y;

            [[no_unique_address]] Z z;
            [[no_unique_address]] W w;
        };

        std::array<T, RealSize> arr;

        [[no_unique_address]] alignas(AlignasSize) VT v;
    };

    // Ctors

    // Operators
};

}
#pragma once

#include <stdint.h>

namespace yavl
{

template <uint32_t N>
constexpr uint32_t next_power_of_2() {
    // Calculates next power of 2 for numbers within 32 bits
    auto v = N;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return ++v;
}

template <uint32_t N>
constexpr uint32_t next_even() {
    auto v = N;
    return (v & 1) + v;
}

}
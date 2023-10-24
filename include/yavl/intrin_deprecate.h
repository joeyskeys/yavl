#pragma once

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wignored-attributes"
#elif defined(_MSC_VER)
// Do nothing here for now
#endif

#include <array>
#include <stdint.h>
#include <type_traits>

#include <yavl/platform.h>

#define INTRINSIC_FUNC(SIZE, OP, TYPE) _mm ## SIZE ## _ ## OP ## _ ## TYPE
#define INTRINSIC_TYPE(SIZE, TYPE) __m ## SIZE ## TYPE

struct empty_t {};

template <typename T, uint32_t Size>
struct intrinsic_type {
    using type = empty_t;
    static constexpr bool fallback = false;
    static constexpr uint32_t comp_size = 0;
};

template <typename T>
struct intrinsic_type<T, 64> {
    using type = std::conditional_t<has_sse42, INTRINSIC_TYPE(64,), empty_t>;
    static constexpr bool fallback = !has_sse42;
    static constexpr uint32_t comp_size = fallback ? 0 : 1;
};

template <>
struct intrinsic_type<float, 128> {
    using type = std::conditional_t<has_sse42, INTRINSIC_TYPE(128,), empty_t>;
    static constexpr bool fallback = !has_sse42;
    static constexpr uint32_t comp_size = fallback ? 0 : 1;
};

template <>
struct intrinsic_type<double, 128> {
    using type = std::conditional_t<has_sse42, INTRINSIC_TYPE(128, d), empty_t>;
    static constexpr bool fallback = !has_sse42;
    static constexpr uint32_t comp_size = fallback ? 0 : 1;
};

template <>
struct intrinsic_type<int, 128> {
    using type = std::conditional_t<has_sse42, INTRINSIC_TYPE(128, i), empty_t>;
    static constexpr bool fallback = !has_sse42;
    static constexpr uint32_t comp_size = fallback ? 0 : 1;
};

template <>
struct intrinsic_type<float, 256> {
    using type = std::conditional_t<has_avx, INTRINSIC_TYPE(256,),
        std::conditional_t<has_sse42, std::array<INTRINSIC_TYPE(128,), 2>, empty_t>>;
    static constexpr bool fallback = !has_avx;
    static constexpr uint32_t comp_size = has_avx ? 1 :
        has_sse42 ? 2 : 0;
};

template <>
struct intrinsic_type<double, 256> {
    using type = std::conditional_t<has_avx, INTRINSIC_TYPE(256, d),
        std::conditional_t<has_sse42, std::array<INTRINSIC_TYPE(128, d), 2>, empty_t>>;
    static constexpr bool fallback = !has_avx;
    static constexpr uint32_t comp_size = has_avx ? 1 :
        has_sse42 ? 2 : 0;
};

template <>
struct intrinsic_type<int, 256> {
    using type = std::conditional_t<has_avx, INTRINSIC_TYPE(256, i),
        std::conditional_t<has_sse42, std::array<INTRINSIC_TYPE(128, i), 2>, empty_t>>;
    static constexpr bool fallback = !has_avx;
    static constexpr uint32_t comp_size = has_avx ? 1 :
        has_sse42 ? 2 : 0;
};

template <>
struct intrinsic_type<float, 512> {
    using type = std::conditional_t<has_avx512f, INTRINSIC_TYPE(512,),
        std::conditional_t<has_avx, std::array<INTRINSIC_TYPE(256,), 2>,
        std::conditional_t<has_sse42, std::array<INTRINSIC_TYPE(128,), 4>,
        empty_t>>>;
    static constexpr bool fallback = !has_avx512f;
    static constexpr uint32_t fallback_size = has_avx512f ? 1 :
        has_avx ? 2 : has_sse42 ? 4 : 0;
};

template <>
struct intrinsic_type<double, 512> {
    using type = std::conditional_t<has_avx512f, INTRINSIC_TYPE(512, d),
        std::conditional_t<has_avx, std::array<INTRINSIC_TYPE(256, d), 2>,
        std::conditional_t<has_sse42, std::array<INTRINSIC_TYPE(128, d), 4>,
        empty_t>>>;
    static constexpr bool fallback = !has_avx512f;
    static constexpr uint32_t fallback_size = has_avx512f ? 1 :
        has_avx ? 2 : has_sse42 ? 4 : 0;
};

template <>
struct intrinsic_type<int, 512> {
    using type = std::conditional_t<has_avx512f, INTRINSIC_TYPE(512, i),
        std::conditional_t<has_avx, std::array<INTRINSIC_TYPE(256, i), 2>,
        std::conditional_t<has_sse42, std::array<INTRINSIC_TYPE(128, i), 4>,
        empty_t>>>;
    static constexpr bool fallback = !has_avx512f;
    static constexpr uint32_t fallback_size = has_avx512f ? 1 :
        has_avx ? 2 : has_sse42 ? 4 : 0;
};

template <typename T, uint32_t Size>
using intrinsic_type_t = typename intrinsic_type<T, Size>::type;

#pragma GCC diagnostic pop

/*
// An attempt to remove macros by using if constexpr.
// Much more typing...
template <typename T, uint32_t Bits>
auto intrin_set1(const T v) {
    static_assert(Bits == 32 || Bits == 64);

    if constexpr (std::is_floating_point_v<T>) {
        if constexpr (Bits == 32)
            return _mm_set1_ps(v);
        else
            return _mm_set1_pd(v);
    }
    else {
        if constexpr (Bits == 32)
            return _mm_set1_epi32(v);
        else
            return _mm_set1_epi64(v);
    }
}
*/
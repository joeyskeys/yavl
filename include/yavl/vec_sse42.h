#pragma once

#include <concepts>

namespace yavl
{

template <>
struct alignas(16) Vec<float, 4> {
    YAVL_VEC_ALIAS(float, 4)

    static constexpr bool vectorized = true;

    union {
        struct {
            Scalar x, y, z, w;
        };
        struct {
            Scalar r, g, b, a;
        };
        
        std::array<Scalar, Size> arr;
        __m128 m;
    };

    // Ctors
    Vec() : m(_mm_set1_ps(0)) {}

    template <typename V>
        requires std::default_initializable<V> && std::convertible_to<V, Scalar>
    Vec(V v) : m(_mm_set1_ps(static_cast<Scalar>(v))) {}

    template <typename ...Ts>
        requires (std::default_initializable<Ts> && ...) && (std::convertible_to<Ts, Scalar> && ...)
    Vec(Ts... args) {
            m = _mm_setr_ps(args...);
    }

    Vec(const __m128 val) : m(val) {}

    // Operators
    auto operator +(const Vec& v) const {
        return Vec(_mm_add_ps(m, v.m));
    }

    auto& operator +=(const Vec& v) {
        m = _mm_add_ps(m, v.m);
        return *this;
    }

    auto operator +(const Scalar v) const {
        auto vv = _mm_set1_ps(v);
        return Vec(_mm_add_ps(m, vv));
    }

    auto& operator +=(const Scalar v) {
        auto vv = _mm_set1_ps(v);
        m = _mm_add_ps(m, vv);
        return *this;
    }

    auto operator -(const Vec& v) const {
        return Vec(_mm_sub_ps(m, v.m));
    }

    auto operator -=(const Vec& v) {
        m = _mm_sub_ps(m, v.m);
        return *this;
    }

    auto operator -(const Scalar v) const {
        auto vv = _mm_set1_ps(v);
        return Vec(_mm_sub_ps(m, vv));
    }

    auto operator -=(const Scalar v) {
        auto vv = _mm_set1_ps(v);
        m = _mm_sub_ps(m, vv);
        return *this;
    }

    auto operator *(const Vec& v) const {
        return Vec(_mm_mul_ps(m, v.m));
    }

    auto operator *=(const Vec& v) {
        m = _mm_mul_ps(m, v.m);
        return *this;
    }

    auto operator *(const Scalar v) const {
        auto vv = _mm_set1_ps(v);
        return Vec(_mm_mul_ps(m, vv));
    }

    auto operator *=(const Scalar v) {
        auto vv = _mm_set1_ps(v);
        m = _mm_mul_ps(m, vv);
        return *this;
    }

    friend auto operator *(const Scalar s, const Vec& v) {
        return v * s;
    }

    auto operator /(const Vec& v) const {
        return Vec(_mm_div_ps(m, v.m));
    }

    auto operator /=(const Vec& v) {
        m = _mm_div_ps(m, v.m);
        return *this;
    }

    auto operator /(const Scalar v) const {
        auto vv = _mm_set1_ps(v);
        return Vec(_mm_div_ps(m, vv));
    }

    auto operator /=(const Scalar v) {
        auto vv = _mm_set1_ps(v);
        m = _mm_div_ps(m, vv);
        return *this;
    }

    friend auto operator /(const Scalar s, const Vec& v) {
        auto vv = _mm_set1_ps(s);
        return Vec(_mm_div_ps(vv, v.m));
    }
};

}
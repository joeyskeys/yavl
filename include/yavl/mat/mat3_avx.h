#pragma once

namespace yavl
{

template <>
struct alignas(32) Mat<float, 3> {
    YAVL_MAT3_ALIAS_VECTORIZED(float, 3)

    YAVL_DEFINE_MAT3_UNION(__m256, __m128)

    // Ctors
    YAVL_MAT3_VECTORIZED_CTOR(256, , ps)

    // Operators
    YAVL_DEFINE_MAT3_COMMON_OP(256, , ps, mul)

    auto operator *(const Vec<Scalar, Size>& v) const {
        auto vm1 = _mm256_setr_ps(v[0], v[0], v[0], v[0], v[1], v[1],
            v[1], v[1]);
        auto vm2 = _mm_set1_ps(v[2]);
        vm1 = _mm256_mul_ps(m1, vm1);
        vm2 = _mm_mul_ps(m2, vm2);

        Vec<Scalar, Size> tmp;
        static_for<2>([&](const auto i) constexpr {
            auto col = _mm256_extractf128_ps(vm1, i);
            tmp.m = _mm_add_ps(col, tmp.m);
        });
        tmp.m = _mm_add_ps(vm2, tmp.m);
        return tmp;
    }

    auto operator *(const Mat& mat) const {
        Mat tmp;
        __m128 col[3];
        col[0] = _mm256_extractf128_ps(mat.m1, 0);
        col[1] = _mm256_extractf128_ps(mat.m1, 1);
        col[2] = mat.m2;
        static_for<Size>([&](const auto i) {
            auto tm = _mm256_broadcast_f32x4(col[i]);
            tmp.m1 = _mm256_fmadd_ps(m1, tm, tmp.m1);
            tmp.m2 = _mm_fmadd_ps(m2, col[i], tmp.m2);
        });
        return tmp;

        /*
        // Or this impl, need to do some benchmarking
        Mat<Scalar, 4> m1, m2;
        memcpy(m1.data(), data(), 12 * sizeof(float));
        memcpy(m2.data(), mat.data(), 12 * sizeof(float));
        m1 *= m2;
        Mat tmp;
        memcpy(tmp.data(), m1.data(), 12 * sizeof(float));
        return tmp;
        */
    }

    // Misc funcs
    YAVL_DEFINE_DATA_METHOD

    // Matrix manipulation methods
    YAVL_DEFINE_MAT3_TRANSPOSE
};

} // namespace yavl
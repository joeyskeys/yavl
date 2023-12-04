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
        constexpr auto vm1 = _mm256_setr_ps(v[0], v[0], v[0], v[0], v[1], v[1],
            v[1], v[1]);
        constexpr auto vm2 = _mm256_set1_ps(v[2]);
        vm1 = _mm256_mul_ps(m1, vm1);
        vm2 = _mm_mul_ps(m2, vm2);

        auto vm1_flip = _mm256_permute2f128_ps(vm1);
        Vec<Scalar, Size> tmp;
        auto vm1_hsum = _mm256_hadd_ps(vm1, vm1_flip);
        tmp.m = _mm_add_ps(vm2, _mm256_extractf128_ps(vm1_hsum, 0));
        return tmp;
    }

    auto operator *(const Col<Scalar, Size>& v) const {
        auto vec = Vec<Scalar, Size>{v.m};
        return operator *(vec);
    }

    auto operator *(const Mat& mat) const {
        __m128 col[3];
        col[0] = _mm256_extractf128_ps(mat.m1, 0);
        col[1] = _mm256_extractf128_ps(mat.m1, 1);
        col[2] = mat.m2;
        static_for<Size>([&](const auto i) {
            col[i] = operator *(Vec<Scalar, Size>{col[i]});
        });
        Mat tmp;
        tmp.m1 = _mm256_insertf128_ps(tmp.m1, col[0], 0);
        tmp.m1 = _mm256_insertf128_ps(tmp.m1, col[1], 1);
        tmp.m2 = col[3];
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
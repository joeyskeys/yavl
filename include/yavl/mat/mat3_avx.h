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

    Vec<Scalar, Size> operator *(const Vec<Scalar, Size>& v) const {
        auto vm1 = _mm256_setr_ps(v[0], v[0], v[0], v[0], v[1], v[1],
            v[1], v[1]);
        auto vm2 = _mm_set1_ps(v[2]);
        vm1 = _mm256_mul_ps(m1, vm1);
        vm2 = _mm_mul_ps(m2, vm2);

        Vec<Scalar, Size> tmp;
        __m128 col[2];
        col[0] = _mm256_extractf128_ps(vm1, 0);
        col[1] = _mm256_extractf128_ps(vm1, 1);
        static_for<2>([&](const auto i) constexpr {
            // Even constexpr cannot make the i parameter an immediate
            //auto col = _mm256_extractf128_ps(vm1, i);
            tmp.m = _mm_add_ps(col[i], tmp.m);
        });
        tmp.m = _mm_add_ps(vm2, tmp.m);
        return tmp;
    }

    Mat operator *(const Mat& mat) const {
        Mat tmp;
        __m128 col[3];
        col[0] = _mm256_extractf128_ps(mat.m1, 0);
        col[1] = _mm256_extractf128_ps(mat.m1, 1);
        col[2] = mat.m2;
        static_for<Size>([&](const auto i) {
            #if defined(YAVL_X86_AVX512VL)
                auto tm = _mm256_broadcast_f32x4(col[i]);
            #else
                auto tm = _mm256_broadcast_ps(&col[i]);
            #endif
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
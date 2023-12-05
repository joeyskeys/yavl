#pragma once

namespace yavl
{

template <typename I>
struct alignas(32) Mat<I, 3, true, enable_if_int32_t<I>> {
    YAVL_MAT3_ALIAS_VECTORIZED(I, 3)

    YAVL_DEFINE_MAT3_UNION(__m256i, __m128i)

    // Ctors
    YAVL_MAT3_VECTORIZED_CTOR(256, , epi32)

    // Operators
    YAVL_DEFINE_MAT3_COMMON_OP(256, , epi32, mullo)

    auto operator *(const Vec<Scalar, Size>& v) const {
        auto vm1 = _mm256_setr_epi32(v[0], v[0], v[0], v[0], v[1], v[1],
            v[1], v[1]);
        auto vm2 = _mm256_set1_epi32(v[2]);
        vm1 = _mm256_mullo_epi32(m1, vm1);
        vm2 = _mm_mullo_epi32(m2, vm2);

        Vec<Scalar, Size> tmp;
        // Use constexpr lambda here to make param i available at compile
        // time since imm8 should be static.
        static_for<2>([&](const auto i) constexpr {
            auto col = _mm256_extracti128_si256(vm1, i);
            tmp.m = _mm_add_epi32(col, tmp.m);
        });
        tmp.m = _mm_add_epi32(vm2, tmp.m);
        return tmp;
    }

    auto operator *(const Mat& mat) const {
        Mat tmp;
        __m128i col[3];
        col[0] = _mm256_extracti128_si256(mat.m1, 0);
        col[1] = _mm256_extracti128_si256(mat.m1, 1);
        col[2] = mat.m2;
        static_for<Size>([&](const auto i) {
            #if defined(YAVL_X86_AVX512VL)
                auto tm = _mm256_broadcast_i32x4(col[i]);
            #else
                auto tm = _mm256_broadcast_ps(&col[i]);
            #endif
            tmp.m1 = _mm256_add_epi32(_mm256_mullo_epi32(m1, tm, tmp.m1));
            tmp.m2 = _mm_add_epi32(_mm256_mullo_epi32(m2, col[i], tmp.m2));
        });
        return tmp;
    }

    // Misc funcs
    YAVL_DEFINE_DATA_METHOD

    // Matrix manipulation methods
    YAVL_DEFINE_MAT3_TRANSPOSE
};

} // namespace yavl
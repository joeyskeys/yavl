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
        constexpr auto vm1 = _mm256_setr_epi32(v[0], v[0], v[0], v[0], v[1],
            v[1], v[1], v[1]);
        constexpr auto vm2 = _mm256_set1_epi32(v[2]);
        vm1 = _mm256_mullo_epi32(m1, vm1);
        vm2 = _mm_mullo_epi32(m2, vm2);

        auto vm1_flip = _mm256_permute2x128_si256(vm1);
        Vec<Scalar, Size> tmp;
        auto vm1_hsum = _mm256_hadd_epi32(vm1, vm1_flip);
        tmp.m = _mm_add_epi32(vm2, _mm256_extracti128_si256(vm1_hsum, 0));
        return tmp;
    }

    auto operator *(const Col<Scalar, Size>& v) const {
        auto vec = Vec<Scalar, Size>{v.m};
        return operator *(vec);
    }

    auto operator *(const Mat& mat) const {
        __m128i col[3];
        col[0] = _mm256_extracti128_si256(mat.m1, 0);
        col[1] = _mm256_extracti128_si256(mat.m1, 1);
        col[2] = mat.m2;
        static_for<Size>([&](const auto i) {
            col[i] = operator *(Vec<Scalar, Size>{col[i]});
        });
        Mat tmp;
        tmp.m1 = _mm256_insertf128_epi32(tmp.m1, col[0], 0);
        tmp.m1 = _mm256_insertf128_epi32(tmp.m1, col[1], 1);
        tmp.m2 = col[3];
        return tmp;
    }

    // Misc funcs
    YAVL_DEFINE_DATA_METHOD

    // Matrix manipulation methods
    YAVL_DEFINE_MAT3_TRANSPOSE
};

} // namespace yavl
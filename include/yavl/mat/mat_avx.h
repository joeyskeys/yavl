#pragma once

namespace yavl
{

template <typename T, uint32_t N>
static inline Vec<T, N> avx_mat_mul_vec_32_impl(const Mat<T, N>& mat, const Vec<T, N>& vec) {
    // Check comments in avx512 impl
    Vec<T, N> tmp;
    if constexpr (N == 2) {
        // TODO : integers
        tmp = mat2_mul_vec_f32(mat, vec);
    }
    else {
        if constexpr (std::is_floating_point_v<T>) {
            auto vm256 = _mm256_set1_ps(0);
            static_for<Mat<T, N>::MSize>([&](const auto i) {
                    auto v1 = vec[i << 1];
                    auto v2 = vec[i << 1 + 1];
                    auto vm = _mm256_setr_ps(v1, v1, v1, v1, v2, v2, v2, v2);
                    vm256 = _mm_fmadd_ps(mat.m[i], vm, vm256);
            });
            auto idx256 = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
            vm256 = _mm256_permutevar_ps(vm256, idx256);
            tmp.m = _mm256_extractf128_ps(_mm256_hadd_ps(vm256, vm256), 0);
        }
    }
    return tmp;
}

}
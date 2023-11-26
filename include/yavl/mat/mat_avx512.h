#pragma once

namespace yavl
{

template <>
struct Col<float, 4> {
    YAVL_TYPE_ALIAS(T, N, N)

    Scalar* arr;
    __m128 m; // As a cache to save a load for const methods

    Col(Scalar* d)
        : arr(const_cast<Scalar*>(d))
    {
        m = _mm_load_ps(arr);
    }

    // Operators
    YAVL_DEFINE_COL_BASIC_FP_OP(, ps, ps)
};

template <typename T, uint32_t N>
static inline Vec<T, N> avx512_mat_mul_vec_32_impl(const Mat<T, N>& mat, const Vec<T, N>& vec) {
    // 1. Two possible impl for mat vec/col multiplication
    // Need further benchmark to pick out one
    // (According to Intel doc, the second one maybe slightly better,
    // it's just I currently don't have device that supports avx512...)
    // (imm8 for extractf32x4 maybe illegal since it's a func
    // param rather than a template param. If this's the case, might
    // need to unfold the expression manually)
    // 2. Design problem:
    // put each impl into specific class or use if constexpr to wrap
    // up all the possible situation?
    Vec<T, N> tmp;
    static_for<Mat<T, N>::MSize>([&](const auto i) {
        if constexpr (std::is_floating_point_v<T>) {
            if constexpr (Vec<T, N>::Size == 4) {
                auto row = _mm_setr_ps(arr[i], arr[4 + i], arr[8 + i],
                    arr[12 + i]);
                tmp.arr[i] = _mm_cvtss_f32(_mm_dp_ps(row, vec.m, 0b11110001));
                /*
                tmp.m = _mm_fmadd_ps(_mm512_extractf32x4_ps(mat.m[0], i),
                    vec.m, tmp.m);
                */
            }
            else if constexpr (Vec<T, N>::Size == 3) {
                auto row = _mm_setr_ps(arr[i], arr[4 + i], arr[8 + i], 0);
                tmp.arr[i] = _mm_cvtss_f32(_mm_dp_ps(row, vec.m, 0b01110001));
            }
        }
        else {
            auto row = _mm_setr_epi32(arr[i], arr[4 + i], arr[8 + i],
                arr[12 + i]);
            tmp.arr[i] = Vec<T, N>(_mm_mul_epi32(row, vec.m)).sum();
        }
    });
    return tmp;
}

#define MAT_MUL_VEC_EXPRS                                           \
{                                                                   \
    return avx512_mat_mul_vec_32_impl(*this, v);                    \
}

#define MAT_MUL_COL_EXPRS                                           \
{                                                                   \
    auto vm = _mm_load_ps(v.arr);                                   \
    return avx512_mat_mul_vec_32_impl(*this, Vec<Scalar, Size>(vm)); \
}

#define MAT_MUL_MAT_EXPRS                                           \
{                                                                   \
    Mat tmp;                                                        \
    static_for<Size>([&](const auto i) {                            \
        Vec<T, N> curv{};                                           \
        static_for<Size>([&](const auto j) {                        \
            /*auto tmpv = operator[](j) * mat[i];*/                 \
            auto lv = _mm_load_ps(operator[](j).arr);               \
            auto rv = _mm_load_ps(mat[i].arr);                      \
            curv.m = _mm_fmadd_ps(curv.m, lv, rv);                  \
        });                                                         \
        _mm_store_ps(&tmp.arr[i << 2], curv.m);                     \
    });                                                             \
    return tmp;                                                     \
}

template <>
struct alignas(64) Mat<float, 4> {
    YAVL_MAT_ALIAS_VECTORIZED(float, 4, 4, 1)

    // Intrinsic type will contain alignment attribute and don't work well
    // as a template parameter, hence legacy array here.
    YAVL_DEFINE_MAT_UNION(__m512)

    // Ctors
    YAVL_MAT_VECTORIZED_CTOR(512, ps, __m512)

    template <typename... Ts>
        requires (std::default_initializable<Ts> && ...)
    constexpr Mat(Ts... args) {
        static_assert(sizeof...(args) == Size2);
        m[0] = _mm512_setr_ps(args...);
    }

    // Operators
    YAVL_DEFINE_MAT_OP(512, ps)

    // Misc funcs
    YAVL_DEFINE_DATA_METHOD

    // Matrix manipulation methods
    auto transpose() const {
        Mat tmp;
        auto c0 = _mm512_extractf32x4_ps(m[0], 0);
        auto c1 = _mm512_extractf32x4_ps(m[0], 1);
        auto c2 = _mm512_extractf32x4_ps(m[0], 2);
        auto c3 = _mm512_extractf32x4_ps(m[0], 3);
        _MM_TRANSPOSE4_PS(c0, c1, c2, c3);
        _mm_store_ps(tmp.arr, c0);
        _mm_store_ps(tmp.arr + 4, c1);
        _mm_store_ps(tmp.arr + 8, c2);
        _mm_store_ps(tmp.arr + 12, c3);
        return tmp;
    }
};

}
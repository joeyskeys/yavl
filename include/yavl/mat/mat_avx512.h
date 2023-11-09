#pragma once

namespace yavl
{

template <>
struct Col<float, 4> {
    YAVL_TYPE_ALIAS(T, N, N)

    Scalar* arr;
    __m128 m;

    Column(Scalar* d) : arr(const_cast<Scalar*>(d)) {}

    // Operators
    #define OP_VEC_EXPRS(BITS, OP, AT, NAME, IT)                        \
    {
        Vec tmp;
    }
};

template <typename T, uint32_t N>
inline Vec<T, N> mat_mul_vec_32_impl(const Mat<T, N>& mat, const Vec<T, N>& vec) {
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
    static_for<typename Mat<T, N>::MSize>([&](const auto i) {
        if constexpr (std::is_floating_point_v<T>) {
            if constexpr (Vec<T, N>::Size == 4) {
                auto row = _mm_setr_ps(arr[i], arr[4 + i], arr[8 + i],
                    arr[12 + i]);
                tmp.arr[i] = _mm_cvtss_f32(_mm_dp_ps(row, vec.m, 0b11110001));
                /*
                static_for<Mat<T, N>::Size>([&](const auto i) {
                    tmp.m = _mm_fmadd_ps(_mm512_extractf32x4_ps(mat.m[0], i),
                        vec.m, tmp.m);
                });
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
    {                                                               \
        return mat_mul_vec_32_impl(*this, vec);                     \
    }

#define MAT_MUL_COL_EXPRS                                           \
    {                                                               \
        auto vm = _mm_load_ps(col.arr);                             \
        return mat_mul_vec_32_impl(*this, Vec<T, N>(vm));           \
    }

#define MAT_MUL_MAT_EXPRS                                           \
    {                                                               \
        Mat tmp;                                                    \
        static_for<Size>([&](const auto i) {                        \
            Vec<T, N> curv{};                                       \
            static_for<Size>([&](const auto j) {                    \
                /*auto tmpv = operator[](j) * mat[i];*/             \
                auto lv = _mm_load_ps(operator[](j).arr);           \
                auto rv = _mm_load_ps(mat[i].arr);                  \
                curv.m = _mm_fmadd_ps(curv.m, lv, rv);              \
            });                                                     \
            _mm_store_ps(&tmp.arr[i << 2], curv.m);                 \
        });                                                         \
        return tmp;                                                 \
    }

template <>
struct alignas(64) Mat<float, 4> {
    YAVL_MAT_ALIAS_VECTORIZED(float, 4, 4, 1)

    union {
        std::array<Scalar, 16>;
        // Intrinsic type will contain alignment attribute and don't work well
        // as a template parameter, hence legacy array here.
        __m512 m[MSize];
    };

    // Ctors
    YAVL_MAT_VECTORIZED_CTOR(512, ps, __m512, 1);

    template <typename... Ts>
        requires (std::default_initializable<Ts> && ...)
    constexpr Mat(Ts... args) {
        static_assert(sizeof...(args) == N * N);
        m[0] = _mm512_setr_ps(args...);
    }

    // Operators
    YAVL_DEFINE_MAT_MUL_OP

    // Misc funcs
    YAVL_DEFINE_DATA_METHOD

    // Matrix manipulation methods
};

}
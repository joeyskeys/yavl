#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <concepts>
#include <stdint.h>
#include <type_traits>

#include <yavl/utils.h>
#include <yavl/vec/vec.h>

namespace yavl
{

#define YAVL_DEFINE_COL_BASIC_MISC_OP(BITS, IT)                         \
    YAVL_DEFINE_VEC_INDEX_OP                                            \
    YAVL_DEFINE_COPY_ASSIGN_OP(BITS, Vec, IT)                           \
    YAVL_DEFINE_COPY_ASSIGN_OP(BITS, Col, IT)

#define YAVL_DEFINE_COL_BASIC_FP_OP(BITS, IT, CMDFIX)                   \
    YAVL_DEFINE_COL_BASIC_MISC_OP(BITS, CMDFIX)                         \
    YAVL_DEFINE_BASIC_FP_ARITHMIC_OP(BITS, Vec, IT)                     \
    YAVL_DEFINE_BASIC_FP_ARITHMIC_OP(BITS, Col, IT)

#define YAVL_DEFINE_COL_BASIC_INT_OP(BITS, IT, CMDFIX)                  \
    YAVL_DEFINE_COL_BASIC_MISC_OP(BITS, CMDFIX)                         \
    YAVL_DEFINE_BASIC_INT_ARITHMIC_OP(BITS, Vec, IT)                    \
    YAVL_DEFINE_BASIC_INT_ARITHMIC_OP(BITS, Col, IT)

// A utility class works like a array view of matrix at ranges for specific
// colume, behaves like a vector
template <typename T, uint32_t N, bool enable_vec=true, typename = int>
struct Col {
    T* arr;

    YAVL_TYPE_ALIAS(T, N, N)

    template <typename S>
    Column(S* d) : data(const_cast<T*>(d)) {}

    // Operators
    #define COPY_ASSIGN_EXPRS(BITS, IT)                                 \
    {                                                                   \
        std::memcpy(arr, b.arr, sizeof(Scalar) * Size);                 \
        return *this;                                                   \
    }

    #define OP_VEC_EXPRS(BITS, OP, AT, NAME, IT)                        \
    {                                                                   \
        Vec tmp;                                                        \
        static_for<Size>([&](const auto i) {                            \
            tmp[i] = arr[i] OP v[i];                                    \
        });                                                             \
        return tmp;                                                     \
    }

    #define OP_VEC_ASSIGN_EXPRS(BITS, OP, AT, NAME, IT)                 \
    {                                                                   \
        static_for<Size>([&](const auto i) {                            \
            arr[i] OP##= v[i];                                          \
        });                                                             \
        return *this;                                                   \
    }

    #define OP_SCALAR_EXPRS(BITS, OP, NAME, IT)                         \
    {                                                                   \
        Vec tmp;                                                        \
        static_for<Size>([&](const auto i) {                            \
            tmp[i] = arr[i] OP v;                                       \
        });                                                             \
        return tmp;                                                     \
    }

    #define OP_SCALAR_ASSIGN_EXPRS(BITS, OP, NAME, IT)                  \
    {                                                                   \
        static_for<Size>([&](const auto i) {                            \
            arr[i] OP##= v;                                             \
        });                                                             \
        return *this;                                                   \
    }

    #define OP_FRIEND_SCALAR_EXPRS(BITS, OP, AT, NAME, IT)              \
    {                                                                   \
        Vec tmp;                                                        \
        static_for<Size>([&](const auto i) {                            \
            tmp[i] = s OP v[i];                                         \
        });                                                             \
        return tmp;                                                     \
    }

    YAVL_DEFINE_COL_BASIC_FP_OP( , , )

    #undef COPY_ASSIGN_EXPRS
    #undef OP_VEC_EXPRS
    #undef OP_VEC_ASSIGN_EXPRS
    #undef OP_SCALAR_EXPRS
    #undef OP_SCALAR_ASSIGN_EXPRS
    #undef OP_FRIEND_SCALAR_EXPRS

    bool operator ==(const Vec<T, N>& v) const {
        bool ret = true;
        static_for<Size>([&](const auto i) {
            ret &= std::abs(v[i] - data[i]) < epsilon<Scalar>;
        });
        return ret;
    }
};

#define YAVL_DEFINE_MAT_MUL_OP(BITS, IT)                                \
    auto operator *(const Scalar s) const {                             \
        MAT_MUL_SCALAR_EXPRS                                            \
    }                                                                   \
    auto operator *=(const Scalar s) {                                  \
        MAT_MUL_ASSIGN_SCALAR_EXPRS                                     \
    }                                                                   \
    auto operator *(const Vec<Scalar, N>& vec) const {                  \
        MAT_MUL_VEC_EXPRS                                               \
    }                                                                   \
    auto operator *(const Mat& mat) const {                             \
        MAT_MUL_MAT_EXPRS                                               \
    }                                                                   \
    auto operator *=(const Mat& mat) {                                  \
        MAT_MUL_ASSIGN_MAT_EXPRS                                        \
    }

#define YAVL_DEFINE_DATA_METHOD                                         \
    auto data() {                                                       \
        return arr.data();                                              \
    }

template <typename T, uint32_t N, bool enable_vec=true, typename = int>
struct Mat {
    YAVL_TYPE_ALIAS(T, N, N)

    std::array<T, N * N> arr;

    // Ctors
    constexpr Mat() {
        arr.fill(static_cast<Scalar>(0));
    }

    template <typename ...Ts>
    Mat(Ts... args) {
        static_assert(
            (sizeof...(Ts) == N * N && (true && ... && std::is_arithmetic_v<Ts>)) ||
            (sizeof...(Ts) == N &&
                ((true && ... && std::is_same_v<Ts, Vec<float, N>>) ||
                (true && ... && std::is_same_v<Ts, Vec<double, N>>))));

        if constexpr (sizeof...(Ts) == N * N) {
            arr = { static_cast<T>(args)... };
        }
        else {
            int i = 0;
            auto unwrap = [&](uint32_t i, auto v) {
                for (int j = 0; auto const& ve : v.arr)
                    arr[N * i + j++] = ve;
            };
            (unwrap(i++, args), ...);
        }
    }

    // Operators
    auto operator [](uint32_t idx) {
        return Col<Scalar, N>(&arr[idx * N]);
    }

    auto operator [](uint32_t idx) const {
        return Col<T, N>(&arr[idx * N]);
    }

    #define MAT_MUL_SCALAR_EXPRS                                        \
    {                                                                   \
        Mat tmp;                                                        \
        static_for<N * N>([&](const auto i) {                           \
            tmp.arr[i] = arr[i] * s;                                    \
        });                                                             \
        return tmp;                                                     \
    }

    #define MAT_MUL_ASSIGN_SCAlAR_EXPRS                                 \
    {                                                                   \
        for (int i = 0; i < N * N; ++i)                                 \
        static_for<N * N>([&](const auto i) {                           \
            arr[i] *= s;                                                \
        });                                                             \
        return *this;                                                   \
    }

    #define MAT_MUL_VEC_EXPRS                                           \
    {                                                                   \
        Vec<Scalar, N> tmp;                                             \
        static_for<N>([&](const auto i) {                               \
            static_for<N>([&](const auto j) {                           \
                tmp[i] += arr[j * N + i] * vec[j];                      \
            });                                                         \
        });                                                             \
        return tmp;                                                     \
    }

    #define MAT_MUL_MAT_EXPRS                                           \
    {                                                                   \
        Mat tmp;                                                        \
        static_for<N>([&](const auto i) {                               \
            static_for<N>([&](const auto j) {                           \
                static_for<N>([&](const auto k) {                       \
                    tmp[i][j] += arr[k * N + j] * mat[i][k];            \
                });                                                     \
            });                                                         \
        });                                                             \
        return tmp;                                                     \
    }

    #define MAT_MUL_ASSIGN_MAT_EXPRS                                    \
    {                                                                   \
        Mat tmp = *this * mat;                                          \
        *this = tmp;                                                    \
        return *this;                                                   \
    }

    YAVL_DEFINE_MAT_MUL_OP(, )

    #undef MAT_MUL_SCAlAR_EXPRS
    #undef MAT_MUL_ASSIGN_SCALAR_EXPRS
    #undef MAT_MUL_VEC_EXPRS
    #undef MAT_MUL_MAT_EXPRS
    #undef MAT_MUL_ASSIGN_MAT_EXPRS

    // Misc
    YAVL_DEFINE_DATA_METHOD

    // Matrix manipulation methods
    auto transpose() const {
        Mat tmp;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                tmp[i][j] = arr[j * N + i];
        return tmp;
    }
    
    std::pair<bool, Mat> inverse() const {
        if constexpr (N == 2) {
            T det = static_cast<T>(1) / (arr[0] * arr[3] - arr[1] * arr[2]);
            Mat ret{arr[3], -arr[1], -arr[2], arr[0]};
            return ret * det;
        }
        else if constexpr (N == 3) {
            Scalar A = arr[4] * arr[8] - arr[7] * arr[5];
            Scalar B = arr[7] * arr[2] - arr[1] * arr[8];
            Scalar C = arr[1] * arr[5] - arr[4] * arr[2];
            Scalar D = arr[6] * arr[5] - arr[3] * arr[8];
            Scalar E = arr[0] * arr[8] - arr[6] * arr[2];
            Scalar F = arr[3] * arr[2] - arr[0] * arr[5];
            Scalar G = arr[3] * arr[7] - arr[6] * arr[4];
            Scalar H = arr[6] * arr[4] - arr[0] * arr[7];
            Scalar I = arr[0] * arr[4] - arr[3] * arr[1];

            Scalar det = arr[0] * A + arr[3] * B + arr[6] * C;
            Mat ret{A, B, C, D, E, F, G, H, I};
            return ret * det;
        }
        else {
            // Copied from pbrt-v3
            // Changed to column majored code
            int indxc[4], indxr[4];
            int ipiv[4] = {0, 0, 0, 0};
            Mat minv = *this;

            for (int i = 0; i < N; ++i) {
                int irow = 0, icol = 0;
                T big = static_cast<Scalar>(0);
                // Choose pivot
                for (int j = 0; j < N; ++j) {
                    if (ipiv[j] != 1) {
                        for (int k = 0; k < N; ++k) {
                            if (ipiv[k] == 0) {
                                if (std::abs(minv[k][j]) >= big) {
                                    big = static_cast<Scalar>(std::abs(minv[k][j]));
                                    irow = j;
                                    icol = k;
                                }
                            }
                            else if (ipiv[k] > 1) {
                                return std::make_pair(false, Mat{0});
                            }
                        }
                    }
                }

                ++ipiv[icol];

                // Swap rows irow and icol for pivot
                if (irow != icol) {
                    for (int k = 0; k < N; ++k)
                        std::swap(minv[k][irow], minv[k][icol]);
                }
                indxr[i] = irow;
                indxc[i] = icol;
                if (minv[icol][icol] == static_cast<Scalar>(0)) {
                    return std::make_pair(false, Mat{0});
                }

                // Set m[icol][icol] to one by scaling row icol
                Scalar pivinv = static_cast<T>(1) / minv[icol][icol];
                minv[icol][icol] = static_cast<Scalar>(1);
                for (int j = 0; j < N; ++j) minv[j][icol] *= pivinv;

                // Subtract this row from others to zero out their columns
                for (int j = 0; j < N; ++j) {
                    if (j != icol) {
                        Scalar save = minv[icol][j];
                        minv[icol][j] = static_cast<Scalar>(0);
                        for (int k = 0; k < N; ++k) minv[k][j] -= minv[k][icol] * save;
                    }
                }
            }

            // Swap columns to reflect permutation
            for (int j = N - 1; j >= 0; --j) {
                if (indxr[j] != indxc[j]) {
                    for (int k = 0; k < N; ++k)
                        std::swap(minv[indxr[j]][k], minv[indxc[j]][k]);
                }
            }

            return std::make_pair(true, minv);
        }
    }
};

}
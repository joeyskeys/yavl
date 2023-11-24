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

#define YAVL_MAT_ALIAS(TYPE, N, INTRIN_N)                               \
    YAVL_TYPE_ALIAS(TYPE, N, INTRIN_N)                                  \
    static constexpr uint32_t Size2 = Size * Size;

#define YAVL_DEFINE_COL_MISC_FUNCS                                      \
    inline auto* data() {                                               \
        return arr;                                                     \
    }                                                                   \
    inline auto* data() const {                                         \
        return arr;                                                     \
    }

#define YAVL_DEFINE_COL_BASIC_MISC_OP(BITS, IT)                         \
    YAVL_DEFINE_VEC_INDEX_OP                                            \
    YAVL_DEFINE_COPY_ASSIGN_OP(BITS, Vec, IT)                           \
    YAVL_DEFINE_COPY_ASSIGN_OP(BITS, Col, IT)

#define YAVL_DEFINE_COL_OP(BITS, OP, AT, NAME, IT)                      \
    YAVL_DEFINE_VEC_OP(BITS, OP, AT, NAME, IT)                          \
    YAVL_DEFINE_FRIEND_OP(BITS, OP, AT, NAME, IT)

#define YAVL_DEFINE_COL_FP_OP(BITS, AT, IT)                             \
    YAVL_DEFINE_COL_OP(BITS, +, AT, add, IT)                            \
    YAVL_DEFINE_COL_OP(BITS, -, AT, sub, IT)                            \
    YAVL_DEFINE_COL_OP(BITS, *, AT, mul, IT)                            \
    YAVL_DEFINE_COL_OP(BITS, /, AT, div, IT)

#define YAVL_DEFINE_COL_INT_OP(BITS, AT, IT)                            \
    YAVL_DEFINE_COL_OP(BITS, +, AT, add, IT)                            \
    YAVL_DEFINE_COL_OP(BITS, -, AT, sub, IT)                            \
    YAVL_DEFINE_COL_OP(BITS, *, AT, mullo, IT)

#define YAVL_DEFINE_COL_BASIC_FP_OP(BITS, IT, CMDFIX)                   \
    YAVL_DEFINE_COL_BASIC_MISC_OP(BITS, CMDFIX)                         \
    YAVL_DEFINE_FP_ARITHMIC_OP(BITS, Vec, IT)                           \
    YAVL_DEFINE_COL_FP_OP(BITS, Col, IT)

#define YAVL_DEFINE_COL_BASIC_INT_OP(BITS, IT, CMDFIX)                  \
    YAVL_DEFINE_COL_BASIC_MISC_OP(BITS, CMDFIX)                         \
    YAVL_DEFINE_INT_ARITHMIC_OP(BITS, Vec, IT)                          \
    YAVL_DEFINE_COL_INT_OP(BITS, Col, IT)

#define YAVL_DEFINE_COL_DOT_FUNC                                        \
    inline Scalar dot(const Vec<Scalar, Size>& b) const {               \
        COL_DOT_VEC_EXPRS                                               \
    }                                                                   \
    inline Scalar dot(const Col<Scalar, Size>& b) const {               \
        COL_DOT_COL_EXPRS                                               \
    }

// A utility class works like a array view of matrix at ranges for specific
// colume, behaves like a vector
template <typename T, uint32_t N, bool enable_vec=true, typename = int>
struct Col {
    YAVL_TYPE_ALIAS(T, N, N)

    Scalar* arr;

    Col(const Scalar* d) : arr(const_cast<Scalar*>(d)) {}

    // Miscs
    YAVL_DEFINE_COL_MISC_FUNCS

    // Operators
    #define COPY_ASSIGN_EXPRS(BITS, IT)                                 \
    {                                                                   \
        std::memcpy(arr, b.data(), sizeof(Scalar) * Size);              \
        return *this;                                                   \
    }

    #define OP_VEC_EXPRS(BITS, OP, AT, NAME, IT)                        \
    {                                                                   \
        Vec<Scalar, Size> tmp;                                          \
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
        Vec<Scalar, Size> tmp;                                          \
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
        Vec<Scalar, Size> tmp;                                          \
        static_for<Size>([&](const auto i) {                            \
            tmp[i] = s OP v[i];                                         \
        });                                                             \
        return tmp;                                                     \
    }

    YAVL_DEFINE_COL_BASIC_FP_OP( , , )

    //#undef COPY_ASSIGN_EXPRS
    #undef OP_VEC_EXPRS
    #undef OP_VEC_ASSIGN_EXPRS
    #undef OP_SCALAR_EXPRS
    #undef OP_SCALAR_ASSIGN_EXPRS
    #undef OP_FRIEND_SCALAR_EXPRS

    bool operator ==(const Vec<T, N>& v) const {
        bool ret = true;
        static_for<Size>([&](const auto i) {
            ret &= std::abs(v[i] - arr[i]) < epsilon<Scalar>;
        });
        return ret;
    }

    // Geometry
    #define COL_DOT_VEC_EXPRS                                           \
    {                                                                   \
        Scalar sum{0};                                                  \
        static_for<Size>([&](const auto i) {                            \
            sum += arr[i] * b.arr[i];                                   \
        });                                                             \
        return sum;                                                     \
    }

    #define COL_DOT_COL_EXPRS COL_DOT_VEC_EXPRS

    YAVL_DEFINE_COL_DOT_FUNC

    #undef COL_DOT_VEC_EXPRS
    #undef COL_DOT_COL_EXPRS
};

#define YAVL_DEFINE_MAT_INDEX_OP                                        \
    auto operator [](uint32_t idx) {                                    \
        return Col<Scalar, Size>(&arr[idx * Size]);                     \
    }                                                                   \
    const auto operator [](uint32_t idx) const {                        \
        return Col<Scalar, Size>(&arr[idx * Size]);                     \
    }

#define YAVL_DEFINE_MAT_MUL_OP(BITS, IT)                                \
    auto operator *(const Scalar s) const {                             \
        MAT_MUL_SCALAR_EXPRS(BITS, IT)                                  \
    }                                                                   \
    auto operator *=(const Scalar s) {                                  \
        MAT_MUL_ASSIGN_SCALAR_EXPRS(BITS, IT)                           \
    }                                                                   \
    auto operator *(const Vec<Scalar, Size>& v) const {                 \
        MAT_MUL_VEC_EXPRS                                               \
    }                                                                   \
    auto operator *(const Col<Scalar, Size>& v) const {                 \
        MAT_MUL_COL_EXPRS                                               \
    }                                                                   \
    auto operator *(const Mat& mat) const {                             \
        MAT_MUL_MAT_EXPRS                                               \
    }                                                                   \
    auto operator *=(const Mat& mat) {                                  \
        Mat tmp = *this * mat;                                          \
        *this = tmp;                                                    \
        return *this;                                                   \
    }

#define YAVL_DEFINE_MAT_OP(BITS, IT)                                    \
    YAVL_DEFINE_MAT_INDEX_OP                                            \
    YAVL_DEFINE_MAT_MUL_OP(BITS, IT)

#define YAVL_DEFINE_DATA_METHOD                                         \
    auto data() {                                                       \
        return arr.data();                                              \
    }

template <typename T, uint32_t N, bool enable_vec=true, typename = int>
struct Mat {
    YAVL_MAT_ALIAS(T, N, N)

    std::array<T, Size2> arr;

    // Ctors
    constexpr Mat() {
        arr.fill(static_cast<Scalar>(0));
    }

    template <typename ...Ts>
    Mat(Ts... args) {
        static_assert(
            (sizeof...(Ts) == Size2 && (true && ... && std::is_arithmetic_v<Ts>)) ||
            (sizeof...(Ts) == Size &&
                ((true && ... && std::is_same_v<Ts, Vec<float, Size>>) ||
                (true && ... && std::is_same_v<Ts, Vec<double, Size>>))) ||
            (sizeof...(Ts) == 1 && (true && ... && std::is_convertible_v<Ts, Scalar>)));

        if constexpr (sizeof...(Ts) == Size2) {
            arr = { static_cast<T>(args)... };
        }
        else if constexpr (sizeof...(Ts) == Size) {
            int i = 0;
            auto unwrap = [&](uint32_t i, auto v) {
                for (int j = 0; auto const& ve : v.arr)
                    arr[N * i + j++] = ve;
            };
            (unwrap(i++, args), ...);
        }
        else {
            arr.fill(static_cast<Scalar>(args)...);
        }
    }

    // Operators
    #define MAT_MUL_SCALAR_EXPRS(BITS, IT)                              \
    {                                                                   \
        Mat tmp;                                                        \
        static_for<Size2>([&](const auto i) {                           \
            tmp.arr[i] = arr[i] * s;                                    \
        });                                                             \
        return tmp;                                                     \
    }

    #define MAT_MUL_ASSIGN_SCALAR_EXPRS(BITS, IT)                       \
    {                                                                   \
        static_for<Size2>([&](const auto i) {                           \
            arr[i] *= s;                                                \
        });                                                             \
        return *this;                                                   \
    }

    #define MAT_MUL_VEC_EXPRS                                           \
    {                                                                   \
        Vec<Scalar, Size> tmp;                                          \
        static_for<Size>([&](const auto i) {                            \
            static_for<Size>([&](const auto j) {                        \
                tmp[i] += arr[j * Size + i] * v[j];                     \
            });                                                         \
        });                                                             \
        return tmp;                                                     \
    }

    #define MAT_MUL_COL_EXPRS MAT_MUL_VEC_EXPRS                     

    #define MAT_MUL_MAT_EXPRS                                           \
    {                                                                   \
        Mat tmp;                                                        \
        static_for<Size>([&](const auto i) {                            \
            static_for<Size>([&](const auto j) {                        \
                static_for<Size>([&](const auto k) {                    \
                    tmp[i][j] += arr[k * Size + j] * mat[i][k];         \
                });                                                     \
            });                                                         \
        });                                                             \
        return tmp;                                                     \
    }

    YAVL_DEFINE_MAT_OP(, )

    #undef MAT_MUL_SCALAR_EXPRS
    #undef MAT_MUL_ASSIGN_SCALAR_EXPRS
    #undef MAT_MUL_VEC_EXPRS
    #undef MAT_MUL_COL_EXPRS
    #undef MAT_MUL_MAT_EXPRS
    #undef MAT_MUL_ASSIGN_MAT_EXPRS

    // Misc
    YAVL_DEFINE_DATA_METHOD

    // Matrix manipulation methods
    auto transpose() const {
        Mat tmp;
        for (int i = 0; i < Size; ++i)
            for (int j = 0; j < Size; ++j)
                tmp[i][j] = arr[j * Size + i];
        return tmp;
    }
    
    std::pair<bool, Mat> inverse() const {
        if constexpr (Size == 2) {
            T det = static_cast<T>(1) / (arr[0] * arr[3] - arr[1] * arr[2]);
            Mat ret{arr[3], -arr[1], -arr[2], arr[0]};
            return ret * det;
        }
        else if constexpr (Size == 3) {
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

            for (int i = 0; i < Size; ++i) {
                int irow = 0, icol = 0;
                T big = static_cast<Scalar>(0);
                // Choose pivot
                for (int j = 0; j < Size; ++j) {
                    if (ipiv[j] != 1) {
                        for (int k = 0; k < Size; ++k) {
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
                    for (int k = 0; k < Size; ++k)
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
                for (int j = 0; j < Size; ++j) minv[j][icol] *= pivinv;

                // Subtract this row from others to zero out their columns
                for (int j = 0; j < Size; ++j) {
                    if (j != icol) {
                        Scalar save = minv[icol][j];
                        minv[icol][j] = static_cast<Scalar>(0);
                        for (int k = 0; k < Size; ++k) minv[k][j] -= minv[k][icol] * save;
                    }
                }
            }

            // Swap columns to reflect permutation
            for (int j = Size - 1; j >= 0; --j) {
                if (indxr[j] != indxc[j]) {
                    for (int k = 0; k < Size; ++k)
                        std::swap(minv[indxr[j]][k], minv[indxc[j]][k]);
                }
            }

            return std::make_pair(true, minv);
        }
    }
};

// Mat type aliasing
template <typename T>
using Mat2 = Mat<T, 2>;

template <typename T>
using _Mat2 = Mat<T, 2, false>;

template <typename T>
using Mat3 = Mat<T, 3>;

template <typename T>
using _Mat3 = Mat<T, 3, false>;

template <typename T>
using Mat4 = Mat<T, 4>;

template <typename T>
using _Mat4 = Mat<T, 4, false>;

template <typename T>
using Mat2f = Mat2<float>;

template <typename T>
using Mat2d = Mat2<double>;

template <typename T>
using Mat2i = Mat2<int>;

template <typename T>
using Mat2u = Mat2<uint32_t>;

template <typename T>
using Mat3f = Mat3<float>;

template <typename T>
using Mat3d = Mat3<double>;

template <typename T>
using Mat3i = Mat3<int>;

template <typename T>
using Mat3u = Mat3<uint32_t>;

template <typename T>
using Mat4f = Mat4<float>;

template <typename T>
using Mat4d = Mat4<double>;

template <typename T>
using Mat4i = Mat4<int>;

template <typename T>
using Mat4u = Mat4<uint32_t>;
}
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

template <typename T, uint32_t N, bool enable_vec=true, typename = int>
struct Col {
    T* data;

    template <typename S>
    Column(S* d) : data(const_cast<T*>(d)) {}

    T& operator [](unsigned int idx) {
        return data[idx];
    }
    
    const T& operator [](uint32_t idx) const {
        return data[idx];
    }

    Col& operator =(const Vec<T, N>& v) {

    }

    Col& operator =(const Col& c) {

    }

    void operator +=(const Vec<T, N>& v) {

    }

    void operator -=(const Vec<T, N>& v) {

    }

    bool operator ==(const Vec<T, N>& v) const {

    }
};

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
        return Col<T, N>(&arr[idx * N]);
    }

    auto operator [](uint32_t idx) const {
        return Col<T, N>(&arr[idx * N]);
    }

    auto transpose() const {
        Mat tmp;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                tmp[i][j] = arr[j * N + i];
        return tmp;
    }

    auto operator *(const Scalar s) const {
        Mat tmp;
        for (int i = 0; i < N * N; ++i)
            tmp.arr[i] = arr[i] * s;
        return tmp;
    }

    auto operator *=(const Scalar s) {
        for (int i = 0; i < N * N; ++i)
            arr[i] *= s;
        return *this;
    }

    auto operator *(const Vec<T, N>& vec) const {
        Vec<T, N> tmp;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                tmp[i] += arr[j * N + i] * vec[j];
        return tmp;
    }

    auto operator *(const Mat& mat) const {
        Mat tmp;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                for (int k = 0; k < N; ++k)
                    tmp[i][j] += arr[k * N + j] * mat[i][k];
        return tmp;
    }

    auto operator *=(const Mat& mat) {
        Mat tmp = *this * mat;
        *this = tmp;
        return *this;
    }

    auto data() {
        return arr.data();
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
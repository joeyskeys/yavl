#include <catch2/catch_all.hpp>

#include <yavl/yavl.h>

using namespace yavl;

using Catch::Approx;

#define PROPER_EQUAL(EXPR, VALUE, VAR)                                  \
    {                                                                   \
        if constexpr (std::is_floating_point_v<typename decltype(VAR)::Scalar>) \
            REQUIRE(EXPR == Approx(VALUE));                             \
        else                                                            \
            REQUIRE(EXPR == VALUE);                                     \
    }

template <typename M>
void ctor_test(const M& m, const typename M::Scalar s) {
    if constexpr (std::is_floating_point_v<typename M::Scalar>) {
        for (int i = 0; i < M::Size2; ++i)
            REQUIRE(m.arr[i] == Approx(s));
    }
    else {
        for (int i = 0; i < M::Size2; ++i)
            REQUIRE(m.arr[i] == s);
    }
}

TEMPLATE_PRODUCT_TEST_CASE("Mat tests", "[mat]", (Mat2, Mat3, Mat4), (float, double)) {
    // Ctor tests
    TestType mat0;
    TestType mat1{1};
    TestType mat2;
    Vec<typename TestType::Scalar, TestType::Size> vec;

    if constexpr (TestType::Size == 2) {
        mat2 = {0, 1, 2, 3};
        vec = {1, 2};
    }
    else if constexpr (TestType::Size == 3) {
        mat2 = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        vec = {1, 2, 3};
    }
    else if constexpr (TestType::Size == 4) {
        mat2 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14, 15};
        vec = {1, 2, 3, 4};
    }
    else
        static_assert(TestType::Size < 5, "Should not reach here");

    SECTION("Ctor tests") {
        ctor_test(mat0, 0);
        ctor_test(mat1, 1);
        if constexpr (TestType::Size == 2) {
            PROPER_EQUAL(mat2[1][0], 2, mat2);
        }
        else if constexpr (TestType::Size == 3) {
            PROPER_EQUAL(mat2[1][0], 3, mat2);
        }
        else {
            PROPER_EQUAL(mat2[1][0], 4, mat2);
        }
    }

    SECTION("Operator tests") {
        auto mat3 = mat1 * 2;
        PROPER_EQUAL(mat3.arr[0], 2, mat3);
        PROPER_EQUAL(mat3.arr[3], 2, mat3);

        mat1 *= 3;
        PROPER_EQUAL(mat1.arr[0], 3, mat1);
        PROPER_EQUAL(mat1.arr[3], 3, mat1);

        auto vec1 = mat2 * vec;
        if constexpr (TestType::Size == 2) {
            PROPER_EQUAL(vec1.arr[0], 4, vec1);
            PROPER_EQUAL(vec1.arr[1], 7, vec1);
        }
        else if constexpr (TestType::Size == 3) {
            PROPER_EQUAL(vec1.arr[0], 24, vec1);
            PROPER_EQUAL(vec1.arr[1], 30, vec1);
        }
        else {
            PROPER_EQUAL(vec1.arr[0], 80, vec1);
            PROPER_EQUAL(vec1.arr[1], 90, vec1);
        }

        auto vec2 = mat2 * mat1[0];
        if constexpr (TestType::Size == 2) {
            PROPER_EQUAL(vec2.arr[0], 6, vec2);
            PROPER_EQUAL(vec2.arr[1], 12, vec2);
        }
        else if constexpr (TestType::Size == 3) {
            PROPER_EQUAL(vec2.arr[0], 27, vec2);
            PROPER_EQUAL(vec2.arr[1], 36, vec2);
        }
        else {
            PROPER_EQUAL(vec2.arr[0], 72, vec2);
            PROPER_EQUAL(vec2.arr[1], 84, vec2);
        }

        auto mat6 = mat2 * mat2;
        if constexpr (TestType::Size == 2) {
            PROPER_EQUAL(mat6.arr[0], 2, mat6);
            PROPER_EQUAL(mat6.arr[3], 11, mat6);
        }
        else if constexpr (TestType::Size == 3) {
            PROPER_EQUAL(mat6[0][0], 15, mat6);
            PROPER_EQUAL(mat6[1][1], 54, mat6);
        }
        else {
            PROPER_EQUAL(mat6[0][0], 56, mat6);
            PROPER_EQUAL(mat6[1][1], 174, mat6);
        }
    }

    SECTION("Method tests") {
        auto mat3 = mat2.transpose();
        if constexpr (TestType::Size == 2) {
            PROPER_EQUAL(mat3.arr[3], 3, mat3);
            PROPER_EQUAL(mat3.arr[2], 1, mat3);
        }
        else if constexpr (TestType::Size == 3) {
            PROPER_EQUAL(mat3[1][1], 4, mat3);
            PROPER_EQUAL(mat3[1][0], 1, mat3);
        }
        else {
            PROPER_EQUAL(mat3[1][1], 5, mat3);
            PROPER_EQUAL(mat3[1][0], 1, mat3);
        }
    }

    SECTION("Vectorization tests") {
        if constexpr (is_float_v<typename TestType::Scalar> && TestType::Size == 4)
            REQUIRE(TestType::vectorized == true);
    }
}
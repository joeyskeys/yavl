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
    
    SECTION("Ctor tests") {
        ctor_test(mat0, 0);
        ctor_test(mat1, 1);
    }
}
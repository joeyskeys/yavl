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

TEMPLATE_PRODUCT_TEST_CASE("Mat tests", "[mat]", (Mat2, Mat3, Mat4), (float, double)) {
    // Ctor tests
    TestType mat0;
    TestType mat1{1};
    
    SECTION("Ctor tests") {
        
    }
}
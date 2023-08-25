#include <type_traits>

#include <catch2/catch_all.hpp>

#include <yavml/intrin.h>
#include <yavml/utils.h>

using namespace yavml;

TEST_CASE("Macro tests", "utils") {
    REQUIRE(has_avx512f == false);
    REQUIRE(has_avx2 == true);
    REQUIRE(has_avx == true);
    REQUIRE(has_sse42 == true);
}
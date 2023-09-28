#include <type_traits>

#include <catch2/catch_all.hpp>

#include <yavl/intrin.h>
#include <yavl/utils.h>

using namespace yavl;

TEST_CASE("Macro tests", "utils") {
    REQUIRE(has_avx512f == false);
    REQUIRE(has_avx2 == true);
    REQUIRE(has_avx == true);
    REQUIRE(has_sse42 == true);
}
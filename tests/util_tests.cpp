#include <type_traits>

#include <catch2/catch_all.hpp>

#include <yavl/utils.h>

using namespace yavl;

TEST_CASE("Util tests", "utils") {
    int x = 0;

    SECTION("Static for tests") {
        static_for<5>([&](int i) {
            x += 1;
        });

        REQUIRE(x == 5);

        static_for<true ? 2 : 1>([&](int i) {
            x += 1;
        });

        REQUIRE(x == 7);
    }
}
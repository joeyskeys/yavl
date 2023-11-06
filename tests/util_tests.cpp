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

    SECTION("Parametr pack split") {
        auto func = [&](auto a1, auto a2) {
            x = a1 + a2;
        };
        //SplitHelper<2, decltype(func), int, int, int, int, int, int>::even_split(func, 1, 2, 3, 4, 5, 6);
        SplitHelper<2>::even_split(func, 1, 2, 3, 4);

        REQUIRE(x == 11);
    }
}
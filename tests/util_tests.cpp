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

        static_for<2>([&](int i) {
            static_for<2>([&](int j) {
                x += 1;
            });
        });

        REQUIRE(x == 11);
    }

    SECTION("Parametr pack split") {
        auto ht = head<1>(1, 2, 2.0, 2.f, 'c', "std");
        REQUIRE(std::tuple_size_v<decltype(ht)> == 1);
        REQUIRE(std::get<0>(ht) == 1);

        auto ht2 = head<2>(1, 2);
        REQUIRE(std::tuple_size_v<decltype(ht2)> == 2);

        auto ht3 = head<2>(1, 2, 3, 4);
        REQUIRE(std::get<0>(ht3) == 1);

        auto st = skip<1>(1, 2);
        REQUIRE(std::get<0>(st) == 2);

        auto tt = tail<1>(1, 2);
        REQUIRE(std::tuple_size_v<decltype(tt)> == 1);
        REQUIRE(std::get<0>(tt) == 2);

        auto tt2 = tail<2>(1, 2);
        REQUIRE(std::tuple_size_v<decltype(tt2)> == 2);
        REQUIRE(std::get<1>(tt2) == 2);

        auto tt3 = tail<2>(1, 2, 3, 4);
        REQUIRE(std::tuple_size_v<decltype(tt3)> == 2);
        REQUIRE(std::get<0>(tt3) == 3);

        //dispatch<2>(func, 1, 2, 3, 4);
        //REQUIRE(x == 7);
    }

    SECTION("Traits") {
        REQUIRE(is_int32_v<int> == true);
        REQUIRE(is_int32_v<uint32_t> == true);
    }
}
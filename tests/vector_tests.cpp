#include <catch2/catch_all.hpp>

#include <yavl/yavl.h>

using namespace yavl;

using Catch::Approx;

TEST_CASE("Vector tests", "vector") {
    Vec<float, 4> v{1, 2, 3, 4};
    REQUIRE(v.x == 1);
    //auto x = _mm_cvtss_f32(v.v);
    //REQUIRE(x == 1);
    REQUIRE(v.w == 4);
    REQUIRE(v.vectorized == true);

    Vec<float, 4> v2{2, 3, 4, 5};
    //auto v3 = _mm_add_ps(v.v, v2.v);
    //auto y = _mm_cvtss_f32(v3);
    //REQUIRE(y == 3);

    //Vec<float, 16> v4;
    //REQUIRE(!std::is_same_v<intrinsic_type_t<float, 512>, __m512>);
    //REQUIRE(std::is_same_v<intrinsic_type_t<float, 512>, std::array<__m256, 2>>);

    Vec<float, 4> v5{1};
    REQUIRE(v5.x == 1);

    Vec<int, 4> v6{1, 2, 3, 4};
    Vec<int, 4> v7{2, 3, 4, 5};
    auto v8 = v6 + v7;
    REQUIRE(v8.x == 3);

    auto v9 = v7.shuffle(2, 3, 0 ,1);
    REQUIRE(v9.x == 4);

    auto v5_sum = v5.sum();
    REQUIRE(v5_sum == 4);

    Vec<float, 4> v10{1, -1, 1, -1};
    auto v11 = v10.abs();
    REQUIRE(v11.x == 1);
    REQUIRE(v11.y == 1);

    Vec<float, 4> v12{1}, v13{2};
    auto v14 = v12.lerp(v13, 0.5);
    REQUIRE(v14.x == Approx(1.5));

    Vec<float, 4, false> v15{1};
    REQUIRE(v14.vectorized == true);
    REQUIRE(v15.vectorized == false);

    Vec<float, 3> v16;
    REQUIRE(v16.vectorizaed == true);
    REQUIRE(sizeof(v16) == 16);
}
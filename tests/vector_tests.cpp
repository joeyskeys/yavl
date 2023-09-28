#include <catch2/catch_all.hpp>

#include <yavl/yavl.h>

using namespace yavl;

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
}
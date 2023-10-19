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

template <typename V>
void check_vec_components(const V& v, const typename V::Scalar s) {
    if constexpr (std::is_floating_point_v<typename V::Scalar>) {
        REQUIRE(v.x == Approx(s));
        REQUIRE(v.r == Approx(s));
        REQUIRE(v.y == Approx(s));
        REQUIRE(v.g == Approx(s));
        if constexpr (V::Size > 2) {
            REQUIRE(v.z == Approx(s));
            REQUIRE(v.b == Approx(s));
        }
        if constexpr (V::Size > 3) {
            REQUIRE(v.w == Approx(s));
            REQUIRE(v.a == Approx(s));
        }
    }
    else {
        REQUIRE(v.x == s);
        REQUIRE(v.r == s);
        REQUIRE(v.y == s);
        REQUIRE(v.g == s);
        if constexpr (V::Size > 2) {
            REQUIRE(v.z == s);
            REQUIRE(v.b == s);
        }
        if constexpr (V::Size > 3) {
            REQUIRE(v.w == s);
            REQUIRE(v.a == s);
        }
    }
}

TEMPLATE_PRODUCT_TEST_CASE("Vec tests", "[vec]", (Vec2, Vec3, Vec4), (float, double, int, uint32_t)) {
    // Ctor tests
    TestType vec0;
    TestType vec1{1};
    TestType vec2{2};
    TestType vec3;
    if constexpr (vec0.Size == 2)
        vec3 = {1, 2};
    else if constexpr (vec0.Size == 3)
        vec3 = {1, 2, 3};
    else if constexpr (vec0.Size == 4)
        vec3 = {1, 2, 3, 4};
    else
        static_assert(vec0.Size < 5, "Should not reach here");

    SECTION("Ctor tests") {
        check_vec_components(vec0, 0);
        check_vec_components(vec1, 1);
    }

    SECTION("Operator tests") {
        auto vec4 = vec1 + vec2;
        PROPER_EQUAL(vec4.x, 3, vec4);
        PROPER_EQUAL(vec4.y, 3, vec4);

        vec4 = vec2 - vec1;
        PROPER_EQUAL(vec4.x, 1, vec4);
        PROPER_EQUAL(vec4.y, 1, vec4);

        vec4 = vec2 * vec3;
        PROPER_EQUAL(vec4.x, 2, vec4);
        PROPER_EQUAL(vec4.y, 4, vec4);

        vec4 = vec3 / vec2;
        if constexpr (std::is_floating_point_v<typename decltype(vec4)::Scalar>) {
            REQUIRE(vec4.x == Approx(0.5));
            REQUIRE(vec4.y == Approx(1));
        }
        else {
            REQUIRE(vec4.x == 0);
            REQUIRE(vec4.y == 1);
        }
    }
}
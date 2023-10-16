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
    REQUIRE(v16.vectorized == true);
    REQUIRE(sizeof(v16) == 16);

    Vec<float, 3> v17{1, 2, 3};
    Vec<float, 3> v18{2, 3, 4};
    REQUIRE(v17.dot(v18) == Approx(20));

    auto v19 = v17.shuffle<1, 2, 0>();
    REQUIRE(v19.x == Approx(2));
    auto v20 = v17.cross(v18);
    REQUIRE(v20.x == Approx(-1));

    Vec<double, 2> v21{1, 2};
    auto v22 = v21.shuffle<1, 0>();
    REQUIRE(v22.x == 2);
}

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
    TestType vec3{3};

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
        PROPER_EQUAL(vec4.x, 6, vec4);
        PROPER_EQUAL(vec4.y, 6, vec4);

        vec4 = vec3 / vec2;
        if constexpr (std::is_floating_point_v<typename decltype(vec4)::Scalar>) {
            REQUIRE(vec4.x == Approx(1.5));
            REQUIRE(vec4.y == Approx(1.5));
        }
        else {
            REQUIRE(vec4.x == 1);
            REQUIRE(vec4.y == 1);
        }
    }
}
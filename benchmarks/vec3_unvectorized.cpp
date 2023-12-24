#include <benchmark/benchmark.h>

#define YAVL_DISABLE_VECTORIZATION
#include <yavl/yavl.h>

using namespace yavl;

static void BM_Vec3fAddition(benchmark::State& state) {
    Vec3f a{1}, b{2}, c;
    for (auto _ : state) {
        static_for<1000>([&](const auto i) {
            c = a + b;
            benchmark::DoNotOptimize(c);
        });
    }
}

BENCHMARK(BM_Vec3fAddition);

static void BM_Vec3fSubtraction(benchmark::State& state) {
    Vec3f a{2}, b{1}, c;
    for (auto _ : state) {
        static_for<1000>([&](const auto i) {
            c = a - b;
            benchmark::DoNotOptimize(c);
        });
    }
}

BENCHMARK(BM_Vec3fSubtraction);

static void BM_Vec3fDivision(benchmark::State& state) {
    Vec3f a{4}, b{2}, c;
    for (auto _ : state) {
        static_for<1000>([&](const auto i) {
            c = a / b;
            benchmark::DoNotOptimize(c);
        }); 
    }
}

BENCHMARK(BM_Vec3fDivision);

static void BM_Vec3fShuffle(benchmark::State& state) {
    Vec3f a{1, 2, 3}, c;
    for (auto _ : state) {
        static_for<1000>([&](const auto i) {
            c = a.template shuffle<2, 1, 0>();
            benchmark::DoNotOptimize(c);
        });
    }
}

BENCHMARK(BM_Vec3fShuffle);

static void BM_Vec3fDot(benchmark::State& state) {
    Vec3f a{1}, b{2}, c;
    for (auto _ : state) {
        static_for<1000>([&](const auto i) {
            c = a.dot(b);
            benchmark::DoNotOptimize(c);
        });
    }
}

BENCHMARK(BM_Vec3fDot);

static void BM_Vec3fSqrt(benchmark::State& state) {
    Vec3f a{1}, c;
    for (auto _ : state) {
        static_for<1000>([&](const auto i) {
            c = a.sqrt();
            benchmark::DoNotOptimize(c);
        });
    }
}

BENCHMARK(BM_Vec3fSqrt);

static void BM_Vec3fRsqrt(benchmark::State& state) {
    Vec3f a{2}, c;
    for (auto _ : state) {
        static_for<1000>([&](const auto i) {
            c = a.rsqrt();
            benchmark::DoNotOptimize(c);
        });
    }
}

BENCHMARK(BM_Vec3fRsqrt);

BENCHMARK_MAIN();
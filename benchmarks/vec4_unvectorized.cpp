#include <benchmark/benchmark.h>

#define YAVL_DISABLE_VECTORIZATION
#include <yavl/yavl.h>

using namespace yavl;

static void BM_Vec4fAddition(benchmark::State& state) {
    Vec4f a{1}, b{2}, c;
    for (auto _ : state) {
        static_for<1000>([&](const auto i) {
            c = a + b;
            benchmark::DoNotOptimize(c);
        });
    }
}

BENCHMARK(BM_Vec4fAddition);

static void BM_Vec4fSubtraction(benchmark::State& state) {
    Vec4f a{2}, b{1}, c;
    for (auto _ : state) {
        static_for<1000>([&](const auto i) {
            c = a - b;
            benchmark::DoNotOptimize(c);
        });
    }
}

BENCHMARK(BM_Vec4fSubtraction);

static void BM_Vec4fDivision(benchmark::State& state) {
    Vec4f a{4}, b{2}, c;
    for (auto _ : state) {
        static_for<1000>([&](const auto i) {
            c = a / b;
            benchmark::DoNotOptimize(c);
        }); 
    }
}

BENCHMARK(BM_Vec4fDivision);

static void BM_Vec4fShuffle(benchmark::State& state) {
    Vec4f a{1, 2, 3, 4}, c;
    for (auto _ : state) {
        static_for<1000>([&](const auto i) {
            c = a.template shuffle<3, 2, 1, 0>();
            benchmark::DoNotOptimize(c);
        });
    }
}

BENCHMARK(BM_Vec4fShuffle);

static void BM_Vec4fDot(benchmark::State& state) {
    Vec4f a{1}, b{2}, c;
    for (auto _ : state) {
        static_for<1000>([&](const auto i) {
            c = a.dot(b);
            benchmark::DoNotOptimize(c);
        });
    }
}

BENCHMARK(BM_Vec4fDot);

static void BM_Vec4fSqrt(benchmark::State& state) {
    Vec4f a{1}, c;
    for (auto _ : state) {
        static_for<1000>([&](const auto i) {
            c = a.sqrt();
            benchmark::DoNotOptimize(c);
        });
    }
}

BENCHMARK(BM_Vec4fSqrt);

static void BM_Vec4fRsqrt(benchmark::State& state) {
    Vec4f a{2}, c;
    for (auto _ : state) {
        static_for<1000>([&](const auto i) {
            c = a.rsqrt();
            benchmark::DoNotOptimize(c);
        });
    }
}

BENCHMARK(BM_Vec4fRsqrt);

BENCHMARK_MAIN();
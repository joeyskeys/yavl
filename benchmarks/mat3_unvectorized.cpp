#include <benchmark/benchmark.h>

#define YAVL_DISABLE_VECTORIZATION
#include <yavl/yavl.h>

using namespace yavl;

static void BM_Mat3fMulVec(benchmark::State& state) {
    Mat3f a{2};
    Vec3f b{2}, c;
    for (auto _ : state) {
        static_for<1000>([&](const auto i) {
            c = a * b;
            benchmark::DoNotOptimize(c);
        });
    }
}

BENCHMARK(BM_Mat3fMulVec);

static void BM_Mat3fMulMat(benchmark::State& state) {
    Mat3f a{1}, b{2}, c;
    for (auto _ : state) {
        static_for<1000>([&](const auto i) {
            c = a * b;
            benchmark::DoNotOptimize(c);
        });
    }
}

BENCHMARK(BM_Mat3fMulMat);

BENCHMARK_MAIN();
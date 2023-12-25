#include <benchmark/benchmark.h>

#define YAVL_DISABLE_VECTORIZATION
#include <yavl/yavl.h>

using namespace yavl;

static void BM_Mat4fMulVec(benchmark::State& state) {
    Mat4f a{2};
    Vec4f b{2}, c;
    for (auto _ : state) {
        static_for<1000>([&](const auto i) {
            c = a * b;
            benchmark::DoNotOptimize(c);
        });
    }
}

BENCHMARK(BM_Mat4fMulVec);

static void BM_Mat4fMulMat(benchmark::State& state) {
    Mat4f a{1}, b{2}, c;
    for (auto _ : state) {
        static_for<1000>([&](const auto i) {
            c = a * b;
            benchmark::DoNotOptimize(c);
        });
    }
}

BENCHMARK(BM_Mat4fMulMat);

BENCHMARK_MAIN();
#include <benchmark/benchmark.h>

//#define YAVL_FORCE_SSE_MAT
#include <yavl/yavl.h>

using namespace yavl;

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
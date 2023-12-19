#include <benchmark/benchmark.h>

#define YAVL_DISABLE_VECTORIZATION
#define YAVL_FORCE_SSE_MAT
#include <yavl/yavl.h>

using namespace yavl;

static void BM_Mat4fMulMat(benchmark::State& state) {
    Mat4f a{1}, b{2}, c;
    for (auto _ : state) {
        benchmark::DoNotOptimize(c = a * b);
    }
}

BENCHMARK(BM_Mat4fMulMat);

BENCHMARK_MAIN();
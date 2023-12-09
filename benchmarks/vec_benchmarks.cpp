#include <benchmark/benchmark.h>

#include <yavl/yavl.h>

using namespace yavl;

static void BM_Vec4fAddition(benchmark::State& state) {
    Vec4f a{1}, b{2}, c;
    for (auto _ : state) {
        benchmark::DoNotOptimize(c = a + b);
    }
}

BENCHMARK(BM_Vec4fAddition);

static void BM_Vec4fSubtraction(benchmark::State& state) {
    Vec4f a{2}, b{1}, c;
    for (auto _ : state) {
        benchmark::DoNotOptimize(c = a - b);
    }
}

BENCHMARK(BM_Vec4fSubtraction);

BENCHMARK_MAIN();
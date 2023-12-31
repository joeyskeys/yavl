#include <benchmark/benchmark.h>

#define YAVL_DISABLE_VECTORIZATION
#include <yavl/yavl.h>

using namespace yavl;

static void BM_Pcg32Generate8(benchmark::State& state) {
    pcg32x<8> rng;
    std::array<float, 8> result;
    for (auto _ : state) {
        static_for<1000>([&](const auto i) {
            rng.next_floats(result);
            benchmark::DoNotOptimize(result);
        });
    }
}

BENCHMARK(BM_Pcg32Generate8);

BENCHMARK_MAIN();
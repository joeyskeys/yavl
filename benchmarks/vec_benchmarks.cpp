#include <cstdio>

#include <yavl/yavl.h>

#include "utils.h"

using namespace yavl;

int main() {
    Vec4f a{1};
    Vec4f b{2};
    auto st = rdtsc();
    auto c = a * b;
    auto dur = static_cast<double>(rdtsc() - st);
    printf("ret: %f, cc: %.2f\n", c[0], dur);

    _Vec4f d{1};
    _Vec4f e{2};
    st = rdtsc();
    auto f = d * e;
    dur = static_cast<double>(rdtsc() - st);
    printf("ret: %f, cc: %.2f\n", c[0], dur);
    return 0;
}
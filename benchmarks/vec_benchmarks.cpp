#include <cstdio>

#include <yavl/yavl.h>

#include "utils.h"

using namespace yavl;

int main() {
    Vec4f a{1};
    Vec4f b{2};
    auto c = a + b;
    printf("ret: %f", c[0]);
    return 0;
}
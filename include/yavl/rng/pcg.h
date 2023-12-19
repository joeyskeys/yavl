#pragma once

// Most copied from:
// https://github.com/wjakob/pcg32/blob/master/pcg32.h

#define PCG32_DEFAULT_STATE  0x853c49e6748fea9bULL
#define PCG32_DEFAULT_STREAM 0xda3e39cb94b95bdbULL
#define PCG32_MULT           0x5851f42d4c957f2dULL

#include <inttypes.h>
#include <cmath>
#include <cassert>
#include <algorithm>

// PCG32 Pseudorandom number generator
struct pcg32 {
    uint64_t state;     // RNG state. All values are possible.
    uint64_t inc;       // Controls which RNG sequence (stream) is selected.

    
};
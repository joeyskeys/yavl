# yavl

## Introduction

yavl is a small vectorization library written for doing experiments and benchmarks to compare the efficiency of manual vectorized code and compiler auto-vectorized code.

Usage in a production project is **not recommended**.

## Features

Or topics that these experiments are about, either done or to be done:

- [x] Vector3/4 calculations
- [x] Matrix3x3/4x4 calculations
- [x] Pseudorandom number generation(PCG32)
- [ ] String manipulation
- [ ] ISPC version of previous topics

Results and conclusion will be updated if more work is done.

## Results

### Benchmark results

All the following results(measured in nanoseconds) are aquired on my Linux PC with a AMD CPU.

#### Vectors:

| Op(1000 iterations) | vec3 | vec3_unvectorized | vec4 | vec4_unvectorized |
|:-------------------:|:----:|:-----------------:|:----:|:-----------------:|
| addition            | 503  | 1665              | 511  | 1299              |
| subtraction         | 505  | 1691              | 520  | 1316              |
| division            | 722  | 1768              | 736  | 1357              |
| shuffle             | 517  | 1388              | 525  | 2711              |
| dot                 | 1035 | 1776              | 1051 | 1606              |
| sqrt                | 1024 | 3160              | 1043 | 4621              |
| rsqrt               | 643  | 4650              | 653  | 7850              |

#### Matrix:
| Op(1000 iterations) | mat3 | mat3_unvectorized | mat4 | mat4_unvectorized |
|:-------------------:|:----:|:-----------------:|:----:|:-----------------:|
| multiplication(vec) | 7482 | 2656              | 1723 | 2031              |
| multiplication(mat) | 4377 | 3164              | 3086 | 2500              |

#### Random number generation:
| Op(1000 iterations) | pcg32 | pcg32_unvectorized |
|:-------------------:|:-----:|:------------------:|
| generate 8 numbers  | 5283  | 9566               |

### Assembly code comparison:

TODO...

## Conclusions

1. Manual intrinsic vecterization do help in scenarios where we need to do homogenerous operations to a data pack, especially when we apply specialized hardware implemented functions like sqrt&rsqrt(with possibly one or two Newton-Raphson iterations to help improve precision).

2. Write a function applying each step to a data pack is better than apply function to each element of a data pack, just like the idea of SoA(Structure of Arrays) from data oriented programming. This could help compiler to better auto-vectorized your code even not using intrinsics manually.
   
3. Your intrinsic implementation can be slower than plain ordinary one.
   - Check the results for matrices. Vectorized matrix3x3 is way slower than unvecotrized implementation. I use a __m256 and a __m128 for three columns(to save some memory space) when AVX available. Turns out to be a performance penalty. 

   - Each intrinsic function comes with a different cost. If you're not familiar enough with them(just like me), your implementation could be slower than the compiler auto-vectorized one. Check the matrices muliplication result.
  
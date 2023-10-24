#pragma once

/*
 * Most of the following code is copied from enoki
 * https://github.com/mitsuba-renderer/enoki
 */

#if defined(__x86_64__) || defined(_M_X64)
#   define ARCH_X86_64 1
#endif

#if (defined(__i386__) || defined(_M_IX86)) && !defined(ARCH_X86_64)
#   define ARCH_X86_32 1
#endif

#if defined(__aarch64__)
#   define ARCH_ARM_64 1
#elif defined(__arm__)
#   define ARCH_ARM_32 1
#endif

#if !defined(YAVL_DISABLE_VECTORIZATION)
// When I wrote down these codes, I finally realized that
// my 5950x doesn't support AVX-512 since my linter greyed
// out these macros...
#   if defined(__AVX512F__)
#       define YAVL_X86_AVX512F 1
#   endif
#   if defined(__AVX512CD__)
#       define YAVL_X86_AVX512CD 1
#   endif
#   if defined(__AVX512DQ__)
#       define YAVL_X86_AVX512DQ 1
#   endif
#   if defined(__AVX512VL__)
#       define YAVL_X86_AVX512VL 1
#   endif
#   if defined(__AVX512BW__)
#       define YAVL_X86_AVX512BW 1
#   endif
#   if defined(__AVX512PF__)
#       define YAVL_X86_AVX512PF 1
#   endif
#   if defined(__AVX512ER__)
#       define YAVL_X86_AVX512ER 1
#   endif
#   if defined(__AVX512VBMI__)
#       define YAVL_X86_AVX512VBMI 1
#   endif
#   if defined(__AVX512VPOPCNTDQ__)
#       define YAVL_X86_AVX512VPOPCNTDQ 1
#   endif
#   if defined(__AVX2__)
#       define YAVL_X86_AVX2 1
#   endif
#   if defined(__FMA__)
#       define YAVL_X86_FMA 1
#   endif
#   if defined(__F16C__)
#       define YAVL_X86_F16C 1
#   endif
#   if defined(__AVX__)
#       define YAVL_X86_AVX 1
#   endif
#   if defined(__SSE4_2__)
#       define YAVL_X86_SSE42 1
#   endif
#   if defined(__ARM_NEON)
#       define YAVL_ARM_NEON
#   endif
#   if defined(__ARM_FEATURE_FMA)
#       define YAVL_ARM_FMA
#   endif
#endif

#if defined(ARCH_X86_64) || defined(ARCH_X86_32)
#   include <immintrin.h>
#endif

#if defined(YAML_ARM_NEON)
#   include <arm_neon.h>
#endif

#if defined(_MSC_VER)
#   include <intrin.h>
#endif

#if defined(YAVL_X86_AVX512F)
    static constexpr bool has_avx512f = true;
#else
    static constexpr bool has_avx512f = false;
#endif

#if defined(YAVL_X86_AVX512CD)
    static constexpr bool has_avx512cd = true;
#else
    static constexpr bool has_avx512cd = false;
#endif

#if defined(YAVL_X86_AVX512DQ)
    static constexpr bool has_avx512dq = true;
#else
    static constexpr bool has_avx512dq = false;
#endif

#if defined(YAVL_X86_AVX512VL)
    static constexpr bool has_avx512vl = true;
#else
    static constexpr bool has_avx512vl = false;
#endif

#if defined(YAVL_X86_AVX512BW)
    static constexpr bool has_avx512bw = true;
#else
    static constexpr bool has_avx512bw = false;
#endif

#if defined(YAVL_X86_AVX512PF)
    static constexpr bool has_avx512pf = true;
#else
    static constexpr bool has_avx512pf = false;
#endif

#if defined(YAVL_X86_AVX512ER)
    static constexpr bool has_avx512er = true;
#else
    static constexpr bool has_avx512er = false;
#endif

#if defined(YAVL_X86_AVX512VBMI)
    static constexpr bool has_avx512vbmi = true;
#else
    static constexpr bool has_avx512vbmi = false;
#endif

#if defined(YAVL_X86_AVX512VPOPCNTDQ)
    static constexpr bool has_avx512vpopcntdq = true;
#else
    static constexpr bool has_avx512vpopcntdq = false;
#endif

#if defined(YAVL_X86_AVX2)
    static constexpr bool has_avx2 = true;
#else
    static constexpr bool has_avx2 = false;
#endif

#if defined(YAVL_X86_FMA)
    static constexpr bool has_fma = true;
#else
    static constexpr bool has_fma = false;
#endif

#if defined(YAVL_X86_F16C)
    static constexpr bool has_f16c = true;
#else
    static constexpr bool has_f16c = false;
#endif

#if defined(YAVL_X86_AVX)
    static constexpr bool has_avx = true;
#else
    static constexpr bool has_avx = false;
#endif

#if defined(YAVL_X86_SSE42)
    static constexpr bool has_sse42 = true;
#else
    static constexpr bool has_sse42 = false;
#endif

#if defined(YAVL_ARM_NEON)
    static constexpr bool has_neon = true;
#else
    static constexpr bool has_neon = false;
#endif
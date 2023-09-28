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

#if !defined(YAVML_DISABLE_VECTORIZATION)
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
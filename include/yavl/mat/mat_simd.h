#pragma once

// Common macros
#define YAVL_MAT_ALIAS_VECTORIZED(TYPE, N, INTRIN_N)                    \
    YAVL_TYPE_ALIAS(TYPE, N, INTIRN_N)                                  \
    static constexpr bool vectorized = true;


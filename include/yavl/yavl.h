#pragma once

#include <yavl/platform.h>
#include <yavl/utils.h>

#include <yavl/vec/vec.h>
#if !defined(YAVL_DISABLE_VECTORIZATION)
    #include <yavl/vec/vec_simd.h>
#endif

#include <yavl/mat/mat.h>
#if !defined(YAVL_DISABLE_VECTORIZATION)
    #include <yavl/mat/mat_simd.h>
#endif
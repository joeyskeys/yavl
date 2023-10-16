#pragma once

#include <yavl/platform.h>
#include <yavl/vec.h>

#if defined(YAVL_X86_SSE42)
#  include <yavl/vec_sse42.h>
#endif

namespace yavl
{

template <typename T>
using Vec2 = Vec<T, 2>;

template <typename T>
using _Vec2 = Vec<T, 2, false>;

template <typename T>
using Vec3 = Vec<T, 3>;

template <typename T>
using _Vec3 = Vec<T, 3, false>;

template <typename T>
using Vec4 = Vec<T, 4>;

template <typename T>
using _Vec4 = Vec<T, 4, false>;

template <typename T>
using Vec2f = Vec2<float>;

template <typename T>
using Vec2d = Vec2<double>;

template <typename T>
using Vec2i = Vec2<int>;

template <typename T>
using Vec2u = Vec2<uint32_t>;

template <typename T>
using Vec3f = Vec3<float>;

template <typename T>
using Vec3d = Vec3<double>;

template <typename T>
using Vec3i = Vec3<int>;

template <typename T>
using Vec3u = Vec3<uint32_t>;

template <typename T>
using Vec4f = Vec4<float>;

template <typename T>
using Vec4d = Vec4<double>;

template <typename T>
using Vec4i = Vec4<int>;

template <typename T>
using Vec4u = Vec4<uint32_t>;

template <typename T>
using _Vec2f = _Vec2<float>;

template <typename T>
using _Vec2d = _Vec2<double>;

template <typename T>
using _Vec2i = _Vec2<int>;

template <typename T>
using _Vec2u = _Vec2<uint32_t>;

template <typename T>
using _Vec3f = _Vec3<float>;

template <typename T>
using _Vec3d = _Vec3<double>;

template <typename T>
using _Vec3i = _Vec3<int>;

template <typename T>
using _Vec3u = _Vec3<uint32_t>;

template <typename T>
using _Vec4f = _Vec4<float>;

template <typename T>
using _Vec4d = _Vec4<double>;

template <typename T>
using _Vec4i = _Vec4<int>;

template <typename T>
using _Vec4u = _Vec4<uint32_t>;
}
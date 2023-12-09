cmake_minimum_required(VERSION 3.20)

project(yavl LANGUAGES CXX)

cmake_policy(SET CMP0074 NEW)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "set build type")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)

set(CMAKE_CXX_FLAGS "-march=native" ${CMAKE_CXX_FLAGS})

include_directories(include)

add_subdirectory(tests)

find_package(benchmark)

if (benchmark_FOUND)
    add_subdirectory(benchmarks)
endif()
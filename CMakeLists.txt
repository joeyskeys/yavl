cmake_minimum_required(VERSION 3.20)

project(yavl LANGUAGES CXX)

cmake_policy(SET CMP0074 NEW)
set(CMAKE_CXX_STANDARD 20)

set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "set build type")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)

set(CMAKE_CXX_FLAGS "-march=native" ${CMAKE_CXX_FLAGS})

add_library(yavl INTERFACE)
target_include_directories(yavl
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

find_package(Catch2)
if (Catch2_FOUND)
    add_subdirectory(tests)
endif()

find_package(benchmark)
if (benchmark_FOUND)
    add_subdirectory(benchmarks)
endif()

set(YAVL_VERSION "0.1")

include(CMakePackageConfigHelpers)
install(DIRECTORY include/ DESTINATION include)
install(TARGETS yavl EXPORT yavl-targets)
install(EXPORT yavl-targets
    FILE yavl-config.cmake
    DESTINATION cmake/)

write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/yavl-config-version.cmake"
    COMPATIBILITY SameMajorVersion
    VERSION "${YAVL_VERSION}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/yavl-config-version.cmake" DESTINATION cmake/)
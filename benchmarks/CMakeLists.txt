add_executable(vec_benchmarks vec_benchmarks.cpp)
target_link_libraries(vec_benchmarks benchmark::benchmark)

add_executable(mat_benchmarks mat_benchmarks.cpp)
target_link_libraries(mat_benchmarks benchmark::benchmark)
add_executable(vec_vectorized vec_vectorized.cpp)
target_link_libraries(vec_vectorized benchmark::benchmark)

add_executable(vec_unvectorized vec_unvectorized.cpp)
target_link_libraries(vec_unvectorized benchmark::benchmark)

add_executable(mat_vectorized mat_vectorized.cpp)
target_link_libraries(mat_vectorized benchmark::benchmark)

add_executable(mat_unvectorized mat_unvectorized.cpp)
target_link_libraries(mat_unvectorized benchmark::benchmark)

add_executable(rng_vectorized rng_vectorized.cpp)
target_link_libraries(rng_vectorized benchmark::benchmark)

add_executable(rng_unvectorized rng_unvectorized.cpp)
target_link_libraries(rng_unvectorized benchmark::benchmark)
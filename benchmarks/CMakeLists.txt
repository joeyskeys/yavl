add_executable(vec3_vectorized vec3_vectorized.cpp)
target_link_libraries(vec3_vectorized benchmark::benchmark)

add_executable(vec3_unvectorized vec3_unvectorized.cpp)
target_link_libraries(vec3_unvectorized benchmark::benchmark)

add_executable(vec4_vectorized vec4_vectorized.cpp)
target_link_libraries(vec4_vectorized benchmark::benchmark)

add_executable(vec4_unvectorized vec4_unvectorized.cpp)
target_link_libraries(vec4_unvectorized benchmark::benchmark)

add_executable(mat3_vectorized mat3_vectorized.cpp)
target_link_libraries(mat3_vectorized benchmark::benchmark)

add_executable(mat3_unvectorized mat3_unvectorized.cpp)
target_link_libraries(mat3_unvectorized benchmark::benchmark)

add_executable(mat4_vectorized mat4_vectorized.cpp)
target_link_libraries(mat4_vectorized benchmark::benchmark)

add_executable(mat4_unvectorized mat4_unvectorized.cpp)
target_link_libraries(mat4_unvectorized benchmark::benchmark)

add_executable(rng_vectorized rng_vectorized.cpp)
target_link_libraries(rng_vectorized benchmark::benchmark)

add_executable(rng_unvectorized rng_unvectorized.cpp)
target_link_libraries(rng_unvectorized benchmark::benchmark)
add_executable(vec_tests vec_tests.cpp)
target_link_libraries(vec_tests PRIVATE Catch2::Catch2WithMain)

add_executable(mat_tests mat_tests.cpp)
target_link_libraries(mat_tests PRIVATE Catch2::Catch2WithMain)

add_executable(util_tests util_tests.cpp)
target_link_libraries(util_tests PRIVATE Catch2::Catch2WithMain)
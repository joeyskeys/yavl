find_package(Catch2 REQUIRED)

add_executable(intrinsic_tests intrinsic_tests.cpp)
target_link_libraries(intrinsic_tests PRIVATE Catch2::Catch2WithMain)

add_executable(vector_tests vec_tests.cpp)
target_link_libraries(vector_tests PRIVATE Catch2::Catch2WithMain)
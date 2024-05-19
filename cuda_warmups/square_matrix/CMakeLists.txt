cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example-app)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(squaring_matrix squaring_matrix.cpp)
target_link_libraries(squaring_matrix "${TORCH_LIBRARIES}")
set_property(TARGET squaring_matrix PROPERTY CXX_STANDARD 17)
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(squaring_matrix LANGUAGES CXX CUDA)

find_package(Python REQUIRED COMPONENTS Development)
find_package(Torch REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 89)

add_executable(squaring_matrix squaring_matrix.cu)
target_compile_features(squaring_matrix PRIVATE cxx_std_11)
target_link_libraries(squaring_matrix PRIVATE ${TORCH_LIBRARIES} Python::Python)
cmake_minimum_required(VERSION 3.18)

cmake_policy(SET CMP0104 NEW)

project(execution_conf LANGUAGES CXX C)

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda")

include(protect_nvcc_flags)
include(protect_pthread_flag)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(execution_conf execution_conf.cu)
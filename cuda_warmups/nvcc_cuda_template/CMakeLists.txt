cmake_minimum_required(VERSION 3.18)

cmake_policy(SET CMP0104 NEW)

project(easy_intro LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(easy_intro easy_cuda_intro.cu)
cmake_minimum_required(VERSION 3.18)

message("Using Cmake version ${CMAKE_VERSION}")
message("Using cmake policy CMP0104 NEW")

cmake_policy(SET CMP0104 NEW)

project(addvector LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_EXTENSIONS OFF)

message("Current Source Directory is    : ${CMAKE_CURRENT_SOURCE_DIR}")

list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cuda")

include(preventBuildInSource)

#init build type: Release, Debug,... Unable to find
# include(initBuildType)

#always export compile commands database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

enable_language(CUDA)

message("Using CUDAToolkit Macros")
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# require the below extra flags for the compile flags to work
include(protect_nvcc_flags)
include(protect_pthread_flag)

#--------------------------------------------------------------#
# Configuration Summary
#--------------------------------------------------------------#

# message("{PROJECT_NAME} build config    : ")
# message("{PROJECT_NAME} version         : ${PROJECT_VERSION} ")

message(" CMAKE Version                 : ${CMAKE_VERSION}")

message("  CUDA COMPILER ID             : ${CMAKE_CUDA_COMPILER_ID}")
message("  CUDA COMPILER Version        : ${CMAKE_CUDA_COMPILER_VERSION}")

message("  CPP Compiler                 : ${CMAKE_CXX_COMPILER}")

message("  CXX   STANDARD               : ${CMAKE_CXX_STANDARD}")
message("  CUDA CXX STANDARD            : ${CMAKE_CUDA_STANDARD}")
message("  CUDA COMPILER                : ${CMAKE_CUDA_COMPILER}")
message("  CUDA COMPILER exec           : ${CUDA_NVCC_EXECUTABLE}")
message("  CUDA_NVCC_FLAGS              : ${CUDA_NVCC_FLAGS}")

message("#########   End Of Config   ##########")

message("Set CUDAFLAGS env Variable     : --expt-extended-lambda")
message("Environment Vars               : -DCMAKE_CUDA_ARCHITECTURES=89 ..")

add_executable(addvector addvector.cu)
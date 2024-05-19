cmake_minimum_required(VERSION 3.12)
project(cuda_example LANGUAGES CUDA)

find_package(CUDA QUIET REQUIRED)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -arch=sm_89
)

cuda_add_executable(cuda_example cuda_example.cu)

target_link_libraries(cuda_example ${CUDA_LIBRARIES})

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(torchtest)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(torchtest working_torch.cpp)
target_link_libraries(torchtest "${TORCH_LIBRARIES}")
set_property(TARGET torchtest PROPERTY CXX_STANDARD 17)
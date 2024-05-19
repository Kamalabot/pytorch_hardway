// #include <c10/cuda/CUDAException.h>
#include <torch/types.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

using namespace at;

__global__
void square_matrix_kernel(const float* matrix, float* result, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int idx = row * width + col;
        result[idx] = matrix[idx] * matrix[idx];
    }
}

torch::Tensor square_matrix(torch::Tensor matrix) {
    const auto height = matrix.size(0);
    const auto width = matrix.size(1);

    auto result = torch::empty_like(matrix);

    dim3 threads_per_block(16, 16);
    dim3 number_of_blocks((width + threads_per_block.x - 1) / threads_per_block.x,
                          (height + threads_per_block.y - 1) / threads_per_block.y);

    square_matrix_kernel<<<number_of_blocks, threads_per_block>>>(
        matrix.data_ptr<float>(), result.data_ptr<float>(), width, height);

    return result;
}

int main(){
    torch::Tensor matrix = torch::Tensor([[1., 2., 3.], [4., 5., 6.]],);
    // torch::Tensor sqr_matrix = torch::Tensor square_matrix(torch::Tensor matrix);
    auto sqr_matrix = torch::Tensor square_matrix(torch::Tensor matrix);
    cudaDeviceSynchronize();
    std::cout << sqr_matrix;
    return 0;
}
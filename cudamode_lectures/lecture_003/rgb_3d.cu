#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/extension.h>
#include <stdio.h>
#include <c10/cuda/CUDAException.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA Tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.device().is_contiguous(), #x " must be a CONTIGUOUS")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}

__global__ void rgb_to_grayscale_kernel(unsigned char* x, unsigned char* out, int w, int h) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c < w && r < h){
        int i = r * w + c;
        int n = h * w;
        out[i] = 0.2989 * x[i] + 0.5870 * x[i + n] + 0.1140 * x[i + 2*n];
    }
}

torch::Tensor rgb_to_gs(torch::Tensor input) {
    CHECK_INPUT(input);
    int h = input.size(1);
    int w = input.size(2);
    printf("h * w: %d * %d\n", h, w);
    auto output = torch::empty({h, w}, input.options());
    dim3 tpb(16, 16);
    dim3 blocks(cdiv(w, tpb.w), cdi(h, tpb.y));
    rgb_to_grayscale_kernel<<<blocks, threads>>>(
        input.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(), w, h
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
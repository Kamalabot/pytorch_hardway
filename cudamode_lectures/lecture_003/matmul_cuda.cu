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

__global__ void matmul_k(float* m, float* n, float* out, int h, int w, int k){
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r >= h || c >= w) return;
    float o = 0;
    for (int i = 0; i < k; ++i) o += m[r * k + i] * n[i * w + c];
    out[r * w + c] = o;
}

torch::Tensor matmul(torch::Tensor m, torch::Tensor n){
    int h = m.size(0);
    int w = n.size(1);
    int k = m.size(1);
    TORCH_CHECK(k==n.size(0), "Size Mismatch");
    auto output = torch::zeros({h, w}, m.options());

    dim3 tpb(16, 16);
    dim3 blocks(cdiv(w, tpb.x), cdiv(h, tpb.y));
    matmul_k<<<blocks, tpb>>>(m.data_ptr<float>(), n.data_ptr<float>(), output.data_ptr<float>(), h, w, k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
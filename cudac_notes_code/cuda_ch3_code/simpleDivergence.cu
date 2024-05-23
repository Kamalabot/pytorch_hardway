#include <iostream>
#include <sys/time.h>
#include <cuda_runtime.h>

using namespace std;

inline double seconds()
{
    struct timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

__global__ void mathKernel1(float *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    float ia, ib;
    ia = ib = 0.0f;
    // following creates a warp divergence
    if (tid % 2 == 0){
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}

__global__ void mathKernel2(float *c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    
    if((tid / warpSize) % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}

__global__ void warmingUp(float *c){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    
    if((tid / warpSize) % 2 == 0)
    {
        ia = 100.0f;
    }
    else
    {
        ib = 200.0f;
    }
    c[tid] = ia + ib;
}

int main() {
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    cout << "Using Device: " << deviceProp.name;

    int size = 64;
    int blocksize = 64;

    cout << "Data Size: " << size;

    dim3 block(blocksize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);

    cout << "Threads: " << block.x << "Blocks: " << grid.x << endl;

    float *d_C;

    size_t nBytes = size * sizeof(float);
    cudaMalloc((float **)&d_C, nBytes);    

    size_t iSt, iEl;

    cudaDeviceSynchronize();

    iSt = seconds(); 
    warmingUp<<<grid, block>>>(d_C);

    cudaDeviceSynchronize();

    iEl = seconds() - iSt;

    iSt = seconds(); 
    mathKernel1<<<grid, block>>>(d_C);

    cudaDeviceSynchronize();

    iEl = seconds() - iSt;

    cout << "MathKernel1<<<" << grid.x << ", " << block.x << ">>>" << "time elapsed " << iEl << endl;
}

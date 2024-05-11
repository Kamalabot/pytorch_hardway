#include <iostream>
#include <math.h>

// function to add element of two arrays
__global__  // global functions are called kernels, and usually device codes
void add(int n, float *x, float *y)
{
    int index = threadIdx.x;
    int stride = blockDim.x;
    for (int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1<<20;  // 1M elements ? 
    // This is equivalent to N = pow(2, 20); where pow() is a standard C++ library function for computing powers.
    float *x, *y;
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    for (int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    add<<<1, 256>>>(N, x, y);  // here the kernel is called
    // 1st parameter is number of thread blocks, and 
    // 2nd is num of threads per block
    std::cout << y[100];
    
    cudaDeviceSynchronize();  // here the data is synchronized

    // check for error, and values should be 3.0f
    float maxError = 0.0f;
    for (int i = 0; i < N; i++){
        // std::cout << y[i];
        maxError = fmax(maxError, fabs(y[i] - 2.0f)); 
    }
    std::cout << "Max Error: " << maxError << std::endl;

    // free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}
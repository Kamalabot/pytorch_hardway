#include <stdio.h>
// There is no output even though the syntax is correct. Fundamental mistake is there

__global__ void exec_conf(int x_f)
{
    printf("Blocks used are %d, %d, %d", blockDim.x, blockDim.y, blockDim.z);
    printf("Threads used are %d, %d, %d", threadIdx.x, threadIdx.y, threadIdx.z);
}

int main(){
    // dim3 threads(12, 12, 12);
    // dim3 blocks(8, 8, 8);
    int threads = 100;
    int blocks = 5;
    int fg = 12;
    exec_conf<<<blocks, threads>>>(fg);
    cudaDeviceSychronize();
    return 0;
}
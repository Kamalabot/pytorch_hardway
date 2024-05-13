#include <stdio.h>
// There is no output even though the syntax is correct. Fundamental mistake is there
// __global__ void exec_conf(int x_f)
// {
    // printf("Blocks used are %d, %d, %d", blockDim.x, blockDim.y, blockDim.z);
    // printf("Threads used are %d, %d, %d", threadIdx.x, threadIdx.y, threadIdx.z);
// }

// int main(){
    // dim3 threads(12, 12, 12);
    // dim3 blocks(8, 8, 8);
    // exec_conf<<<blocks, threads>>>(12);
    // cudaDeviceSychronize();
    // return 0;
// }


#include <stdio.h>

__global__ void strideCHK(int n)
{
    int index = threadIdx.x;
    int stride = blockDim.x;

    printf("threadIndex is %d and BlockDim.x is %d\n", index, stride);
    printf("threadIdx.x is %d, threadIdx.y is %d and threadIdx.z is %d\n",
            threadIdx.x, threadIdx.y, threadIdx.z);
    printf("blockDim.x is %d, blockDim.y is %d and blockDim.z is %d\n",
            blockDim.x, blockDim.y, blockDim.z);

    for(int i = index; i < n; i += stride){
        printf("Stride is %d\n", stride);
        // stride above prints 15, as the blockDim.x is 15
        printf("I value is %d\n", i);
        // i value increments only with one
    }
}

int main(){
    dim3 blocks(16, 16, 16);
    dim3 threads(16, 16, 16);
    strideCHK<<<blocks, threads>>>(10);
    cudaDeviceSynchronize();
    return 0;
}
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

__global__ void checkDim(void){
    printf("threadIdx: (%d, %d, %d) blockIdx: (%d, %d, %d) blockDim: (%d, %d, %d) gridDim: (%d, %d, %d) \n",
        threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, 
        blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
        // all four internal variables that are initialized by the kernel are printed
}

int main(){
    int nElem = 4 * 4;
    dim3 block (2, 2);
    dim3 grid ((nElem + block.x - 1)/block.x, (nElem + block.y - 1)/block.y);

    // printf("grid.x %d grid.y %d grid.z %d", grid.x, grid.y, grid.z);
    cout << "grid.x grid.y grid.z " << grid.x << " " << grid.y << " " << " " << grid.z << endl;
    // printf("block.x %d block.y %d block.z %d", block.x, block.y, block.z);
    cout << "block.x, block.y, block.z " << block.x << " " << block.y << " " << block.z << endl;

    checkDim<<<grid, block>>>();

    cudaDeviceReset();

    return 0;
}
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

#define CHECK(call)
{
    const cudaError_t error = call;
    if (error != cudaSuccess)
    {
        cout << "Error: " << __FILE__ << "Line: " << __LINE__ << endl;
        cout << "code: " << error << "reason: " << cudaGetErrorString(error) << endl;
        exit(1);
    }
}

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL); // this will work in C++
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void initialInt(int *ip, int size){
    for (int i = 0; i < size; i++){
        ip[i] = i;
    }
}

void printMatrix(int *C, const int nx, const int ny){
    int *ic = C;
    cout << "Matrix Size: " << nx << " * " << ny << endl;

    for (int iy = 0; iy < ny; iy++){
        for (int ix = 0; ix < nx; ix++){
            cout << ic[ix]
        }
        ic += nx;  // this will increment the address in the ic
        cout << endl;
    }
    cout << endl;
}

__global__ void printThreadIndex(int *A, const int nx, const int ny){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    // there will be 4 * 2 threads, like that 3 rows * 2 cols blocks
    unsigned int idx = iy * nx + ix;
    printf("thread_id (%d, %d) block_id (%d, %d) coordinate_id (%d, %d) global_index %2d ival %2d \n",
            threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);
}

int main(){
    cout << "Starting the Index checker" << endl;

    int dev = 0;
    CHECK(cudaSetDevice(dev));

    int nx = 8;
    int ny = 6;
    int nxy = nx * ny;

    int nBytes = nxy * sizeof(float);

    int *h_A;
    h_A = (int *)malloc(nBytes);

    intialInt(h_A, nxy);
    printMatrix(h_A, nx, ny);

    int *d_MatA;
    cudaMalloc((void **)&d_Mat_A, nBytes);

    cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice);

    dim3 block(4, 2);
    dim3 grid((nx + block.x - 1)/ block.x), (ny + block.y - 1)/ block.y);

    printThreadIndex <<<grid, block>>>(d_MatA, nx, ny);

    cudaDeviceSynchronize();

    cudaFree(d_MatA);
    free(h_A);

    cudaDeviceReset();

    return 0;
    
}
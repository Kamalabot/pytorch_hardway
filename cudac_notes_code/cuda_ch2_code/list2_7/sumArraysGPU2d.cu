
#include <iostream>
#include <cuda_runtime.h>
#include <sys/time.h>

using namespace std;
// used for checking the errors in the function calls and print them
#define CHECK(call)                                                                 \
{                                                                                   \
    const cudaError_t error = call;                                                 \
    if (error != cudaSuccess)                                                       \
    {                                                                               \
        cout << "code: " << error << "reason: " << cudaGetErrorString(error) << endl;\
        exit(1);                                                                    \
    }                                                                               \
}
// Comparing the result of matrix operation by host function and kernel
void checkResult(float *hostRef, float *gpuRef, const int N){
    double epsilon = 1.0E-8;
    bool match = 1;
    for (int i = 0; i < N; i++){
        if(abs(hostRef[i] - gpuRef[i]) > epsilon){
            match = 0;
            cout << "Arrays do not match." << endl;
            cout << "host: " << hostRef[i] << "gpu: " << gpuRef[i] << endl;
            break;
        }
    }
    if (match) cout << "Arrays Match" << endl;
}

// summing array on host
void sumArrayOnHost(float *a, float *b, float *c, const int N){
    // enumerate and sum
    for(int idx = 0; idx < N; idx++){
        c[idx] = a[idx] + b[idx]; // just take two elements and add them
    }
}

void initialData(float *ip, int size){
    time_t t;  // t is of time time_t, and its address is sent to srand
    // srand((unsigned int) time(&t)); // the returned time_t value is casted
    srand(static_cast<unsigned int>(time(0)));
    for (int j=0; j < size; j++){
        ip[j] = (float) ( rand() & 0xFF ) / 10.0f;
    }
}

int cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

__global__ void sumArrayOnGPU2d(float *A, float *B, float *C, int nx, int ny){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x; // calc ix from the ids of threads & blocks
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y; // calc iy from the ids of blocks & threads
    unsigned int idx = iy * nx + ix;  // calc the id of the array from ix & iy
    if(ix < nx && iy < ny){
        C[idx] = A[idx] + B[idx];  // the 2d matrix is linearly placed in the memory
    }
}

__global__ void sumArrayOnGPU1d(float *A, float *B, float *C, int nx, int ny){
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    // calc ix from the ids of threads & blocks
    if(ix < nx){
        // for every spawned thread, below loop executes for ny times
        for (int iy = 0; iy < ny; iy++){
            int idx = iy * nx + ix;
            // this is inside the loop now.
            C[idx] = A[idx] + B[idx];
            // the 2d matrix is linearly placed in the memory
        }
    }
}
// block(32, 1) and grid((nx + block.x - 1) / block.x, 1) 

__global__ void sumArrayOnGPUmix(float *A, float *B, float *C, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x; // calc ix from the ids of threads & blocks
    unsigned int iy = blockIdx.y;
    unsigned int idx = iy * nx + ix;
    if(ix < nx && iy < ny) C[idx] = A[idx] + B[idx];  // the blocks are growing along cols, sideways
} // block(32, 1) and grid((nx + block.x - 1) / block.x, 1) 


int main(){
    int dev = 0; // setup device to be 0
    cudaSetDevice(dev);

    int nx = 1 << 14; // set data 16,384 elems
    int ny = 1 << 14; // set data 16,384 elems
    cout << "Vector x size: " << nx << endl;
    cout << "Vector y size: " << ny << endl;
    
    int nxy = nx * ny;
    size_t nBytes = nxy * sizeof(float); 

    float *h_A, *h_B, *hostRef, *gpuRef;
    // https://www.geeksforgeeks.org/malloc-vs-new/ 
    // we can implement new based memory allocation 
    h_A = (float *)malloc(nBytes);  // its going to linear memory alloc
    h_B = (float *)malloc(nBytes);

    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    initialData(h_A, nxy);
    initialData(h_B, nxy);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A , nBytes);
    // &d_A is address in device memory, which is holding the 
    // pointer to the array of data
    cudaMalloc((float **)&d_B , nBytes);
    cudaMalloc((float **)&d_C , nBytes);

    // move data from host to device
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

    int dimx = 32; // 32, 16
    int dimy = 32; // 16, 16
    // invoke kernel at host side
    dim3 block (dimx, dimy);
    dim3 grid ((nx + block.x - 1) / block.x, (ny + block.y -1) / block.y);
    double iSt = cpuSecond();    
    sumArrayOnGPU1d<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    double iEl = cpuSecond() - iSt;

    cout << "grid.x " << grid.x << "block.x " << block.x << endl;
    printf("sumArrayOnGPU1D<<<(%d, %d), (%d, %d)>>> elapsed %f sec. \n",
          grid.x, grid.y, block.x, block.y, iEl);

    iSt = cpuSecond();    
    sumArrayOnGPU2d<<<grid, block>>>(d_A, d_B, d_C, nx, ny);
    iEl = cpuSecond() - iSt;

    cout << "grid.x " << grid.x << "block.x " << block.x << endl;
    printf("sumArrayOnGPU2d<<<(%d, %d), (%d, %d)>>> elapsed %f sec. \n",
          grid.x, grid.y, block.x, block.y, iEl);

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);

    sumArrayOnHost(h_A, h_B, hostRef, nxy);

    checkResult(hostRef, gpuRef, nxy);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}

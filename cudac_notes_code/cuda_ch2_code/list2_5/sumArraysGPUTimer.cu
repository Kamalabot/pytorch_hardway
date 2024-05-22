#include <iostream>
#include <ctime>
#include <sys/time.h>
#include <cuda_runtime.h>

using namespace std;

#define CHECK(call)                                                                 \
{                                                                                   \
    const cudaError_t error = call;                                                 \
    if (error != cudaSuccess)                                                       \
    {                                                                               \
        cout << "code: " << error << "reason: " << cudaGetErrorString(error) << endl;\
        exit(1);                                                                    \
    }                                                                               \
}

double cpuSecond(){
    struct timeval tp;
    gettimeofday(&tp, NULL); // this will work in C++
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

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

void sumArrayOnHost(float *a, float *b, float *c, const int N){
    for(int idx = 0; idx < N; idx++){
        c[idx] = a[idx] + b[idx]; // just take two elements and add them
    }
}

void initialData(float *ip, int size){
    // time_t t;  // t is of time time_t, and its address is sent to srand
    // srand((unsigned int) time(&t)); // the returned time_t value is casted
    srand(static_cast<unsigned int>(time(0)));
    for (int j=0; j < size; j++){
        ip[j] = (float) ( rand() & 0xFF ) / 10.0f;
    }
}

__global__ void sumArrayOnGPU(float *A, float *B, float *C){
    int i = threadIdx.x; // getting the ids of initialized threads
    // int i = blockIdx.x; // if the threadIdx.x is set to 1
    C[i] = A[i] + B[i];
}

__global__ void gensumArrayOnGPU(float *A, float *B, float *C, const int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main(){
    int dev = 0; // setup device to be 0
    cudaSetDevice(dev);
    cudaDeviceProp devProp;
    CHECK(cudaGetDeviceProperties(&devProp, dev))
    cout << "Using device: " << dev << " Named as: " << devProp.name << endl;

    int nElem = 32; // set data
    cout << "Vector size: " << nElem << endl;

    size_t nBytes = nElem * sizeof(float); 

    float *h_A, *h_B, *hostRef, *gpuRef;
    // https://www.geeksforgeeks.org/malloc-vs-new/ 
    // we can implement new based memory allocation 
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    initialData(h_A, nElem);
    initialData(h_B, nElem);

    // Memset() is a C++ library function used to
    // fill a memory block with a particular value. This function
    // takes three arguments: a pointer to the starting address of the
    // memory block to be filled, the value to be set, and the number 
    // of bytes to be filled.
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

    // invoke kernel at host side
    dim3 block (nElem);
    dim3 grid (nElem/block.x);
    // measuring the elapsed time.
    double iSt = cpuSecond();
    sumArrayOnGPU<<<grid, block>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize(); 
    // ensure all threads have been completed, data transfer is not considered 
    double iEl = cpuSecond() - iSt;
    cout << "grid.x " << grid.x << "block.x " << block.x << endl;
    cout << "Time elapsed on GPU Kernel " << iEl << " secs " << endl;

    cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
    iSt = cpuSecond();
    sumArrayOnHost(h_A, h_B, hostRef, nElem);
    iEl = cpuSecond() - iSt;

    cout << "Time elapsed on Host function " << iEl << " secs";
    checkResult(hostRef, gpuRef, nElem);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}
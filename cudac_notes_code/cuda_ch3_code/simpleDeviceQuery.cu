
#include <iostream>
#include <cuda_runtime.h>

int main(){
    int iDev = 0;
    cudaDeviceProp iProp;
    cudaGetDeviceProperties(&iProp, iDev);

    cout << "Device name: " << iProp.name << endl;

    cout << "Number of multi-processor: " << iProp.multiProcessorCount << endl;
    
    cout << "Constant Memory: " << iProp.totalConstMem / 1024.0 << endl;
    cout << "Number of registers per block:  " << iProp.regsPerBlock << endl;
    cout << "Warp Size: " << iProp.warpSize << endl;
    cout << "Max number of threads per block: " << iProp.maxThreadsPerBlock << endl;
    cout << "Threads per multiprocessor: " << iProp.maxThreadsPerMultiProcessor << endl;
    cout << "Warps per multiprocessor: " << iProp.maxThreadsPerMultiProcessor / 32 << endl;
    return EXIT_SUCCESS;
}

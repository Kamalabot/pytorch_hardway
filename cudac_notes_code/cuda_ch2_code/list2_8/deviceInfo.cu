#include <cuda_runtime.h>
#include <iostream>

int main(){
    int devCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&devCount);

    if (error_id != cudaSuccess){
        cout << "Cuda device count returned: " << "Error id: " << error_id << "Error: " << cudaErrorString(error_id) << endl;
        cout << "Result: Failed" << endl;
        exit(EXIT_FAILURE)
    }

    if (devCount == 0){
        cout << "There is no available device that support CUDA" << endl;
    } else {
        cout << "Detected device-" << devCount << " CUDA compatible device" << endl; 
    }

    int dev, driverVersion = 0, runtimeVersion = 0;

    dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev)

    cout << "Device no: " << dev << " and Device Name: " << deviceProp.name); 

    cudaDriverGetVersion(&driverVersion);  // gets the driver version
    cudaRuntimeGetVersion(&runtimeVersion);  // gets runtime version

    cout << "Cuda Driver Version " << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << endl;
    cout << "Runtime Version " << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10 << endl;

    cout << "Cuda Capability: Major version " << deviceProp.major << " Minor Version: " << deviceProp.minor << endl; 
    
    cout << "Total Amount of Global Memory: " << deviceProp.totalGlobalMem / (pow(1024.0, 3)) << " MBytes" << endl;

    cout << "Max Multi-processor: " < deviceProps.multiProcessorCount << endl;

    cout << "GPU Clock Rate: " << deviceProp.clockRate * 1e-3f << endl;
    
    cout << "Memory Clock Rate: " << deviceProp.memoryClockRate * 1e-3f << endl;

    cout << "Memory Bus Width: " << deviceProp.memoryBusWidth << endl;

    if (deviceProp.l2CacheSize){
        cout << "L2 Cache size: " << deviceProp.l2CacheSize << endl;
    }

    cout << " Max Texture 1D " << deviceProp.maxTexture1D << endl; 
    cout << " Max Texture 2D[0] " << deviceProp.maxTexture2D[0] << endl; 
    cout << " Max Texture 2D[1] " << deviceProp.maxTexture2D[0] << endl; 
    cout << " Max Texture 3D[0] " << deviceProp.maxTexture3D[0] << endl; 
    cout << " Max Texture 3D[1] " << deviceProp.maxTexture3D[1] << endl; 
    cout << " Max Texture 3D[2] " << deviceProp.maxTexture3D[2] << endl; 

    cout << "Total const memory: " << deviceProp.totalConstMem << endl;
    cout << "Shared memory per Block: " << deviceProp.sharedMemPerBlock << endl;
    cout << "Register per Block: " << deviceProp.regsPerBlock << endl;

    cout << "Warp Size: " << deviceProp.warpSize << endl;

    cout << "Max Threads per SM: " << deviceProp.maxThreadsPerMultiProcessor << endl;

    cout << "Max Threads per Block: " << deviceProp.maxThreadsPerBlock << endl; 

    cout << "Max Thread DIM[0]: " << deviceProp.maxThreadsDim[0] << endl;
    cout << "Max Thread DIM[1]: " << deviceProp.maxThreadsDim[1] << endl;
    cout << "Max Thread DIM[2]: " << deviceProp.maxThreadsDim[2] << endl;

    cout << "Max Grid Size[0]: " << deviceProp.maxGridSize[0] << endl;
    cout << "Max Grid Size[1]: " << deviceProp.maxGridSize[1] << endl;
    cout << "Max Grid Size[2]: " << deviceProp.maxGridSize[2] << endl;

    cout << "Max mem pitch: " << deviceProp.memPitch << endl;
}
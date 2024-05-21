#include <iostream>
using namespace std;

__global__ void helloGPU(void){
    printf("Hello from GPU...");
}

int main(){
    cout << "Hello from CPU" << endl;
    helloGPU <<<1, 10>>>();
    cudaDeviceReset(); // with this one version
    // cudaDeviceSynchronize(); // with this 2nd version
    return 0;
}
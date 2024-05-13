#include <stdio.h>

__global__ void helloCUDA(float f)
{
    printf("Hello thread %d, f=%f\n", threadIdx.x, f);

    if(threadIdx.x > 10)
      printf("This is a different path %d thread\n", threadIdx.x);
}

int main(){
    helloCUDA<<<1, 15>>>(1.2345f);
    cudaDeviceSynchronize();
    return 0;
}
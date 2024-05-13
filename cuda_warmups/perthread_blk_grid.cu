# include <stdio.h>

__global__ void strideCHK(int n)
{
    int index = threadIdx.x;
    int stride = blockDim.x;

    printf("threadIndex is %d and BlockDim.x is %d\n", index, stride);
    
    for(int i = index; i < n; i += stride){
        printf("Stride is %d\n", stride);
        // stride above prints 15, as the blockDim.x is 15
        printf("I value is %d\n", i);
        // i value increments only with one
    }
}

int main(){
    strideCHK<<<1, 15>>>(10);
    cudaDeviceSynchronize();
    return 0;
}
// cuda_example.cu
#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void add_kernel(int *a, int *b, int *c, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  printf("%d is entered \n", i);
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  int n = 10;
  int *a, *b, *c;

  cudaMallocManaged(&a, n * sizeof(int));
  cudaMallocManaged(&b, n * sizeof(int));
  cudaMallocManaged(&c, n * sizeof(int));

  for (int i = 0; i < n; i++) {
    a[i] = i;
    b[i] = i * 2;
    cout << "a[i]: "<< a[i] * 2 << endl;
  }

  // add_kernel<<<1, 1>>>(a, b, c, n);
  add_kernel<<<1, 10>>>(a, b, c, n);
  cudaDeviceSynchronize();

  for (int i = 0; i < n; i++) {
    printf("%d + %d = %d\n", a[i], b[i], c[i]);
  }
  
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  return 0;
}
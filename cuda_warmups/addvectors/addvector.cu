// cuda_example.cu
#include <stdio.h>

__global__ void add_kernel(int *a, int *b, int *c, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  int n = 10;
  int *a, *b, *c;

  cudaMalloc(&a, n * sizeof(int));
  cudaMalloc(&b, n * sizeof(int));
  cudaMalloc(&c, n * sizeof(int));

  for (int i = 0; i < n; i++) {
    a[i] = i;
    b[i] = i * 2;
  }

  add_kernel<<<1, 1>>>(a, b, c, n);

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);

  for (int i = 0; i < n; i++) {
    printf("%d + %d = %d\n", a[i], b[i], c[i]);
  }

  return 0;
}
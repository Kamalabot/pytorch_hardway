#include <stdio.h>
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/torch.h>
#include <iostream>

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;
}
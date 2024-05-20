The CMakeLists is compiling more like pure CUDA C code, as the extension of the file itself is *.cu 
In addition the CUDA_LIBRARIES and CUDA_RUNTIME is accessed from the code.

The NVCC compiler will be used for compiling the code.

A deep dive on the CUDA support in CMAKE is treated here 
https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html

Got Some more examples 
- https://github.com/pkestene/cuda-proj-tmpl/tree/master
- https://github.com/r-barnes/pytorch_cmake_example.git is little more involved, which is getting compiled 

Compilation Challenges:
    - Code is compiling.
    - The kernel is getting called
    - when add_kernel<<<1, 1>>> is used, the operations did not happen 
    - when add_kernel<<<1, 10>>> is used, the correct results came up 
    There is no issue of CudaDeviceSynchronize() error
    - Adding env variable export CUDAFLAGS="-arch=sm_89 --expt-extended-lambda" does help in getting the kernel call

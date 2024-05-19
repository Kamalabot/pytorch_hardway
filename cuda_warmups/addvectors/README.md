The CMakeLists is compiling more like pure CUDA C code, as the extension of the file itself is *.cu 
In addition the CUDA_LIBRARIES and CUDA_RUNTIME is accessed from the code.

The NVCC compiler will be used for compiling the code.

A deep dive on the CUDA support in CMAKE is treated here 
https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html

Got Some more examples 
- https://github.com/pkestene/cuda-proj-tmpl/tree/master
- https://github.com/r-barnes/pytorch_cmake_example.git is little more involved, which is getting compiled 

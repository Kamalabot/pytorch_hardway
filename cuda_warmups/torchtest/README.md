Underlying code is C++, through which the torch libraries are accessed.
The CMakeLists.txt is similar to regular C++ CMakeLists, with added linking to TORCH_LIBRARIES 

The code executes on Cuda, and uses g++ compiler

Following option is also shown . If PyTorch was installed via conda or pip, CMAKE_PREFIX_PATH can be
queried using torch.utils.cmake_prefix_path variable.
-CMAKE_PREFIX_PATH=python -c ‘import torch;print(torch.utils.cmake_prefix_path)’`

Following is provided in the pytorch docs https://pytorch.org/cppdocs/installing.html 
-DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch

Compilation Challenges:
    - CMakeLists work, and the compiler there is g++ which also works 
    - executable works without any challenge
    - cmake -S .. -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch is required 
 
Compilation Challenges:
- CudaDeviceSynchronize() is undefined error arose:
    - Found that it is deprecated based on https://forums.developer.nvidia.com/t/cant-run-my-program-on-rtx-4080/237457/2
    solution is to use "-D CUDA_FORCE_CDP1_IF_SUPPORTED" (not working..)
    - Tried understand the purpose of the CudaDeviceSynchronize() https://stackoverflow.com/questions/11888772/when-to-call-cudadevicesynchronize
    - The HostSide DeviceSynchronize seems to be still allowed as per https://forums.developer.nvidia.com/t/cudadevicesynchronize-from-device-code-is-deprecated/215900/3
    review the same.
- The app is being parked for further understanding, and working on it later.
    - The App is only trying to print the threadIds and blockIds
    by sending the details through the Kernel call 
    - Then trying is trying to get the printing done.  

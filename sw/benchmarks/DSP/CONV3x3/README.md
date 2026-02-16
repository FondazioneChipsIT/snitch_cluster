# CONV 3x3
This convolution Kernel does not use padding. If the input matrix is NxN, the output will be (N-2)x(N-2). 

If needed one could change the filter size (e.g 5x5 or 7x7), but this would need a change in the lenght of the assembly code found in the kernel, as it is hardcoded to function with 3x3.
This is a result of high optimization using FREP and SSRs.

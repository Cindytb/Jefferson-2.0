#pragma once
#ifndef _CUDAPART_H_
#define _CUDAPART_H_
#include "Universal.cuh"
#include "hrtf_signals.cuh"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

/*CUDA Includes*/
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <cmath>

#include "kernels.cuh"
// Complex data type



typedef struct Data_tag Data;


/*Forward Declarations*/
void cudaFFT(int argc, char **argv, Data *p);

////////////////////////////////////////////////////////////////////////////////
///*NOTE: GPU Convolution was not fast enough because of the large overhead
//of FFT and IFFT. Keeping the code here for future purposes*/
//
//static __global__ void interleaveMe(float *output, float *input, int size);
//__global__ void copyMe(int size, float *output, float *input);
//void convolveMe(float *output, float *input, int x_len, float *p_hrtf, float gain, float *d_hrtf);
////////////////////////////////////////////////////////////////////////////////

#endif
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

// Complex data type
typedef float2 Complex; 
typedef struct Data_tag Data;

template <typename T>
struct square
{
	__host__ __device__
		T operator()(const T& x) const
	{
		return x * x;
	}
};
/*Forward Declarations*/
void cudaFFT(int argc, char **argv, Data *p);
int PadData(const float *signal, float **padded_signal, int signal_size,
	const float *filter_kernel, float **padded_filter_kernel, int filter_kernel_size);

static __device__ __host__ inline Complex ComplexScale(Complex, float);
static __device__ __host__ inline Complex ComplexMul(Complex, Complex);
static __global__ void ComplexPointwiseMulAndScale(Complex *a, const Complex *b, int size, float scale);
static __global__ void MyFloatScale(float *a, int size, float scale);


////////////////////////////////////////////////////////////////////////////////
///*NOTE: GPU Convolution was not fast enough because of the large overhead
//of FFT and IFFT. Keeping the code here for future purposes*/
//
//static __global__ void interleaveMe(float *output, float *input, int size);
//__global__ void copyMe(int size, float *output, float *input);
//void convolveMe(float *output, float *input, int x_len, float *p_hrtf, float gain, float *d_hrtf);
////////////////////////////////////////////////////////////////////////////////

#endif
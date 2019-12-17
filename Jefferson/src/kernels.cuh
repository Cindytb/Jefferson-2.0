#pragma once 
#ifndef __KERNELS__
#define __KERNELS__
#include "Universal.cuh"
#include "cufftDefines.cuh"
/*CUDA Includes*/
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <helper_functions.h>
#include <helper_cuda.h>


/*Thrust*/
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <cmath>

__global__ void simple_vbo_kernel(float4 *pos, unsigned int width, unsigned int height, float time);
__global__ void MyFloatScale(float *a, int size, float scale);
__device__ __host__ inline cufftComplex cufftComplexScale(cufftComplex a, float s);
__device__ __host__ inline cufftComplex cufftComplexMul(cufftComplex a, cufftComplex b);
int PadData(const float *signal, float **padded_signal, int signal_size,
	const float *filter_kernel, float **padded_filter_kernel, int filter_kernel_size);
__global__ void averagingKernel(float4 *pos, float *d_buf, unsigned int size, double ratio, int averageSize);
__global__ void generateDistanceFactor(cufftComplex* in, float frac, float fsvs, float r, int N);
__global__ void crossFade(float* out1, float* out2, int numFrames);
__global__ void ComplexPointwiseMulAndScale(cufftComplex *a, const cufftComplex *b, int size, float scale);
__global__ void ComplexPointwiseMulAndScaleOutPlace(const cufftComplex* a, const cufftComplex* b, cufftComplex* c, int size, float scale);
__global__ void ComplexPointwiseMulInPlace(const cufftComplex* in, cufftComplex* out, int size);
__global__ void ComplexPointwiseMul(cufftComplex* a, const cufftComplex* b, cufftComplex* c, int size);
__global__ void ComplexPointwiseAdd(cufftComplex* in, cufftComplex* out, int size);
__global__ void timeDomainConvolutionNaive(float* ibuf, float* rbuf, float* obuf, long long oframes,
	long long rframes, int ch, float gain);
__global__ void doNothing();
__global__ void interleave(float* input, float* output, int size);
void fillWithZeroes(float** target_buf, long long old_size, long long new_size, cudaStream_t s);
void fillWithZeroes(float** target_buf, long long old_size, long long new_size);
void fillWithZeroesKernel(float* buf, int size, cudaStream_t s);

#endif
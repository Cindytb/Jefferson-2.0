#pragma once
#ifndef _CUDAPART_H_
#define _CUDAPART_H_
#include "Universal.cuh"
#include "DataTag.cuh"
#include "kernels.cuh"
#include "cufftDefines.cuh"

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

extern fftwf_complex* fft_hrtf;
/*Forward Declarations*/
void cudaFFT(int argc, char **argv, Data *p);

void transform_hrtfs();

#endif
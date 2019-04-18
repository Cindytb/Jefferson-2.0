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
typedef float2 Complex; 
typedef struct Data_tag Data;


/*Forward Declarations*/
void cudaFFT(int argc, char **argv, Data *p);


// cuFFT API errors
static const char *_ctb_cudaGetErrorEnum(cufftResult error)
{
	switch (error)
	{
	case CUFFT_SUCCESS:
		return "CUFFT_SUCCESS";

	case CUFFT_INVALID_PLAN:
		return "CUFFT_INVALID_PLAN";

	case CUFFT_ALLOC_FAILED:
		return "CUFFT_ALLOC_FAILED";

	case CUFFT_INVALID_TYPE:
		return "CUFFT_INVALID_TYPE";

	case CUFFT_INVALID_VALUE:
		return "CUFFT_INVALID_VALUE";

	case CUFFT_INTERNAL_ERROR:
		return "CUFFT_INTERNAL_ERROR";

	case CUFFT_EXEC_FAILED:
		return "CUFFT_EXEC_FAILED";

	case CUFFT_SETUP_FAILED:
		return "CUFFT_SETUP_FAILED";

	case CUFFT_INVALID_SIZE:
		return "CUFFT_INVALID_SIZE";

	case CUFFT_UNALIGNED_DATA:
		return "CUFFT_UNALIGNED_DATA";

	case CUFFT_INCOMPLETE_PARAMETER_LIST:
		return "CUFFT_INCOMPLETE_PARAMETER_LIST";

	case CUFFT_INVALID_DEVICE:
		return "CUFFT_INVALID_DEVICE";

	case CUFFT_PARSE_ERROR:
		return "CUFFT_PARSE_ERROR";

	case CUFFT_NO_WORKSPACE:
		return "CUFFT_NO_WORKSPACE";

	case CUFFT_NOT_IMPLEMENTED:
		return "CUFFT_NOT_IMPLEMENTED";

	case CUFFT_LICENSE_ERROR:
		return "CUFFT_LICENSE_ERROR";

	case CUFFT_NOT_SUPPORTED:
		return "CUFFT_NOT_SUPPORTED";
	}

	return "<unknown>";
}
#ifndef CHECK_CUFFT_ERRORS
#define CHECK_CUFFT_ERRORS(call) { \
    cufftResult_t err; \
    if ((err = (call)) != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT error %d:%s at %s:%d\n", err, _ctb_cudaGetErrorEnum(err), \
                __FILE__, __LINE__); \
        exit(1); \
    } \
}
#endif

#endif
#pragma once
#ifndef __HRTF_SIGNALS_CUDA
#define __HRTF_SIGNALS_CUDA
#include "Universal.cuh"
#include <stdio.h>
#include <math.h>       /* round, floor, ceil */
#include <stdlib.h>     /* malloc() */
#include <stdbool.h>    /* true, false */
#include <string.h>     /* memset() */
#include <sndfile.h>    /* libsndfile */

/*CUDA Includes*/
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>


#define HRTF_DIR	"full"

#define SAMP_RATE   44100
#define NUM_ELEV    14


#define PATH_LEN    256

/* function prototypes */
int read_hrtf_signals(void);
int pick_hrtf(float obj_ele, float obj_azi);

void GPUconvolve_hrtf(float *input, int hrtf_idx, float *output, int outputLen, float gain, cudaStream_t *stream);
#endif /* __HRTF_SIGNALS_CUDA */
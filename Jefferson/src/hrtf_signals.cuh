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
#include "cudaPart.cuh"

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define HRTF_DIR	"compact"

#define SAMP_RATE   44100
#define NUM_ELEV    14
#define NUM_HRFT    366

#define PATH_LEN    256

/* function prototypes */
int read_hrtf_signals(void);
int pick_hrtf(float obj_ele, float obj_azi);
/*Convolution on CPU*/
int convolve_hrtf(float *input, int hrtf_idx, float *output, int outputLen, float gain);


////////////////////////////////////////////////////////////////////////////////
/*NOTE: GPU Convolution was not fast enough because of the large overhead
of FFT and IFFT. Keeping the code here for future purposes*/
void GPUconvolve_hrtf(float *input, int hrtf_idx, float *output, int outputLen, float gain, cudaStream_t *stream);
////////////////////////////////////////////////////////////////////////////////
#endif /* __HRTF_SIGNALS_CUDA */
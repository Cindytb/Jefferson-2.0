#pragma once
#ifndef __HRTF_SIGNALS_CUDA
#define __HRTF_SIGNALS_CUDA
#include "Universal.cuh"
#include "kernels.cuh"
#include "functions.h"
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

#include <fftw3.h>


#define HRTF_DIR	"full"

#define SAMP_RATE   44100
#define NUM_ELEV    14

extern float* hrtf;
extern float* d_hrtf;
extern fftwf_complex* fft_hrtf;
extern cufftComplex* d_fft_hrtf;

#define PATH_LEN    256
extern int elevation_pos[NUM_ELEV];
extern float azimuth_inc[NUM_ELEV];
extern int azimuth_offset[NUM_ELEV + 1];
/* function prototypes */
int read_hrtf_signals(void);
int pick_hrtf(float obj_ele, float obj_azi);
void cleanup_hrtf_buffers();
void transform_hrtfs();

extern fftwf_complex* fft_hrtf;
#endif /* __HRTF_SIGNALS_CUDA */
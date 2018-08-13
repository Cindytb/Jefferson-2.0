//#pragma once
//#ifndef __HRTF_SIGNALS
//#define __HRTF_SIGNALS
//#include "Universal.cuh"
//
//
///* specific to this HRTF set
//* azimuth 0 is directly in front
//* increasing azimuth moves to the right
//* azimuth 180 is directly behind
//*
//* HRTF positions on left half-sphere are identical to
//* symmetrical positions on right half-sphere and
//* HRTF signals on the left half-sphere have L, R channels exchanged
//*/
//#define HRTF_DIR	"compact"
//
//
//#define SAMP_RATE   44100
//#define NUM_ELEV    14
//#define NUM_HRFT    366
//
//#define PATH_LEN    256
//
///* function prototypes */
//int read_hrtf_signals(void);
//int pick_hrtf(float obj_ele, float obj_azi);
//int convolve_hrtf(float *x, int x_len, int hrtf_idx, float *y, int y_len, float gain);
//float* getHRTF();
//#endif /* __HRTF_SIGNALS */
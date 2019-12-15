#pragma once
#ifndef __SOUND_SOURCE__
#define __SOUND_SOURCE__

#include "Universal.cuh"
#include "VBO.cuh"
#include "cufftDefines.cuh"
#include "hrtf_signals.cuh"

#include <math.h>
#include <cmath>
extern __device__ __constant__ float d_hrtf[NUM_HRTF * HRTF_CHN * (PAD_LEN + 2)];
class SoundSource {
public:
	float* buf;							/*Reverberated signal on host*/
	float* x[FLIGHT_NUM];				/*Buffer for PA output on host, mono, pinned memory, PAD_LEN + 2 size */
	float* intermediate[FLIGHT_NUM];	/*Host data of the output*/
	float* d_input[FLIGHT_NUM];			/*PAD_LEN + 2 sized for the input*/
	float* d_convbufs[FLIGHT_NUM * 8];	/*2 * (PAD_LEN + 2) sized for each interpolation buffer*/
	float* d_output[FLIGHT_NUM];		/*2 * (PAD_LEN + 2)  sized for the output*/

	cufftHandle in_plan, out_plan;	/*cufft plans*/
	cudaStream_t* streams;			/*Streams associated with each block in flight*/

	int count;						/*Current frame count for the audio callback*/
	int hrtf_idx; 					/*Index to the correct HRTF elevation/azimuth*/
	int old_hrtf_idx;
	int length;						/*Length of the input signal*/

	/*Graphics Writable & Audio Readable*/
	float3 coordinates;				/*3D Coordinates of the source*/
	float gain;						/*Gain for distance away*/
	float ele;						/*Elevation of the sound source*/
	float azi;						/*Azimuth of the sound source*/

	VBO* waveform;
	SoundSource(); /*Initialize plans and allocate memory on GPU*/
	~SoundSource(); /*Destroy plans and deallocate memory on GPU*/
	void updateInfo(); /*Update the azimuth and elevation. Called by display()*/
	void drawWaveform(); /*Renders the VBO onto the screen. Called by display()*/
	void fftConvolve(int blockNo); /*Uses time domain convolution rounding to the nearest HRTF in the database*/
	void interpolateConvolve(int blockNo); /*Uses Belloch's technique of interpolation*/
private:
	void SoundSource::interpolationCalculations(int* hrtf_indices, float* omegas);

};
#endif
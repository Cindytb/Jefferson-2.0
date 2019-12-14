#pragma once
#ifndef __SOUND_SOURCE__
#define __SOUND_SOURCE__

#include "Universal.cuh"
#include "VBO.cuh"
#include "cufftDefines.cuh"
#include "hrtf_signals.cuh"

#include <math.h>
#include <cmath>
class SoundSource {
public:
	float* buf;							/*Reverberated signal on host*/
	float* x[FLIGHT_NUM];				/*Buffer for PA output on host, mono, pinned memory, PAD_LEN + 2 size */
	float* intermediate[FLIGHT_NUM];	/*Host data of the output*/
	float* d_input[FLIGHT_NUM];			/*PAD_LEN + 2 sized for the input*/
	float* d_output[FLIGHT_NUM];		/*2 * (PAD_LEN + 2)  sized for the output*/

	cufftHandle in_plan, out_plan;	/*cufft plans*/
	cudaStream_t* streams;			/*Streams associated with each block in flight*/

	int count;						/*Current frame count for the audio callback*/
	int hrtf_idx; 					/*Index to the correct HRTF elevation/azimuth*/
	int length;						/*Length of the input signal*/

	/*Graphics Writable & Audio Readable*/
	float3 coordinates;				/*3D Coordinates of the source*/
	float gain;						/*Gain for distance away*/
	float ele;						/*Elevation of the sound source*/
	float azi;						/*Azimuth of the sound source*/

	VBO* waveform;
	SoundSource();
	~SoundSource();
	void updateInfo();
	void drawWaveform();
	void fftConvolve(int blockNo);

};
#endif
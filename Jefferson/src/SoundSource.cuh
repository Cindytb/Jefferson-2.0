#pragma once
#ifndef __SOUND_SOURCE__
#define __SOUND_SOURCE__

#include "Universal.cuh"
#include "VBO.cuh"
#include "cufftDefines.cuh"
#include "hrtf_signals.cuh"

#include <math.h>
#include <cmath>
#include <fftw3.h>
#include <omp.h>

class SoundSource {
public:
	float* buf;									/*Reverberated signal on host*/
	float* x[FLIGHT_NUM];						/*Buffer for PA output on host, mono, pinned memory, PAD_LEN + 2 size */
	float* intermediate[FLIGHT_NUM];			/*Host data of the output*/
	cufftComplex* distance_factor[FLIGHT_NUM]; 	/*Buffer for the complex factor to account for distance scaling PAD_LEN / 2 + 1 Complex values*/
	float* d_input[FLIGHT_NUM];					/*PAD_LEN + 2 sized for the input*/
	cufftComplex* d_convbufs[FLIGHT_NUM];		/*4 * 2 * (PAD_LEN + 2) sized for each interpolation buffer*/
	float* d_output[FLIGHT_NUM];				/*2 * (PAD_LEN + 2)  sized for the output*/
	fftwf_complex* fftw_intermediate;			/*2 * (PAD_LEN / 2 + 1) complex values. Padded buffer for fftw output*/

	
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
	unsigned long long num_calls = 0;

	VBO* waveform;
	SoundSource(); /*Initialize plans and allocate memory on GPU*/
	~SoundSource(); /*Destroy plans and deallocate memory on GPU*/
	void updateInfo(); /*Update the azimuth and elevation. Called by display()*/
	void drawWaveform(); /*Renders the VBO onto the screen. Called by display()*/
	void process(int blockNo); /*Wrapper for the processor convolver. Calls fftConvolve() or interpolateConvolve()*/
	void cpuTDConvolve(float *input, float *output, int outputLen, float gain);
	void cpuFFTConvolve();
	void chunkProcess(int blockNo);
private:
	void fftConvolve(int blockNo); /*Uses time domain convolution rounding to the nearest HRTF in the database*/
	void interpolateConvolve(int blockNo); /*Uses Belloch's technique of interpolation*/
	void gpuTDConvolve(float* input, float* d_output, int outputLen, float gain, cudaStream_t* streams);
	void allKernels(float* d_input, float* d_output, cufftComplex* d_convbufs, cufftComplex* d_distance_factor, cudaStream_t* streams, float* omegas, int* hrtf_indices); /*All of the kernels for interpolation*/
	void interpolationCalculations(int* hrtf_indices, float* omegas); /*Determine all omegas and hrtf indices*/
	void calculateDistanceFactor(int blockNo);
	int hrtf_indices[4];
	int old_hrtf_indices[4];
	float omegas[6];
	float old_omegas[6];
	float sum_ms = 0;
	float avg_ms = 0;
	int num_iterations = 0;
	cufftHandle in_plan, out_plan;
	fftwf_plan fftw_in_plan, fftw_out_plan; 
};
#endif
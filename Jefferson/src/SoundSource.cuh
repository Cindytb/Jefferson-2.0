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
struct callback_data {
	float* input;
	float* output;
	size_t size;
	int blockNo;
};
class SoundSource {
public:
	float* buf;									/*Reverberated signal on host*/
	float* x[FLIGHT_NUM];						/*Buffer for PA output on host, mono, pinned memory, PAD_LEN + 2 size */
	float* intermediate[FLIGHT_NUM];			/*Host data of the output*/
	cufftComplex* distance_factor[FLIGHT_NUM]; 	/*Buffer for the complex factor to account for distance scaling PAD_LEN / 2 + 1 Complex values*/
	float* d_input[FLIGHT_NUM];					/*PAD_LEN + 2 sized for the input*/
	cufftComplex* d_convbufs[FLIGHT_NUM];		/*4 * 2 * (PAD_LEN + 2) sized for each interpolation buffer*/
	cufftComplex* d_convbufs2[FLIGHT_NUM];		/*4 * 2 * (PAD_LEN + 2) sized for each interpolation buffer, part of the crossfading functions*/
	float* d_output[FLIGHT_NUM];				/*2 * (PAD_LEN + 2) * 2  sized for the output, *2 for switching method*/
	float* d_output2[FLIGHT_NUM];				/*2 * (PAD_LEN + 2)  sized for the output, part of the crossfading function*/

	callback_data callback_data_blocks[FLIGHT_NUM * 3];
	fftwf_complex* fftw_intermediate;			/*2 * (PAD_LEN / 2 + 1) complex values. Padded buffer for fftw output*/
	fftwf_complex* fftw_conv_bufs;				/*2 * (PAD_LEN / 2 + 1) complex values. Buffer for interpolation, 8 of them total*/
	fftwf_complex* fftw_distance_factor;		/*Buffer for the complex factor to account for distance scaling PAD_LEN / 2 + 1 Complex values*/
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
	float r;						/*Sound source distance*/
	unsigned long long num_calls = 0;

	VBO* waveform;
	SoundSource(); /*Initialize plans and allocate memory on GPU*/
	~SoundSource(); /*Destroy plans and deallocate memory on GPU*/
	void drawWaveform(); /*Renders the VBO onto the screen. Called by display()*/
	void process(int blockNo); /*Wrapper for the processor convolver. Calls fftConvolve() or interpolateConvolve()*/
	void cpuTDConvolve(float *input, float *output, int outputLen, float gain);
	void cpuFFTConvolve();
	void updateFromCartesian(float3 point);
	void updateFromCartesian();
	void updateFromSpherical(float azi, float ele, float r);
	void updateFromSpherical();
	void chunkProcess(int blockNo);
	void sendBlock(int blockNo);
	void overlapSave(int blockNo);
	void copyIncomingBlock(int blockNo);
	void receiveBlock(int blockNo);
	
	void cpuInterpolateLoops(fftwf_complex* output, fftwf_complex* convbufs, int* hrtf_indices, float* omegas);
private:
	void cpuFFTInterpolate();
	void fftConvolve(int blockNo); /*Uses time domain convolution rounding to the nearest HRTF in the database*/
	void interpolateConvolve(int blockNo); /*Uses Belloch's technique of interpolation*/
	void gpuTDConvolve(float* input, float* d_output, int outputLen, float gain, cudaStream_t* streams);
	void allKernels(float* d_input, float* d_output, cufftComplex* d_convbufs, cufftComplex* d_distance_factor, cudaStream_t* streams, float* omegas, int* hrtf_indices, cudaEvent_t fft_in); /*All of the kernels for interpolation*/
	void interpolationCalculations(float ele, float azi, int* hrtf_indices, float* omegas); /*Determine all omegas and hrtf indices*/
	void gpuCalculateDistanceFactor(int blockNo, cudaStream_t stream);
	void cpuCalculateDistanceFactor();
	int hrtf_indices[4];
	float old_ele;
	float omegas[6];
	float old_azi;
	float sum_ms = 0;
	float avg_ms = 0;
	int num_iterations = 0;
	cufftHandle in_plan, out_plan, out_plan2;
	fftwf_plan fftw_in_plan, fftw_out_plan; 
	cudaEvent_t incomingTransfers[FLIGHT_NUM * 3];
};
#endif
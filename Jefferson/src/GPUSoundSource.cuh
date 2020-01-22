#pragma once
#ifndef __GPU_SOUND_SOURCE__
#define __GPU_SOUND_SOURCE__

#include "SoundSource.cuh"
#include "CPUSoundSource.h"
#include "Universal.cuh"
#include "VBO.cuh"
#include "cufftDefines.cuh"
#include "hrtf_signals.cuh"

struct callback_data {
	float* input;
	float* output;
	size_t size;
};
class GPUSoundSource : CPUSoundSource {
public:

	cudaStream_t streams[STREAMS_PER_FLIGHT];			/*Streams associated with each block in flight*/
	float* intermediate;			/*Host data of the output*/
	void process(processes type); /*Wrapper for the processor convolver. Calls fftConvolve() or interpolateConvolve()*/
	GPUSoundSource();
	~GPUSoundSource();
	
//private:
	float* x;						/*Buffer for PA output on host, mono, pinned memory, PAD_LEN + 2 size */
	cufftComplex* d_distance_factor; 	/*Buffer for the complex factor to account for distance scaling PAD_LEN / 2 + 1 Complex values*/
	float* d_input;					/*PAD_LEN + 2 sized for the input*/
	cufftComplex* d_convbufs;		/*4 * 2 * (PAD_LEN + 2) sized for each interpolation buffer*/
	cufftComplex* d_convbufs2;		/*4 * 2 * (PAD_LEN + 2) sized for each interpolation buffer, part of the crossfading functions*/
	float* d_output;				/*2 * (PAD_LEN + 2) * 2  sized for the output, *2 for switching method*/
	float* d_output2;				/*2 * (PAD_LEN + 2)  sized for the output, part of the crossfading function*/

	callback_data callback_data_blocks[3];
	void chunkProcess();
	void sendBlock();
	void overlapSave();
	void copyIncomingBlock();
	void receiveBlock();
	void fftConvolve(); /*Uses time domain convolution rounding to the nearest HRTF in the database*/
	void interpolateConvolve(); /*Uses Belloch's technique of interpolation*/
	void gpuTDConvolve(float* input, float* d_output, int outputLen, float gain, cudaStream_t* streams);
	void GPUSoundSource::caseOneConvolve(float* d_input, float* d_output, cufftComplex* d_convbufs, cufftComplex* d_distance_factor, cudaStream_t* streams, int* hrtf_indices);
	void GPUSoundSource::caseTwoConvolve(float* d_input, float* d_output,
		cufftComplex* d_convbufs, cufftComplex* d_distance_factor,
		cudaStream_t* streams, int* hrtf_indices, float* omegas);
	void GPUSoundSource::caseThreeConvolve(float* d_input, float* d_output,
		cufftComplex* d_convbufs, cufftComplex* d_distance_factor,
		cudaStream_t* streams, int* hrtf_indices, float* omegas);
	void GPUSoundSource::caseFourConvolve(float* d_input, float* d_output,
		cufftComplex* d_convbufs, cufftComplex* d_distance_factor,
		cudaStream_t* streams, int* hrtf_indices, float* omegas);
	void allKernels(float* d_input, float* d_output, cufftComplex* d_convbufs, cufftComplex* d_distance_factor, cudaStream_t* streams, float* omegas, int* hrtf_indices, cudaEvent_t fft_in); /*All of the kernels for interpolation*/
	void gpuCalculateDistanceFactor(cudaStream_t stream);
	void gpuCalculateDistanceFactor();
	cufftHandle plans[3];
	cudaEvent_t incomingTransfers[3];
	cudaEvent_t fft_events;
	cudaEvent_t kernel_launches[STREAMS_PER_FLIGHT];

};
#endif
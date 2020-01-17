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
	int blockNo;
};
class GPUSoundSource : CPUSoundSource {
public:

	cudaStream_t streams[FLIGHT_NUM * STREAMS_PER_FLIGHT];			/*Streams associated with each block in flight*/
	float* intermediate[FLIGHT_NUM];			/*Host data of the output*/
	void process(int blockNo, processes type); /*Wrapper for the processor convolver. Calls fftConvolve() or interpolateConvolve()*/
	GPUSoundSource();
	~GPUSoundSource();
	
//private:
	float* x[FLIGHT_NUM];						/*Buffer for PA output on host, mono, pinned memory, PAD_LEN + 2 size */
	cufftComplex* distance_factor[FLIGHT_NUM]; 	/*Buffer for the complex factor to account for distance scaling PAD_LEN / 2 + 1 Complex values*/
	float* d_input[FLIGHT_NUM];					/*PAD_LEN + 2 sized for the input*/
	cufftComplex* d_convbufs[FLIGHT_NUM];		/*4 * 2 * (PAD_LEN + 2) sized for each interpolation buffer*/
	cufftComplex* d_convbufs2[FLIGHT_NUM];		/*4 * 2 * (PAD_LEN + 2) sized for each interpolation buffer, part of the crossfading functions*/
	float* d_output[FLIGHT_NUM];				/*2 * (PAD_LEN + 2) * 2  sized for the output, *2 for switching method*/
	float* d_output2[FLIGHT_NUM];				/*2 * (PAD_LEN + 2)  sized for the output, part of the crossfading function*/

	callback_data callback_data_blocks[FLIGHT_NUM * 3];
	void chunkProcess(int blockNo);
	void sendBlock(int blockNo);
	void overlapSave(int blockNo);
	void copyIncomingBlock(int blockNo);
	void receiveBlock(int blockNo);
	void fftConvolve(int blockNo); /*Uses time domain convolution rounding to the nearest HRTF in the database*/
	void interpolateConvolve(int blockNo); /*Uses Belloch's technique of interpolation*/
	void gpuTDConvolve(float* input, float* d_output, int outputLen, float gain, cudaStream_t* streams);
	void allKernels(float* d_input, float* d_output, cufftComplex* d_convbufs, cufftComplex* d_distance_factor, cudaStream_t* streams, float* omegas, int* hrtf_indices, cudaEvent_t fft_in); /*All of the kernels for interpolation*/
	void gpuCalculateDistanceFactor(int blockNo, cudaStream_t stream);
	void gpuCalculateDistanceFactor(int blockNo);
	cufftHandle plans[FLIGHT_NUM * 3];
	cudaEvent_t incomingTransfers[FLIGHT_NUM * 3];
	cudaEvent_t fft_events[FLIGHT_NUM];
	cudaEvent_t kernel_launches[STREAMS_PER_FLIGHT * FLIGHT_NUM];

};
#endif
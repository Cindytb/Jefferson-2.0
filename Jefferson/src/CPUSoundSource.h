#pragma once
#ifndef __CPU_SOUND_SOURCE__
#define __CPU_SOUND_SOURCE__

#include "SoundSource.cuh"
#include "Universal.cuh"
#include "VBO.cuh"
#include "hrtf_signals.cuh"
#include "functions.h"

#include <math.h>
#include <cmath>
#include <fftw3.h>
#include <omp.h>

class CPUSoundSource : public SoundSource {
public:
	CPUSoundSource();
    ~CPUSoundSource();
    void process(processes type);
	fftwf_complex* intermediate;			/*2 * (PAD_LEN / 2 + 1) complex values. Padded buffer for fftw output*/
	float* x;						/*Buffer for PA output on host, mono, pinned memory, PAD_LEN + 2 size */
//private:
	void cpuTDConvolve();
	void cpuFFTConvolve();
	
	void cpuInterpolateLoops(fftwf_complex* output, fftwf_complex* convbufs, int* hrtf_indices, float* omegas);
    
	fftwf_complex* conv_bufs;				/*2 * (PAD_LEN / 2 + 1) complex values. Buffer for interpolation, 8 of them total*/
	fftwf_complex* distance_factor;		/*Buffer for the complex factor to account for distance scaling PAD_LEN / 2 + 1 Complex values*/

	void cpuFFTInterpolate();
	void fftConvolve(int blockNo); /*Uses time domain convolution rounding to the nearest HRTF in the database*/
	void interpolateConvolve(int blockNo); /*Uses Belloch's technique of interpolation*/

	void calculateDistanceFactor();
	
	fftwf_plan in_plan, out_plan; 
};
#endif
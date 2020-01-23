#include "CPUSoundSource.h"


CPUSoundSource::CPUSoundSource() {
	x = fftwf_alloc_real(PAD_LEN + 2);
	for (int i = 0; i < PAD_LEN + 2; i++) {
		x[i] = 0.0f;
	}
	intermediate = fftwf_alloc_complex(4UL * (PAD_LEN + 2));
	conv_bufs = fftwf_alloc_complex(8UL * (PAD_LEN + 2));
	distance_factor = fftwf_alloc_complex(PAD_LEN / 2 + 1);

	int n[] = { (int)PAD_LEN };
	in_plan = fftwf_plan_dft_r2c_1d((int)PAD_LEN, x, intermediate, FFTW_MEASURE);
	out_plan = fftwf_plan_many_dft_c2r(
		1, n, 2,
		intermediate, NULL,
		1, PAD_LEN / 2 + 1,
		(float*)intermediate, NULL,
		2, 1, FFTW_MEASURE
	);

}

/*
	R(r) = (1 / (1 + (fs / vs) (r - r0)^2) ) * e^ ((-j2PI (fs/vs) * (r - r0) *k) / N)
			|----------FRAC-----------------|	  |------------exponent------------------|

	FRAC * e^(exponent)
	FRAC * (cosine(exponent) - sine(exponent))
	R[r].x = FRAC * cosine(exponent)
	R[r].y = -FRAC * sine(exponent)
	*/
void CPUSoundSource::calculateDistanceFactor() {
	float r = std::sqrt(
		coordinates.x * coordinates.x +
		coordinates.y * coordinates.y +
		coordinates.z * coordinates.z
	);
	r /= 5;
	float fsvs = 44100.0 / 343.0;
	float frac = 1 + fsvs * pow(r, 2);
	int N = PAD_LEN / 2 + 1;
	//#pragma omp for
	for (int i = 0; i < N; i++) {
		distance_factor[i][0] = cos(2 * PI * fsvs * r * i / N) / frac;
		distance_factor[i][1] = -sin(2 * PI * fsvs * r * i / N) / frac;
	}
}
void CPUSoundSource::process(processes type) {
	hrtf_idx = pick_hrtf(ele, azi);
	switch (type) {
	case processes::CPU_TD:
		cpuTDConvolve();
		break;
	case processes::CPU_FD_BASIC:
		cpuFFTConvolve();
		break;
	case processes::CPU_FD_COMPLEX:
		cpuFFTInterpolate();
		break;

	}
}

void CPUSoundSource::cpuTDConvolve() {
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start);
	float* input = x + PAD_LEN + 2 - FRAMES_PER_BUFFER;
	float* output = (float*)intermediate;
	float outputLen = FRAMES_PER_BUFFER;
	float gain = 1;
	float* l_hrtf = hrtf + (size_t) hrtf_idx * 2UL * (PAD_LEN + 2);
	float* r_hrtf = hrtf + (size_t) hrtf_idx * 2UL * (PAD_LEN + 2) + PAD_LEN + 2;
	if (gain > 1)
		gain = 1;

	/* zero output buffer */
	//#pragma omp for
	for (int i = 0; i < outputLen * 2UL; i++) {
		output[i] = 0.0;
	}
	for (int n = 0; n < outputLen; n++) {
		for (int k = 0; k < HRTF_LEN; k++) {
			for (int j = 0; j < 2UL; j++) {
				/* outputLen and HRTF_LEN are n frames, output and hrtf are interleaved
				* input is mono
				*/
				if (j == 0) {
					output[2 * n + j] += input[n - k] * l_hrtf[k];
				}
				else {
					output[2 * n + j] += input[n - k] * r_hrtf[k];
				}

			}
			output[2 * n] *= gain;
			output[2 * n + 1] *= gain;
		}
	}
	num_calls++;
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	//float milliseconds = 0;
	//cudaEventElapsedTime(&milliseconds, start, stop);
	//sum_ms += milliseconds;
	//num_iterations++;
	//avg_ms = sum_ms / float(num_iterations);
	//fprintf(stderr, "Average CPU Time Domain Kernel Time: %f\n", avg_ms);
}
void CPUSoundSource::cpuFFTConvolve() {
	cudaEvent_t start, stop;
	/*cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);*/
	fftwf_execute(in_plan); /*FFT on x --> intermediate*/
	complexScaling(intermediate, 1.0 / PAD_LEN, PAD_LEN / 2 + 1);
	/*Copying over for both channels*/
#pragma omp parallel for
	for (int i = 0; i < PAD_LEN / 2 + 1; i++) {
		intermediate[i + PAD_LEN / 2 + 1][0] = intermediate[i][0];
		intermediate[i + PAD_LEN / 2 + 1][1] = intermediate[i][1];
	}
	/*Doing both channels at once since they're contiguous in memory*/
	pointwiseMultiplication(intermediate,
		fft_hrtf + (size_t)hrtf_idx * 2UL * (PAD_LEN / 2 + 1),
		PAD_LEN + 2);
	fftwf_execute(out_plan);

	num_calls++;
	/*cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	sum_ms += milliseconds;
	num_iterations++;
	avg_ms = sum_ms / float(num_iterations);
	fprintf(stderr, "Average CPU Basic FD Kernel Time: %f\n", avg_ms);*/

}
void CPUSoundSource::caseOneConvolve(fftwf_complex* output, int* hrtf_indices) {
	int buf_size = PAD_LEN + 2;
	int complex_buf_size = buf_size / 2;
	pointwiseMultiplication(output,
		fft_hrtf + (size_t)hrtf_indices[0] * 2UL * complex_buf_size,
		buf_size);
	pointwiseMultiplication(
		output,
		distance_factor,
		complex_buf_size
	);
	pointwiseMultiplication(
		output + complex_buf_size,
		distance_factor,
		complex_buf_size
	);
}

void CPUSoundSource::caseTwoConvolve(fftwf_complex* output, fftwf_complex* convbufs, int* hrtf_indices, float* omegas) {
	size_t buf_size = PAD_LEN + 2;
	size_t complex_buf_size = buf_size / 2;
	pointwiseMultiplication(output,
		fft_hrtf + (size_t)hrtf_indices[0] * 2UL * complex_buf_size,
		convbufs,
		buf_size
	);
	pointwiseMultiplication(output,
		fft_hrtf + (size_t)hrtf_indices[1] * 2UL * complex_buf_size,
		convbufs + buf_size,
		buf_size
	);
	complexScaling(convbufs, omegas[1], buf_size);
	complexScaling(convbufs + buf_size, omegas[0], buf_size);
	for (unsigned int i = 0; i < 4; i++) {
		pointwiseMultiplication(
			convbufs + complex_buf_size * i,
			distance_factor,
			complex_buf_size
		);
	}
	pointwiseAddition(
		convbufs,
		convbufs + buf_size,
		output,
		buf_size);
}
void CPUSoundSource::caseThreeConvolve(fftwf_complex* output, fftwf_complex* convbufs, int* hrtf_indices, float* omegas) {
	size_t buf_size = PAD_LEN + 2;
	size_t complex_buf_size = buf_size / 2;
	pointwiseMultiplication(output,
		fft_hrtf + (size_t)hrtf_indices[0] * 2UL * complex_buf_size,
		convbufs,
		buf_size
	);
	pointwiseMultiplication(output,
		fft_hrtf + (size_t)hrtf_indices[2] * 2UL * complex_buf_size,
		convbufs + buf_size,
		buf_size
	);
	complexScaling(convbufs, omegas[5], buf_size);
	complexScaling(convbufs + buf_size, omegas[4], buf_size);
	for (int i = 0; i < 4; i++) {
		pointwiseMultiplication(
			convbufs + complex_buf_size * i,
			distance_factor,
			complex_buf_size
		);
	}
	pointwiseAddition(
		convbufs,
		convbufs + (buf_size),
		output,
		buf_size);
}
void CPUSoundSource::caseFourConvolve(fftwf_complex* output, fftwf_complex* convbufs, int* hrtf_indices, float* omegas) {
	size_t buf_size = PAD_LEN + 2;
	size_t complex_buf_size = buf_size / 2;
#pragma omp parallel for
	for (int i = 0; i < 4; i++) {
		pointwiseMultiplication(
			output,
			fft_hrtf + (size_t)hrtf_indices[i] * 2UL * complex_buf_size,
			convbufs + buf_size * i,
			buf_size
		);
		pointwiseMultiplication(
			convbufs + buf_size * i,
			distance_factor,
			complex_buf_size
		);
		pointwiseMultiplication(
			convbufs + buf_size * i + complex_buf_size,
			distance_factor,
			complex_buf_size
		);
	}
	complexScaling(convbufs, omegas[5] * omegas[1], buf_size);
	complexScaling(convbufs + buf_size, omegas[5] * omegas[0], buf_size);
	complexScaling(convbufs + 2UL * buf_size, omegas[4] * omegas[3], buf_size);
	complexScaling(convbufs + 3UL * buf_size, omegas[4] * omegas[2], buf_size);

	pointwiseAddition(
		convbufs,
		convbufs + (buf_size),
		output,
		buf_size);
	for (unsigned i = 2; i < 4; i++) {
		pointwiseAddition(output,
			convbufs + buf_size * i,
			buf_size);
	}
}
void CPUSoundSource::cpuInterpolateLoops(fftwf_complex* output, fftwf_complex* convbufs, int* hrtf_indices, float* omegas) {
	int buf_size = PAD_LEN + 2;
	int complex_buf_size = buf_size / 2;
	if (hrtf_indices[0] == hrtf_indices[1] && hrtf_indices[1] == hrtf_indices[2] && hrtf_indices[2] == hrtf_indices[3]) {
		caseOneConvolve(output, hrtf_indices);
	}
	/*If the elevation falls on the resolution, interpolate the azimuth*/
	else if (hrtf_indices[0] == hrtf_indices[2]) {
		caseTwoConvolve(output, convbufs, hrtf_indices, omegas);
	}
	/*If the azimuth falls on the resolution, interpolate the elevation*/
	else if (hrtf_indices[0] == hrtf_indices[1] && hrtf_indices[0] != hrtf_indices[2]) {
		caseThreeConvolve(output, convbufs, hrtf_indices, omegas);
	}
	/*Worst case scenario*/
	else {
		caseFourConvolve(output, convbufs, hrtf_indices, omegas);
	}
}
void CPUSoundSource::cpuFFTInterpolate() {
	/*cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);*/
	fftwf_execute(in_plan); /*FFT on x --> intermediate*/
	complexScaling(intermediate, 1.0 / PAD_LEN, PAD_LEN / 2 + 1);
	/*Copying over for both channels*/
	memcpy(
		intermediate + PAD_LEN / 2 + 1,
		intermediate,
		(PAD_LEN / 2 + 1) * sizeof(fftwf_complex)
	);

	int old_hrtf_indices[4];
	float old_omegas[6];
	interpolationCalculations(ele, azi, hrtf_indices, omegas);
	bool xfade = false;
	if (old_azi != azi || old_ele != ele) {
		xfade = true;
		interpolationCalculations(old_ele, old_azi, old_hrtf_indices, old_omegas);
		memcpy(
			intermediate + PAD_LEN + 2,
			intermediate,
			(PAD_LEN + 2) * sizeof(fftwf_complex)
		);
	}
	calculateDistanceFactor();
	if (!xfade) {
		cpuInterpolateLoops(intermediate, conv_bufs, hrtf_indices, omegas);
		fftwf_execute(out_plan);
	}
	else{
		cpuInterpolateLoops(intermediate, conv_bufs, old_hrtf_indices, old_omegas);
		fftwf_execute(out_plan);
		cpuInterpolateLoops(intermediate + PAD_LEN + 2, conv_bufs + (PAD_LEN + 2) * 4, hrtf_indices, omegas);
		fftwf_execute_dft_c2r(
			out_plan,
			intermediate + (PAD_LEN + 2),
			(float*)(intermediate + PAD_LEN + 2)
		);
		float* out1 = ((float*)intermediate) + 2 * (PAD_LEN - FRAMES_PER_BUFFER);
		float* out2 = ((float*)intermediate) + 2 * PAD_LEN + 4 + 2 * (PAD_LEN - FRAMES_PER_BUFFER);//PAD_LEN + 2 + 2 * (PAD_LEN - FRAMES_PER_BUFFER);
		
		for (int i = 0; i < FRAMES_PER_BUFFER; i++) {
			float fn = float(i) / (FRAMES_PER_BUFFER - 1.0f);
			float old = out1[i * 2];
			out1[i * 2] = out1[i * 2] * (1.0f - fn) + out2[i * 2] * fn;
			out1[i * 2 + 1] = out1[i * 2 + 1] * (1.0f - fn) + out2[i * 2 + 1] * fn;
			
		}
	}
	num_calls++;

	old_azi = azi;
	old_ele = ele;
	/*cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	fprintf(stderr, "Kernel Time Time: %f\n", milliseconds);
	sum_ms += milliseconds;
	num_iterations++;
	avg_ms = sum_ms / float(num_iterations);
	fprintf(stderr, "Average CPU FD Kernel Time: %f\n", avg_ms);*/
}
CPUSoundSource::~CPUSoundSource() {
	fftwf_free(x);
	fftwf_free(intermediate);
	fftwf_free(conv_bufs);
	fftwf_free(distance_factor);
}

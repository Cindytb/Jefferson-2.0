#include "CPUSoundSource.h"


CPUSoundSource::CPUSoundSource() {
	x = fftwf_alloc_real(PAD_LEN + 2);
	for (int i = 0; i < PAD_LEN + 2; i++) {
		x[i] = 0.0f;
	}
	intermediate = fftwf_alloc_complex(4 * (PAD_LEN + 2));
	conv_bufs = fftwf_alloc_complex(8 * (PAD_LEN + 2));
	distance_factor = fftwf_alloc_complex(PAD_LEN / 2 + 1);


	in_plan = fftwf_plan_dft_r2c_1d(PAD_LEN, x, intermediate, FFTW_ESTIMATE);
	out_plan = fftwf_plan_many_dft_c2r(
		1, &PAD_LEN, 2,
		intermediate, NULL,
		1, PAD_LEN / 2 + 1,
		(float*)intermediate, NULL,
		2, 1, FFTW_ESTIMATE
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
	float N = PAD_LEN / 2 + 1;
	//#pragma omp for
	for (int i = 0; i < N; i++) {
		distance_factor[i][0] = cos(2 * PI * fsvs * r * i / N) / frac;
		distance_factor[i][1] = -sin(2 * PI * fsvs * r * i / N) / frac;
	}
}
void CPUSoundSource::process(processes type) {
	hrtf_idx = pick_hrtf(ele, azi);
	switch (type) {
	case CPU_TD:
		cpuTDConvolve();
		break;
	case CPU_FD_BASIC:
		cpuFFTConvolve();
		break;
	case CPU_FD_COMPLEX:
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
	float* l_hrtf = hrtf + hrtf_idx * HRTF_CHN * (PAD_LEN + 2);
	float* r_hrtf = hrtf + hrtf_idx * HRTF_CHN * (PAD_LEN + 2) + PAD_LEN + 2;
	if (gain > 1)
		gain = 1;

	/* zero output buffer */
	//#pragma omp for
	for (int i = 0; i < outputLen * HRTF_CHN; i++) {
		output[i] = 0.0;
	}
	for (int n = 0; n < outputLen; n++) {
		for (int k = 0; k < HRTF_LEN; k++) {
			for (int j = 0; j < HRTF_CHN; j++) {
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
#pragma omp for
	for (int i = 0; i < PAD_LEN / 2 + 1; i++) {
		intermediate[i + PAD_LEN / 2 + 1][0] = intermediate[i][0];
		intermediate[i + PAD_LEN / 2 + 1][1] = intermediate[i][1];
	}
	/*Doing both channels at once since they're contiguous in memory*/
	pointwiseMultiplication(intermediate,
		fft_hrtf + hrtf_idx * HRTF_CHN * (PAD_LEN / 2 + 1),
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

void CPUSoundSource::cpuInterpolateLoops(fftwf_complex* output, fftwf_complex* convbufs, int* hrtf_indices, float* omegas) {
	if (hrtf_indices[0] == hrtf_indices[1] && hrtf_indices[1] == hrtf_indices[2] && hrtf_indices[2] == hrtf_indices[3]) {
		pointwiseMultiplication(output,
			fft_hrtf + hrtf_indices[0] * HRTF_CHN * (PAD_LEN / 2 + 1),
			PAD_LEN + 2);
		pointwiseMultiplication(
			output,
			distance_factor,
			PAD_LEN / 2 + 1
		);
		pointwiseMultiplication(
			output + PAD_LEN / 2 + 1,
			distance_factor,
			PAD_LEN / 2 + 1
		);
	}
	/*If the elevation falls on the resolution, interpolate the azimuth*/
	else if (hrtf_indices[0] == hrtf_indices[2]) {
		pointwiseMultiplication(output,
			fft_hrtf + hrtf_indices[0] * HRTF_CHN * (PAD_LEN / 2 + 1),
			convbufs,
			PAD_LEN + 2
		);
		pointwiseMultiplication(output,
			fft_hrtf + hrtf_indices[1] * HRTF_CHN * (PAD_LEN / 2 + 1),
			convbufs + PAD_LEN + 2,
			PAD_LEN + 2
		);
		complexScaling(convbufs, omegas[1], PAD_LEN + 2);
		complexScaling(convbufs + PAD_LEN + 2, omegas[0], PAD_LEN + 2);
		for (int i = 0; i < 4; i++) {
			pointwiseMultiplication(
				convbufs + (PAD_LEN / 2 + 1) * i,
				distance_factor,
				PAD_LEN / 2 + 1
			);
		}
		pointwiseAddition(
			convbufs,
			convbufs + PAD_LEN + 2,
			output,
			PAD_LEN + 2);


	}
	/*If the azimuth falls on the resolution, interpolate the elevation*/
	else if (hrtf_indices[0] == hrtf_indices[1] && hrtf_indices[0] != hrtf_indices[2]) {
		pointwiseMultiplication(output,
			fft_hrtf + hrtf_indices[0] * HRTF_CHN * (PAD_LEN / 2 + 1),
			convbufs,
			PAD_LEN + 2
		);
		pointwiseMultiplication(output,
			fft_hrtf + hrtf_indices[2] * HRTF_CHN * (PAD_LEN / 2 + 1),
			convbufs + PAD_LEN + 2,
			PAD_LEN + 2
		);
		complexScaling(convbufs, omegas[5], PAD_LEN + 2);
		complexScaling(convbufs + PAD_LEN + 2, omegas[4], PAD_LEN + 2);
		for (int i = 0; i < 4; i++) {
			pointwiseMultiplication(
				convbufs + (PAD_LEN / 2 + 1) * i,
				distance_factor,
				PAD_LEN / 2 + 1
			);
		}
		pointwiseAddition(
			convbufs,
			convbufs + (PAD_LEN + 2),
			output,
			PAD_LEN + 2);
	}
	/*Worst case scenario*/
	else {
#pragma omp parallel for
		for (int i = 0; i < 4; i++) {
			pointwiseMultiplication(
				output,
				fft_hrtf + hrtf_indices[i] * HRTF_CHN * (PAD_LEN / 2 + 1),
				convbufs + (PAD_LEN + 2) * i,
				PAD_LEN + 2
			);
			pointwiseMultiplication(
				output,
				fft_hrtf + hrtf_indices[0] * HRTF_CHN * (PAD_LEN / 2 + 1),
				convbufs,
				PAD_LEN + 2
			);
			pointwiseMultiplication(
				convbufs + (PAD_LEN + 2) * i,
				distance_factor,
				PAD_LEN / 2 + 1
			);
			pointwiseMultiplication(
				convbufs + (PAD_LEN + 2) * i + PAD_LEN / 2 + 1,
				distance_factor,
				PAD_LEN / 2 + 1
			);
		}
		complexScaling(convbufs, omegas[5] * omegas[1], PAD_LEN + 2);
		complexScaling(convbufs + PAD_LEN + 2, omegas[5] * omegas[0], PAD_LEN + 2);
		complexScaling(convbufs + 2 * (PAD_LEN + 2), omegas[4] * omegas[3], PAD_LEN + 2);
		complexScaling(convbufs + 3 * (PAD_LEN + 2), omegas[4] * omegas[2], PAD_LEN + 2);

		pointwiseAddition(
			convbufs,
			convbufs + (PAD_LEN + 2),
			output,
			PAD_LEN + 2);
		for (int i = 2; i < 4; i++) {
			pointwiseAddition(output,
				convbufs + (PAD_LEN + 2) * i,
				PAD_LEN + 2);
		}
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
	cpuInterpolateLoops(intermediate, conv_bufs, hrtf_indices, omegas);
	fftwf_execute(out_plan);

	if (xfade) {
		cpuInterpolateLoops(intermediate + PAD_LEN + 2, conv_bufs + (PAD_LEN + 2) * 4, old_hrtf_indices, old_omegas);
		fftwf_execute_dft_c2r(
			out_plan,
			intermediate + (PAD_LEN + 2),
			(float*)(intermediate + PAD_LEN + 2)
		);
		float* out1 = ((float*)intermediate) + 2 * (PAD_LEN - FRAMES_PER_BUFFER);
		float* out2 = ((float*)intermediate) + 2 * PAD_LEN + 4 + 2 * (PAD_LEN - FRAMES_PER_BUFFER);//PAD_LEN + 2 + 2 * (PAD_LEN - FRAMES_PER_BUFFER);
#pragma omp parallel for
		for (int i = 0; i < FRAMES_PER_BUFFER; i++) {
			float fn = float(i) / (FRAMES_PER_BUFFER - 1.0f);
			out1[i * 2] = out1[i * 2] * fn + out2[i * 2] * (1.0f - fn);
			out1[i * 2 + 1] = out1[i * 2 + 1] * fn + out2[i * 2 + 1] * (1.0f - fn);
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

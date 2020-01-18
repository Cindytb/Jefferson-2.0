#include "GPUSoundSource.cuh"

void callback_memcpy(float* src, float* dest, size_t size) {
	memcpy(
		dest,
		src,
		sizeof(float) * size
	);
}

void CUDART_CB memcpyCallback(void* data) {
	// Check status of GPU after stream operations are done
	callback_data* p = (callback_data*)(data);
	callback_memcpy(p->input, p->output, p->size);
}

GPUSoundSource::GPUSoundSource() {
	for (int i = 0; i < FLIGHT_NUM; i++) {
		checkCudaErrors(cudaEventCreateWithFlags(&incomingTransfers[i * 3], cudaEventDisableTiming));
		checkCudaErrors(cudaEventCreateWithFlags(&incomingTransfers[i * 3 + 1], cudaEventDisableTiming));
		checkCudaErrors(cudaEventCreateWithFlags(&incomingTransfers[i * 3 + 2], cudaEventDisableTiming));
		for (int j = 0; j < STREAMS_PER_FLIGHT; j++) {
			checkCudaErrors(cudaEventCreateWithFlags(kernel_launches + j, cudaEventDisableTiming));
		}
		checkCudaErrors(cudaEventCreateWithFlags(fft_events + i, cudaEventDisableTiming));
		
		/*Allocating pinned memory for incoming transfer*/
		checkCudaErrors(cudaMallocHost(x + i, (PAD_LEN + 2) * sizeof(float)));
		/*Allocating memory for the inputs*/
		checkCudaErrors(cudaMalloc(d_input + i, 2 * (PAD_LEN + 2) * sizeof(float)));
		/*Allocating memory for the outputs*/
		checkCudaErrors(cudaMalloc(d_output + i, HRTF_CHN * (PAD_LEN + 2) * sizeof(float)));
		checkCudaErrors(cudaMalloc(d_output2 + i, HRTF_CHN * (PAD_LEN + 2) * sizeof(float)));
		/*Allocating memory for the distance factor*/
		checkCudaErrors(cudaMalloc(distance_factor + i, (PAD_LEN / 2 + 1) * sizeof(cufftComplex)));
		/*Creating the streams*/
		for(int j = 0; j < STREAMS_PER_FLIGHT; j++){
			checkCudaErrors(cudaStreamCreate(streams + i * STREAMS_PER_FLIGHT + j));
		}
		checkCudaErrors(cudaMalloc(d_convbufs + i, 8 * HRTF_CHN * (PAD_LEN / 2 + 1) * sizeof(cufftComplex)));
		/*Allocating pinned memory for outgoing transfer*/
		checkCudaErrors(cudaMallocHost(intermediate + i, (FRAMES_PER_BUFFER * HRTF_CHN) * sizeof(float)));

		CHECK_CUFFT_ERRORS(cufftPlan1d(plans + i * 3, PAD_LEN, CUFFT_R2C, 1));
		/*cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n,
			int *inembed, int istride, int idist,
			int *onembed, int ostride, int odist,
			cufftType type, int batch);*/
			/*stride = skip length for interleaving. Ex 1 = every element, 2 = every other element*/
			/*idist/odist is number of elements between batches of transforms, in terms of float or complex depending on the input/output*/
			/*inembed/onembed are for 2D/3D, num elements per dimension*/
		/*This type of cufft plan will take 2 mono channels located contiguously in memory, take the IFFT, and interleave them*/
		int n = PAD_LEN;
		CHECK_CUFFT_ERRORS(
			cufftPlanMany(
				plans + i * 3 + 1, 1, &n,
				&n, 1, n / 2 + 1,
				&n, 2, 1,
				CUFFT_C2R, 2)
		);
		CHECK_CUFFT_ERRORS(
			cufftPlanMany(
				plans + i * 3 + 2, 1, &n,
				&n, 1, n / 2 + 1,
				&n, 2, 1,
				CUFFT_C2R, 2)
		);
	}
	for (int i = 0; i < FLIGHT_NUM; i++) {
		for (int j = 0; j < PAD_LEN + 2; j++) {
			x[i][j] = 0.0f;
		}
	}
}
/*
	R(r) = (1 / (1 + (fs / vs) (r - r0)^2) ) * e^ ((-j2PI (fs/vs) * (r - r0) *k) / N)
			|----------FRAC-----------------|	  |------------exponent------------------|

	FRAC * e^(exponent)
	FRAC * (cosine(exponent) - sine(exponent))
	R[r].x = FRAC * cosine(exponent)
	R[r].y = -FRAC * sine(exponent)
	*/
void GPUSoundSource::gpuCalculateDistanceFactor(int blockNo, cudaStream_t stream){
	cufftComplex* d_distance_factor = this->distance_factor[blockNo];
	float r = std::sqrt(
		coordinates.x * coordinates.x + 
		coordinates.y * coordinates.y + 
		coordinates.z * coordinates.z
	);
	r /= 5;
	float fsvs = 44100.0 / 343.0;
	float frac = 1 + fsvs * pow(r, 2);
	float N = PAD_LEN / 2 + 1;
	int numThreads = 64;
	int numBlocks = (PAD_LEN / 2 + numThreads ) / numThreads;
	generateDistanceFactor << < numThreads, numBlocks, 0, stream >> > (d_distance_factor, frac, fsvs, r, N);

}
void GPUSoundSource::gpuCalculateDistanceFactor(int blockNo) {
	cufftComplex* d_distance_factor = this->distance_factor[blockNo];
	float r = std::sqrt(
		coordinates.x * coordinates.x +
		coordinates.y * coordinates.y +
		coordinates.z * coordinates.z
	);
	r /= 5;
	float fsvs = 44100.0 / 343.0;
	float frac = 1 + fsvs * pow(r, 2);
	float N = PAD_LEN / 2 + 1;
	int numThreads = 64;
	int numBlocks = (PAD_LEN / 2 + numThreads) / numThreads;
	generateDistanceFactor << < numThreads, numBlocks>> > (d_distance_factor, frac, fsvs, r, N);

}

/*
This method is a slightly tweaked implementation of Jose Belloch's
"Headphone-Based Virtual Spatialization of Sound with a GPU Accelerator"
paper from the Journal of the Audio Engineering Society,
Volume 61, No 7/8, 2013, July/August
*/
//void GPUSoundSource::allKernels(float* d_input, float* d_output, 
//	cufftComplex* d_convbufs, cufftComplex* d_distance_factor, 
//	cudaStream_t* streams, float* omegas, int* hrtf_indices, cudaEvent_t fft_in){
//	
//
//	float scale = 1.0f / ((float)PAD_LEN);
//	
//	int numThreads = 64;
//	int numBlocks = (PAD_LEN / 2 + numThreads) / numThreads;
//	size_t buf_size = PAD_LEN + 2;
//	for (int i = 0; i < 8; i++) {
//		checkCudaErrors(cudaStreamWaitEvent(streams[i], fft_in, 0));
//	}
//	/*The azi & ele falls exactly on an hrtf resolution*/
//	if (hrtf_indices[0] == hrtf_indices[1] && hrtf_indices[1] == hrtf_indices[2] && hrtf_indices[2] == hrtf_indices[3]) {
//		/*+ Theta 1 Left*/
//		ComplexPointwiseMulAndScaleOutPlace << < numBlocks, numThreads, 0, streams[0] >> > (
//			(cufftComplex*)d_input,
//			(cufftComplex*)(d_hrtf + hrtf_indices[0] * (PAD_LEN + 2) * HRTF_CHN),
//			d_convbufs,
//			PAD_LEN / 2 + 1,
//			scale
//			);
//		ComplexPointwiseMulInPlace << < numBlocks, numThreads, 0, streams[0] >> > (
//			d_distance_factor, 
//			d_convbufs, 
//			PAD_LEN / 2 + 1
//			);
//		ComplexPointwiseAdd << < numBlocks, numThreads, 0, streams[0] >> > (
//			d_convbufs,
//			(cufftComplex*)d_output,
//			PAD_LEN / 2 + 1
//			);
//		/*+ Theta 1 Right*/
//		ComplexPointwiseMulAndScaleOutPlace << < numBlocks, numThreads, 0, streams[1] >> > (
//			(cufftComplex*)d_input,
//			(cufftComplex*)(d_hrtf + hrtf_indices[0] * (PAD_LEN + 2) * HRTF_CHN + PAD_LEN + 2),
//			d_convbufs + buf_size / 2,
//			PAD_LEN / 2 + 1,
//			scale
//			);
//		ComplexPointwiseMulInPlace << < numBlocks, numThreads, 0, streams[1] >> > (
//			d_distance_factor, 
//			d_convbufs + buf_size / 2, 
//			PAD_LEN / 2 + 1
//			);
//		
//		ComplexPointwiseAdd << < numBlocks, numThreads, 0, streams[1] >> > (
//			d_convbufs + buf_size / 2,
//			(cufftComplex*)(d_output + buf_size),
//			PAD_LEN / 2 + 1
//			);
//	}
//	/*If the elevation falls on the resolution, interpolate the azimuth*/
//	else if (hrtf_indices[0] == hrtf_indices[2]) {
//		for (int buf_no = 0; buf_no < 4; buf_no++) {
//			/*Even buf numbers are the left channel, odd ones are the right channel*/
//			float curr_scale;
//			int hrtf_index;
//			if (buf_no < 2){
//				curr_scale = scale * omegas[1];
//				hrtf_index = hrtf_indices[0];
//			}
//			else {
//				curr_scale = scale * omegas[0];
//				hrtf_index = hrtf_indices[1];
//			}
//			ComplexPointwiseMulAndScaleOutPlace << < numBlocks, numThreads, 0, streams[buf_no] >> > (
//				(cufftComplex*)d_input,
//				(cufftComplex*)(d_hrtf + hrtf_index * (PAD_LEN + 2) * HRTF_CHN + ((buf_no % 2) * (PAD_LEN + 2))),
//				d_convbufs + buf_size / 2 * buf_no,
//				PAD_LEN / 2 + 1,
//				curr_scale
//				);
//			ComplexPointwiseMulInPlace << < numBlocks, numThreads, 0, streams[buf_no] >> > (
//				d_distance_factor, 
//				d_convbufs + buf_size / 2 * buf_no,
//				PAD_LEN / 2 + 1
//				);
//			ComplexPointwiseAdd << < numBlocks, numThreads, 0, streams[buf_no] >> > (
//				d_convbufs + buf_size / 2 * buf_no,
//				(cufftComplex*)(d_output + buf_size * (buf_no % 2)),
//				PAD_LEN / 2 + 1
//				);
//		}
//
//	}
//	/*If the azimuth falls on the resolution, interpolate the elevation*/
//	else if (hrtf_indices[0] == hrtf_indices[1] && hrtf_indices[0] != hrtf_indices[2]) {
//		for (int buf_no = 0; buf_no < 4; buf_no++) {
//			/*Even buf numbers are the left channel, odd ones are the right channel*/
//			float curr_scale;
//			int hrtf_index;
//			switch (buf_no) {
//				case 0:
//				case 1:
//					curr_scale = scale * omegas[5];
//					hrtf_index = hrtf_indices[0];
//					break;
//				case 2:
//				case 3:
//					curr_scale = scale * omegas[4];
//					hrtf_index = hrtf_indices[2];
//					break;
//			}
//
//			ComplexPointwiseMulAndScaleOutPlace << < numBlocks, numThreads, 0, streams[buf_no] >> > (
//				(cufftComplex*)d_input,
//				(cufftComplex*)(d_hrtf + hrtf_index * (PAD_LEN + 2) * HRTF_CHN + ((buf_no % 2) * (PAD_LEN + 2))),
//				d_convbufs + buf_size / 2 * buf_no,
//				PAD_LEN / 2 + 1,
//				curr_scale
//				);
//			ComplexPointwiseMulInPlace << < numBlocks, numThreads, 0, streams[buf_no] >> > (
//				d_distance_factor,
//				d_convbufs + buf_size / 2 * buf_no,
//				PAD_LEN / 2 + 1
//				);
//			ComplexPointwiseAdd << < numBlocks, numThreads, 0, streams[buf_no] >> > (
//				d_convbufs + buf_size / 2 * buf_no,
//				(cufftComplex*)(d_output + buf_size * (buf_no % 2)),
//				PAD_LEN / 2 + 1
//				);
//		}
//	}
//	/*Worst case scenario*/
//	else {
//		for (int buf_no = 0; buf_no < 8; buf_no++) {
//			/*Even buf numbers are the left channel, odd ones are the right channel*/
//			float curr_scale;
//			int hrtf_index = buf_no / 2;
//			switch (hrtf_index) {
//			case 0:
//				curr_scale = scale * omegas[5] * omegas[1];
//				break;
//			case 1:
//				curr_scale = scale * omegas[5] * omegas[0];
//				break;
//			case 2:
//				curr_scale = scale * omegas[4] * omegas[3];
//				break;
//			case 3:
//				curr_scale = scale * omegas[4] * omegas[2];
//				break;
//			}
//			hrtf_index = hrtf_indices[hrtf_index];
//			ComplexPointwiseMulAndScaleOutPlace << < numBlocks, numThreads, 0, streams[buf_no] >> > (
//				(cufftComplex*)d_input,
//				(cufftComplex*)(d_hrtf + hrtf_index * (PAD_LEN + 2) * HRTF_CHN + ((buf_no % 2) * (PAD_LEN + 2))),
//				d_convbufs + buf_size / 2 * buf_no,
//				PAD_LEN / 2 + 1,
//				curr_scale
//				);
//			ComplexPointwiseMulInPlace << < numBlocks, numThreads, 0, streams[buf_no] >> > (
//				d_distance_factor,
//				d_convbufs + buf_size / 2 * buf_no,
//				PAD_LEN / 2 + 1
//				);
//			ComplexPointwiseAdd << < numBlocks, numThreads, 0, streams[buf_no] >> > (
//				d_convbufs + buf_size / 2 * buf_no,
//				(cufftComplex*)(d_output + buf_size * (buf_no % 2)),
//				PAD_LEN / 2 + 1
//				);
//		}
//	}
//}
//
///*blockNo MUST be modulo FLIGHT_NUM*/
//void GPUSoundSource::interpolateConvolve(int blockNo) {
//	cufftComplex* d_distance_factor = this->distance_factor[blockNo];
//	float* d_input = this->d_input[blockNo];
//	float* d_output = this->d_output[blockNo];
//	float* d_output2 = this->d_output2[blockNo];
//	cufftComplex* d_convbufs = this->d_convbufs[blockNo];
//	cufftComplex* d_convbufs2 = this->d_convbufs[blockNo] + 8 * (PAD_LEN + 2);
//	cudaStream_t* streams = this->streams + blockNo * STREAMS_PER_FLIGHT;
//	
//	int old_hrtf_indices[4];
//	float old_omegas[6];
//	int hrtf_indices[4];
//	float omegas[6];
//	interpolationCalculations(ele, azi, hrtf_indices, omegas);
//	bool xfade = false;
//	if (old_azi != azi || old_ele != ele) {
//		xfade = true;
//		interpolationCalculations(old_ele, old_azi, old_hrtf_indices, old_omegas);
//	}
//	gpuCalculateDistanceFactor(blockNo, streams[1]);
//	fillWithZeroesKernel(d_output, 2 * (PAD_LEN + 2), streams[2]);
//	if (xfade) {
//		fillWithZeroesKernel(d_output2, 2 * (PAD_LEN + 2), streams[8]);
//	}
//
//	CHECK_CUFFT_ERRORS(cufftSetStream(plans[blockNo * 3], streams[0]));
//	CHECK_CUFFT_ERRORS(cufftSetStream(plans[blockNo * 3 + 1], streams[0]));
//	CHECK_CUFFT_ERRORS(cufftExecR2C(plans[blockNo * 3], (cufftReal*)d_input, (cufftComplex*)d_input));
//	checkCudaErrors(cudaEventRecord(fft_events[blockNo], streams[0]));
//	if (!xfade) {
//		allKernels(d_input, d_output, d_convbufs, d_distance_factor, streams, omegas, hrtf_indices, fft_events[blockNo]);
//		for (int i = 1; i < 8; i++) {
//			checkCudaErrors(cudaEventRecord(kernel_launches[blockNo * i], streams[i]));
//		}
//		for (int i = 1; i < 8; i++) {
//			checkCudaErrors(cudaStreamWaitEvent(streams[0], kernel_launches[blockNo * i], 0));
//		}
//		CHECK_CUFFT_ERRORS(cufftExecC2R(plans[blockNo * 3 + 1], (cufftComplex*)d_output, d_output));
//	}
//	else {
//		allKernels(d_input, d_output, d_convbufs, d_distance_factor, streams, old_omegas, old_hrtf_indices, fft_events[blockNo]);
//		allKernels(d_input, d_output2, d_convbufs2, d_distance_factor, streams + 8, omegas, hrtf_indices, fft_events[blockNo]);
//		for (int i = 0; i < 16; i++) {
//			checkCudaErrors(cudaEventRecord(kernel_launches[blockNo * i], streams[i]));
//		}
//		for (int i = 1; i < 8; i++) {
//			checkCudaErrors(cudaStreamWaitEvent(streams[0], kernel_launches[blockNo * i], 0));
//		}
//		for (int i = 9; i < 16; i++) {
//			checkCudaErrors(cudaStreamWaitEvent(streams[8], kernel_launches[blockNo * i], 0));
//		}
//		CHECK_CUFFT_ERRORS(cufftSetStream(plans[blockNo * 3 + 2], streams[8]));
//		CHECK_CUFFT_ERRORS(cufftExecC2R(plans[blockNo * 3 + 1], (cufftComplex*)d_output, d_output));
//		CHECK_CUFFT_ERRORS(cufftExecC2R(plans[blockNo * 3 + 2], (cufftComplex*)d_output2, d_output2));
//		checkCudaErrors(cudaEventRecord(fft_events[blockNo], streams[8]));
//		checkCudaErrors(cudaStreamWaitEvent(streams[0], fft_events[blockNo], 0));
//		int numThreads = FRAMES_PER_BUFFER;
//		crossFade << <numThreads, 1, 0, streams[0] >> > (
//			d_output + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
//			d_output2 + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
//			FRAMES_PER_BUFFER);
//	}
//	old_azi = azi;
//	old_ele = ele;
//}
void GPUSoundSource::allKernels(float* d_input, float* d_output,
	cufftComplex* d_convbufs, cufftComplex* d_distance_factor,
	cudaStream_t* streams, float* omegas, int* hrtf_indices, cudaEvent_t fft_in) {

	float scale = 1.0f / ((float)PAD_LEN);

	size_t buf_size = PAD_LEN + 2;
	size_t complex_buf_size = buf_size / 2;
	int numThreads = 128;
	int numBlocks = (buf_size + numThreads - 1) / numThreads;
	for (int i = 0; i < 8; i++) {
		checkCudaErrors(cudaStreamWaitEvent(streams[i], fft_in, 0));
	}
	
	MyFloatScale << < numBlocks, numThreads >> > (d_input, scale, buf_size);
	checkCudaErrors(cudaMemcpy(d_input + buf_size, d_input, buf_size * sizeof(float), cudaMemcpyDeviceToDevice));
	/*The azi & ele falls exactly on an hrtf resolution*/
	if (hrtf_indices[0] == hrtf_indices[1] && hrtf_indices[1] == hrtf_indices[2] && hrtf_indices[2] == hrtf_indices[3]) {
		ComplexPointwiseMul << < numBlocks, numThreads >> > (
			(cufftComplex*)d_input,
			d_fft_hrtf + hrtf_indices[0] * complex_buf_size * HRTF_CHN,
			d_convbufs,
			buf_size
			);
		ComplexPointwiseMulInPlace << < numBlocks, numThreads >> > (
			d_distance_factor,
			d_convbufs,
			complex_buf_size
			);
		ComplexPointwiseMulInPlace << < numBlocks, numThreads >> > (
			d_distance_factor,
			d_convbufs + complex_buf_size,
			complex_buf_size
			);
		ComplexPointwiseAdd << < numBlocks, numThreads >> > (
			d_convbufs,
			(cufftComplex*)d_output,
			buf_size
			);
	}
	/*If the elevation falls on the resolution, interpolate the azimuth*/
	else if (hrtf_indices[0] == hrtf_indices[2] && hrtf_indices[1] == hrtf_indices[3]) {
		for (int buf_no = 0; buf_no < 2; buf_no++) {
			float curr_scale;
			int hrtf_index;
			switch (buf_no) {
			case 0:
				curr_scale = omegas[1];
				hrtf_index = hrtf_indices[0];
				break;
			case 1:
				curr_scale = omegas[0];
				hrtf_index = hrtf_indices[1];
				break;
			}
			ComplexPointwiseMulAndScaleOutPlace << < numBlocks, numThreads >> > (
				(cufftComplex*)d_input,
				d_fft_hrtf + hrtf_index * complex_buf_size * HRTF_CHN,
				d_convbufs + buf_size * buf_no,
				curr_scale,
				buf_size
				);
			ComplexPointwiseMulInPlace << < numBlocks, numThreads >> > (
				d_distance_factor,
				d_convbufs + buf_size * buf_no,
				complex_buf_size
				);
			ComplexPointwiseMulInPlace << < numBlocks, numThreads >> > (
				d_distance_factor,
				d_convbufs + buf_size * buf_no + complex_buf_size,
				complex_buf_size
				);
			ComplexPointwiseAdd << < numBlocks, numThreads >> > (
				d_convbufs + buf_size * buf_no,
				(cufftComplex*)(d_output),
				buf_size
				);
		}

	}
	/*If the azimuth falls on the resolution, interpolate the elevation*/
	else if (hrtf_indices[0] == hrtf_indices[1] && hrtf_indices[0] != hrtf_indices[2]) {
		for (int buf_no = 0; buf_no < 2; buf_no++) {
			float curr_scale;
			int hrtf_index;
			switch (buf_no) {
			case 0:
				curr_scale = omegas[5];
				hrtf_index = hrtf_indices[0];
				break;
			case 1:
				curr_scale = omegas[4];
				hrtf_index = hrtf_indices[2];
				break;
			}
			ComplexPointwiseMulAndScaleOutPlace << < numBlocks, numThreads >> > (
				(cufftComplex*)d_input,
				d_fft_hrtf + hrtf_index * complex_buf_size * HRTF_CHN,
				d_convbufs + buf_size * buf_no,
				curr_scale,
				buf_size
				);
			ComplexPointwiseMulInPlace << < numBlocks, numThreads >> > (
				d_distance_factor,
				d_convbufs + buf_size * buf_no,
				complex_buf_size
				);
			ComplexPointwiseMulInPlace << < numBlocks, numThreads >> > (
				d_distance_factor,
				d_convbufs + buf_size * buf_no + complex_buf_size,
				complex_buf_size
				);
			ComplexPointwiseAdd << < numBlocks, numThreads >> > (
				d_convbufs + buf_size * buf_no,
				(cufftComplex*)(d_output),
				buf_size
				);
		}
	}
	/*Worst case scenario*/
	else {
		for (int buf_no = 0; buf_no < 4; buf_no++) {
			/*Even buf numbers are the left channel, odd ones are the right channel*/
			float curr_scale;
			int hrtf_index = hrtf_indices[buf_no];
			switch (buf_no) {
			case 0:
				curr_scale = omegas[5] * omegas[1];
				break;
			case 1:
				curr_scale = omegas[5] * omegas[0];
				break;
			case 2:
				curr_scale = omegas[4] * omegas[3];
				break;
			case 3:
				curr_scale = omegas[4] * omegas[2];
				break;
			}
			ComplexPointwiseMulAndScaleOutPlace << < numBlocks, numThreads >> > (
				(cufftComplex*)d_input,
				d_fft_hrtf + hrtf_index * complex_buf_size * HRTF_CHN,
				d_convbufs + buf_size * buf_no,
				curr_scale,
				buf_size
				);
			ComplexPointwiseMulInPlace << < numBlocks, numThreads >> > (
				d_distance_factor,
				d_convbufs + buf_size * buf_no,
				complex_buf_size
				);
			ComplexPointwiseMulInPlace << < numBlocks, numThreads >> > (
				d_distance_factor,
				d_convbufs + buf_size * buf_no + complex_buf_size,
				complex_buf_size
				);
			ComplexPointwiseAdd << < numBlocks, numThreads >> > (
				d_convbufs + buf_size * buf_no,
				(cufftComplex*)(d_output),
				buf_size
				);
		}
	}
}

/*blockNo MUST be modulo FLIGHT_NUM*/
void GPUSoundSource::interpolateConvolve(int blockNo) {
	cufftComplex* d_distance_factor = this->distance_factor[blockNo];
	float* d_input = this->d_input[blockNo];
	float* d_output = this->d_output[blockNo];
	float* d_output2 = this->d_output2[blockNo];
	cufftComplex* d_convbufs = this->d_convbufs[blockNo];
	cufftComplex* d_convbufs2 = this->d_convbufs[blockNo] + 8 * (PAD_LEN + 2);
	cudaStream_t* streams = this->streams + blockNo * STREAMS_PER_FLIGHT;

	int old_hrtf_indices[4];
	float old_omegas[6];
	int hrtf_indices[4];
	float omegas[6];
	interpolationCalculations(ele, azi, hrtf_indices, omegas);
	bool xfade = false;
	if (old_azi != azi || old_ele != ele) {
		xfade = true;
		interpolationCalculations(old_ele, old_azi, old_hrtf_indices, old_omegas);
	}
	gpuCalculateDistanceFactor(blockNo);
	//gpuCalculateDistanceFactor(blockNo, streams[0]);
	fillWithZeroesKernel(d_output, 2 * (PAD_LEN + 2), streams[0]);
	//fillWithZeroes(&d_output, 0, 2 * (PAD_LEN + 2));
	if (xfade) {
		fillWithZeroesKernel(d_output2, 2 * (PAD_LEN + 2), streams[0]);
		//fillWithZeroes(&d_output2, 0, 2 * (PAD_LEN + 2));
	}
	checkCudaErrors(cudaDeviceSynchronize());
	/*CHECK_CUFFT_ERRORS(cufftSetStream(plans[blockNo * 3], streams[0]));
	CHECK_CUFFT_ERRORS(cufftSetStream(plans[blockNo * 3 + 1], streams[0]));
	CHECK_CUFFT_ERRORS(cufftSetStream(plans[blockNo * 3 + 2], streams[0]));*/
	CHECK_CUFFT_ERRORS(cufftExecR2C(plans[blockNo * 3], (cufftReal*)d_input, (cufftComplex*)d_input));
	
	if (!xfade) {
		allKernels(d_input, d_output, d_convbufs, d_distance_factor, streams, omegas, hrtf_indices, fft_events[blockNo]);
		CHECK_CUFFT_ERRORS(cufftExecC2R(plans[blockNo * 3 + 1], (cufftComplex*)d_output, d_output));

	}
	else {
		allKernels(d_input, d_output, d_convbufs, d_distance_factor, streams, old_omegas, old_hrtf_indices, fft_events[blockNo]);
		allKernels(d_input, d_output2, d_convbufs2, d_distance_factor, streams, omegas, hrtf_indices, fft_events[blockNo]);
		
		CHECK_CUFFT_ERRORS(cufftExecC2R(plans[blockNo * 3 + 1], (cufftComplex*)d_output, d_output));
		CHECK_CUFFT_ERRORS(cufftExecC2R(plans[blockNo * 3 + 2], (cufftComplex*)d_output2, d_output2));
		
		int numThreads = FRAMES_PER_BUFFER;
		crossFade << <numThreads, 1>> > (
			d_output + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
			d_output2 + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
			FRAMES_PER_BUFFER);
	}
	old_azi = azi;
	old_ele = ele;
}
void GPUSoundSource::fftConvolve(int blockNo) {
	cudaEvent_t start, stop;
	/*cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);*/

	float* d_input = this->d_input[blockNo % FLIGHT_NUM];
	float* d_output = this->d_output[blockNo % FLIGHT_NUM];
	cudaStream_t* streams = this->streams + (blockNo % FLIGHT_NUM) * STREAMS_PER_FLIGHT;

	float scale = 1.0f / ((float) PAD_LEN);
	CHECK_CUFFT_ERRORS(cufftSetStream(plans[blockNo * 3], streams[0]));
	CHECK_CUFFT_ERRORS(cufftSetStream(plans[blockNo * 3 + 1], streams[0]));
	CHECK_CUFFT_ERRORS(cufftExecR2C(plans[blockNo * 3], (cufftReal*)d_input, (cufftComplex*)d_input));
	int numThreads = 256;
	int numBlocks = (PAD_LEN / 2 + numThreads - 1) / numThreads;
	//interpolateConvolve(blockNo);
	ComplexPointwiseMulAndScaleOutPlace << < numBlocks, numThreads, 0, streams[0] >> > (
		(cufftComplex*)d_input,
		(cufftComplex*)(d_hrtf + hrtf_idx * (PAD_LEN + 2) * HRTF_CHN),
		(cufftComplex*)d_output,
		PAD_LEN / 2 + 1,
		scale
	);
	
	ComplexPointwiseMulAndScaleOutPlace << < numBlocks, numThreads, 0, streams[0] >> > (
		(cufftComplex*)d_input,
		(cufftComplex*)(d_hrtf + hrtf_idx * (PAD_LEN + 2) * HRTF_CHN + PAD_LEN + 2),
		(cufftComplex*)(d_output + PAD_LEN + 2),
		PAD_LEN / 2 + 1,
		scale
		);
	CHECK_CUFFT_ERRORS(cufftExecC2R(plans[blockNo * 3 + 1], (cufftComplex*)d_output, d_output));

}
void GPUSoundSource::gpuTDConvolve(float* input, float* d_output, int outputLen, float gain, cudaStream_t* streams) {
	if (gain > 1)
		gain = 1;
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start);
	int numThread = 128;
	int numBlocks = (outputLen + numThread - 1) / numThread;
	timeDomainConvolutionNaive << < numBlocks, numThread, 0, streams[0] >> > (
		input,
		d_hrtf + hrtf_idx * HRTF_CHN * (PAD_LEN + 2),
		d_output,
		outputLen,
		HRTF_LEN,
		0,
		gain);
	timeDomainConvolutionNaive << < numBlocks, numThread, 0, streams[0] >> > (
		input,
		d_hrtf + hrtf_idx * HRTF_CHN * (PAD_LEN + 2) + PAD_LEN + 2,
		d_output,
		outputLen,
		HRTF_LEN,
		1,
		gain);

}
void GPUSoundSource::sendBlock(int blockNo) {
	// cudaStream_t* streams = this->streams + blockNo * STREAMS_PER_FLIGHT;
	// checkCudaErrors(cudaStreamWaitEvent(streams[0], incomingTransfers[blockNo * 3], 0));
	// checkCudaErrors(cudaStreamWaitEvent(streams[0], incomingTransfers[blockNo * 3 + 1], 0));
	/*Send*/
	// checkCudaErrors(cudaMemcpyAsync(
	// 	d_input[blockNo],
	// 	x[blockNo],
	// 	PAD_LEN * sizeof(float),
	// 	cudaMemcpyHostToDevice,
	// 	streams[0])
	// );
	checkCudaErrors(cudaMemcpy(
		d_input[blockNo],
		x[blockNo],
		PAD_LEN * sizeof(float),
		cudaMemcpyHostToDevice)
	);
	
}
void GPUSoundSource::receiveBlock(int blockNo) {
	cudaStream_t* streams = this->streams + blockNo * STREAMS_PER_FLIGHT;
	// checkCudaErrors(cudaMemcpyAsync(
	// 	intermediate[blockNo],
	// 	d_output[blockNo] + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
	// 	FRAMES_PER_BUFFER * 2 * sizeof(float),
	// 	cudaMemcpyDeviceToHost,
	// 	streams[0])
	// );
	checkCudaErrors(cudaMemcpy(
		intermediate[blockNo],
		d_output[blockNo] + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
		FRAMES_PER_BUFFER * 2 * sizeof(float),
		cudaMemcpyDeviceToHost)
	);
}
void GPUSoundSource::chunkProcess(int blockNo) {
	copyIncomingBlock(blockNo % FLIGHT_NUM);
	overlapSave(blockNo);
	sendBlock(blockNo % FLIGHT_NUM);
	/*Process*/
	interpolateConvolve(blockNo % FLIGHT_NUM);
	/*Receive*/
	receiveBlock(blockNo % FLIGHT_NUM);
}
/*Must send un-modded block number*/
void GPUSoundSource::overlapSave(int blockNo) {
	if (blockNo == 0) {
		return;
	}
	int moddedBlockNo = blockNo % FLIGHT_NUM;
	checkCudaErrors(cudaStreamWaitEvent(streams[moddedBlockNo * STREAMS_PER_FLIGHT], incomingTransfers[((blockNo - 1) % FLIGHT_NUM) * 3], 0));
	checkCudaErrors(cudaStreamWaitEvent(streams[moddedBlockNo * STREAMS_PER_FLIGHT], incomingTransfers[((blockNo - 1) % FLIGHT_NUM) * 3 + 1], 0));
	/*Overlap-save, put the function in a stream*/
	
	callback_data_blocks[moddedBlockNo * 3 + 1].input = x[(blockNo - 1) % FLIGHT_NUM] + FRAMES_PER_BUFFER;
	callback_data_blocks[moddedBlockNo * 3 + 1].output = x[moddedBlockNo];
	callback_data_blocks[moddedBlockNo * 3 + 1].size = PAD_LEN - FRAMES_PER_BUFFER;
	cudaHostFn_t fn = memcpyCallback;
	checkCudaErrors(cudaLaunchHostFunc(streams[moddedBlockNo * STREAMS_PER_FLIGHT + 1], fn, &callback_data_blocks[moddedBlockNo * 3 + 1]));
	checkCudaErrors(cudaEventRecord(incomingTransfers[moddedBlockNo * 3 + 1], streams[moddedBlockNo * STREAMS_PER_FLIGHT + 1]));
	checkCudaErrors(cudaDeviceSynchronize());
	
}
/*Must send modulo'd block number*/
void GPUSoundSource::copyIncomingBlock(int blockNo) {
	/*Copy into curr_source->x pinned memory*/
	if (count + FRAMES_PER_BUFFER < length) {
		callback_data_blocks[blockNo * 3].input = buf + count;
		callback_data_blocks[blockNo * 3].output = x[blockNo] + (PAD_LEN - FRAMES_PER_BUFFER);
		callback_data_blocks[blockNo * 3].size = FRAMES_PER_BUFFER;
		callback_data_blocks[blockNo * 3].blockNo = blockNo;
		cudaHostFn_t fn = memcpyCallback;
		checkCudaErrors(cudaLaunchHostFunc(streams[blockNo * STREAMS_PER_FLIGHT], fn, &callback_data_blocks[blockNo * 3]));
		checkCudaErrors(cudaEventRecord(incomingTransfers[blockNo * 3], streams[blockNo * STREAMS_PER_FLIGHT]));
		count += FRAMES_PER_BUFFER;
	}
	else {
		int rem = length - count;
		callback_data_blocks[blockNo * 3].input = buf + count;
		callback_data_blocks[blockNo * 3].output = x[blockNo] + (PAD_LEN - FRAMES_PER_BUFFER);
		callback_data_blocks[blockNo * 3].size = rem;
		callback_data_blocks[blockNo * 3].blockNo = blockNo;
		cudaHostFn_t fn = memcpyCallback;
		checkCudaErrors(cudaLaunchHostFunc(streams[blockNo * STREAMS_PER_FLIGHT], fn, &callback_data_blocks[blockNo * 3]));
		memcpy(
			x[blockNo] + (PAD_LEN - FRAMES_PER_BUFFER),
			buf + count,
			rem * sizeof(float));
		memcpy(
			x[blockNo] + (PAD_LEN - FRAMES_PER_BUFFER) + rem,
			buf,
			(FRAMES_PER_BUFFER - rem) * sizeof(float));
		callback_data_blocks[blockNo * 3 + 2].input = buf;
		callback_data_blocks[blockNo * 3 + 2].output = x[blockNo] + (PAD_LEN - FRAMES_PER_BUFFER) + rem;
		callback_data_blocks[blockNo * 3 + 2].size = FRAMES_PER_BUFFER - rem;
		callback_data_blocks[blockNo * 3 + 2].blockNo = blockNo;
		fn = memcpyCallback;
		checkCudaErrors(cudaLaunchHostFunc(streams[blockNo * STREAMS_PER_FLIGHT], fn, &callback_data_blocks[blockNo * 3 + 2]));
		checkCudaErrors(cudaEventRecord(incomingTransfers[blockNo * 3], streams[blockNo * STREAMS_PER_FLIGHT]));
		count = FRAMES_PER_BUFFER - rem;
	}
	checkCudaErrors(cudaDeviceSynchronize());
}
void GPUSoundSource::process(int blockNo, processes type){
    switch(type){
        case GPU_FD_COMPLEX:
            chunkProcess(blockNo);
            break;
        case GPU_FD_BASIC:
            fftConvolve(blockNo);
            break;
        case GPU_TD:
            gpuTDConvolve(
                d_input[blockNo % FLIGHT_NUM] + PAD_LEN - FRAMES_PER_BUFFER,
                d_output[blockNo % FLIGHT_NUM] + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
                FRAMES_PER_BUFFER,
                gain, streams + blockNo * STREAMS_PER_FLIGHT);
                break;
    }

}
GPUSoundSource::~GPUSoundSource() {
	free(buf);
	for (int i = 0; i < FLIGHT_NUM; i++) {
		checkCudaErrors(cudaFreeHost(x[i]));
		checkCudaErrors(cudaFree(d_input[i]));
		checkCudaErrors(cudaFree(d_output[i]));
		checkCudaErrors(cudaFree(distance_factor[i]));
		for (int j = 0; j < STREAMS_PER_FLIGHT; j++) {
			checkCudaErrors(cudaStreamDestroy(streams[i * STREAMS_PER_FLIGHT + j]));
			checkCudaErrors(cudaEventDestroy(kernel_launches[i * FLIGHT_NUM + j]));
		}
		checkCudaErrors(cudaEventDestroy(fft_events[i]));
		checkCudaErrors(cudaFree(d_convbufs[i]));
		checkCudaErrors(cudaFreeHost(intermediate[i]));
		CHECK_CUFFT_ERRORS(cufftDestroy(plans[i * 3]));
		CHECK_CUFFT_ERRORS(cufftDestroy(plans[i * 3 + 1]));
		CHECK_CUFFT_ERRORS(cufftDestroy(plans[i * 3 + 2]));
	}
	delete[] streams;
	delete[] fft_events;
	delete[] kernel_launches;
}

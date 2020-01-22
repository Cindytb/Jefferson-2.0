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
	for (int i = 0; i < 3; i++) {
		checkCudaErrors(cudaEventCreateWithFlags(&incomingTransfers[i], cudaEventDisableTiming));
	}
	for (int j = 0; j < STREAMS_PER_FLIGHT; j++) {
		checkCudaErrors(cudaEventCreateWithFlags(kernel_launches + j, cudaEventDisableTiming));
	}
	checkCudaErrors(cudaEventCreateWithFlags(&fft_events, cudaEventDisableTiming));

	/*Allocating pinned memory for incoming transfer*/
	checkCudaErrors(cudaMallocHost(&x, (PAD_LEN + 2) * sizeof(float)));
	/*Allocating memory for the inputs*/
	checkCudaErrors(cudaMalloc(&d_input, 2 * (PAD_LEN + 2) * sizeof(float)));
	/*Allocating memory for the outputs*/
	checkCudaErrors(cudaMalloc(&d_output, HRTF_CHN * (PAD_LEN + 2) * sizeof(float)));
	checkCudaErrors(cudaMalloc(&d_output2, HRTF_CHN * (PAD_LEN + 2) * sizeof(float)));
	/*Allocating memory for the distance factor*/
	checkCudaErrors(cudaMalloc(&d_distance_factor, (PAD_LEN / 2 + 1) * sizeof(cufftComplex)));
	/*Creating the streams*/
	for (int j = 0; j < STREAMS_PER_FLIGHT; j++) {
		checkCudaErrors(cudaStreamCreate(streams + j));
	}
	checkCudaErrors(cudaMalloc(&d_convbufs, 8 * HRTF_CHN * (PAD_LEN / 2 + 1) * sizeof(cufftComplex)));
	/*Allocating pinned memory for outgoing transfer*/
	checkCudaErrors(cudaMallocHost(&intermediate, (FRAMES_PER_BUFFER * HRTF_CHN) * sizeof(float)));

	CHECK_CUFFT_ERRORS(cufftPlan1d(plans, PAD_LEN, CUFFT_R2C, 1));
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
			plans + 1, 1, &n,
			&n, 1, n / 2 + 1,
			&n, 2, 1,
			CUFFT_C2R, 2)
	);
	CHECK_CUFFT_ERRORS(
		cufftPlanMany(
			plans + 2, 1, &n,
			&n, 1, n / 2 + 1,
			&n, 2, 1,
			CUFFT_C2R, 2)
	);

	for (int j = 0; j < PAD_LEN + 2; j++) {
		x[j] = 0.0f;
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
void GPUSoundSource::gpuCalculateDistanceFactor(cudaStream_t stream) {
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
	generateDistanceFactor << < numThreads, numBlocks, 0, stream >> > (d_distance_factor, frac, fsvs, r, N);

}
void GPUSoundSource::gpuCalculateDistanceFactor() {
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
	generateDistanceFactor << < numThreads, numBlocks >> > (d_distance_factor, frac, fsvs, r, N);

}

/*
This method is a slightly tweaked implementation of Jose Belloch's
"Headphone-Based Virtual Spatialization of Sound with a GPU Accelerator"
paper from the Journal of the Audio Engineering Society,
Volume 61, No 7/8, 2013, July/August
*/
void GPUSoundSource::caseOneConvolve(float* d_input, float* d_output,
	cufftComplex* d_convbufs, cufftComplex* d_distance_factor,
	cudaStream_t* streams, int* hrtf_indices) {
	int buf_size = PAD_LEN + 2;
	int complex_buf_size = buf_size / 2;
	int numThreads = 64;
	int numBlocks = (buf_size + numThreads - 1) / numThreads;
	ComplexPointwiseMul << < numBlocks, numThreads, 0, streams[0] >> > (
		(cufftComplex*)d_input,
		d_fft_hrtf + hrtf_indices[0] * complex_buf_size * HRTF_CHN,
		d_convbufs,
		buf_size
		);
	numBlocks = (complex_buf_size + numThreads - 1) / numThreads;
	ComplexPointwiseMulInPlace << < numBlocks, numThreads, 0, streams[0] >> > (
		d_distance_factor,
		d_convbufs,
		complex_buf_size
		);
	ComplexPointwiseMulInPlace << < numBlocks, numThreads, 0, streams[0] >> > (
		d_distance_factor,
		d_convbufs + complex_buf_size,
		complex_buf_size
		);
	numBlocks = (buf_size + numThreads - 1) / numThreads;
	ComplexPointwiseAdd << < numBlocks, numThreads, 0, streams[0] >> > (
		d_convbufs,
		(cufftComplex*)d_output,
		buf_size
		);
}
void GPUSoundSource::caseTwoConvolve(float* d_input, float* d_output,
	cufftComplex* d_convbufs, cufftComplex* d_distance_factor,
	cudaStream_t* streams, int* hrtf_indices, float* omegas) {
	int buf_size = PAD_LEN + 2;
	int complex_buf_size = buf_size / 2;
	int numThreads = 64;
	int numBlocks = (buf_size + numThreads - 1) / numThreads;
	for (int buf_no = 0; buf_no < 2; buf_no++) {
		float curr_scale;
		int hrtf_index;
		switch (buf_no) {
		case 0:
			hrtf_index = hrtf_indices[0];
			curr_scale = omegas[1];
			break;
		case 1:
			hrtf_index = hrtf_indices[1];
			curr_scale = omegas[0];
			break;
		}
		numBlocks = (buf_size + numThreads - 1) / numThreads;
		ComplexPointwiseMulAndScaleOutPlace << < numBlocks, numThreads, 0, streams[buf_no] >> > (
			(cufftComplex*)d_input,
			d_fft_hrtf + hrtf_index * complex_buf_size * HRTF_CHN,
			d_convbufs + buf_size * buf_no,
			curr_scale,
			buf_size
			);
		numBlocks = (complex_buf_size + numThreads - 1) / numThreads;
		ComplexPointwiseMulInPlace << < numBlocks, numThreads, 0, streams[buf_no] >> > (
			d_distance_factor,
			d_convbufs + buf_size * buf_no,
			complex_buf_size
			);
		ComplexPointwiseMulInPlace << < numBlocks, numThreads, 0, streams[buf_no] >> > (
			d_distance_factor,
			d_convbufs + buf_size * buf_no + complex_buf_size,
			complex_buf_size
			);
		numBlocks = (buf_size + numThreads - 1) / numThreads;
		ComplexPointwiseAdd << < numBlocks, numThreads, 0, streams[buf_no] >> > (
			d_convbufs + buf_size * buf_no,
			(cufftComplex*)(d_output),
			buf_size
			);
	}
}
void GPUSoundSource::caseThreeConvolve(float* d_input, float* d_output,
	cufftComplex* d_convbufs, cufftComplex* d_distance_factor,
	cudaStream_t* streams, int* hrtf_indices, float* omegas) {
	int buf_size = PAD_LEN + 2;
	int complex_buf_size = buf_size / 2;
	int numThreads = 64;
	int numBlocks = (buf_size + numThreads - 1) / numThreads;
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
		ComplexPointwiseMulAndScaleOutPlace << < numBlocks, numThreads, 0, streams[buf_no] >> > (
			(cufftComplex*)d_input,
			d_fft_hrtf + hrtf_index * complex_buf_size * HRTF_CHN,
			d_convbufs + buf_size * buf_no,
			curr_scale,
			buf_size
			);
		numBlocks = (complex_buf_size + numThreads - 1) / numThreads;
		ComplexPointwiseMulInPlace << < numBlocks, numThreads, 0, streams[buf_no] >> > (
			d_distance_factor,
			d_convbufs + buf_size * buf_no,
			complex_buf_size
			);
		ComplexPointwiseMulInPlace << < numBlocks, numThreads, 0, streams[buf_no] >> > (
			d_distance_factor,
			d_convbufs + buf_size * buf_no + complex_buf_size,
			complex_buf_size
			);
		numBlocks = (buf_size + numThreads - 1) / numThreads;
		ComplexPointwiseAdd << < numBlocks, numThreads, 0, streams[buf_no] >> > (
			d_convbufs + buf_size * buf_no,
			(cufftComplex*)(d_output),
			buf_size
			);
	}
}

void GPUSoundSource::caseFourConvolve(float* d_input, float* d_output,
	cufftComplex* d_convbufs, cufftComplex* d_distance_factor,
	cudaStream_t* streams, int* hrtf_indices, float* omegas) {
	int buf_size = PAD_LEN + 2;
	int complex_buf_size = buf_size / 2;
	int numThreads = 64;
	int numBlocks = (buf_size + numThreads - 1) / numThreads;
	for (int buf_no = 0; buf_no < 4; buf_no++) {
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
		ComplexPointwiseMulAndScaleOutPlace << < numBlocks, numThreads, 0, streams[buf_no] >> > (
			(cufftComplex*)d_input,
			d_fft_hrtf + hrtf_index * complex_buf_size * HRTF_CHN,
			d_convbufs + buf_size * buf_no,
			curr_scale,
			buf_size
			);
		numBlocks = (complex_buf_size + numThreads - 1) / numThreads;
		ComplexPointwiseMulInPlace << < numBlocks, numThreads, 0, streams[buf_no] >> > (
			d_distance_factor,
			d_convbufs + buf_size * buf_no,
			complex_buf_size
			);
		ComplexPointwiseMulInPlace << < numBlocks, numThreads, 0, streams[buf_no] >> > (
			d_distance_factor,
			d_convbufs + buf_size * buf_no + complex_buf_size,
			complex_buf_size
			);
		numBlocks = (buf_size + numThreads - 1) / numThreads;
		ComplexPointwiseAdd << < numBlocks, numThreads, 0, streams[buf_no] >> > (
			d_convbufs + buf_size * buf_no,
			(cufftComplex*)(d_output),
			buf_size
			);
	}
}

void GPUSoundSource::allKernels(float* d_input, float* d_output,
	cufftComplex* d_convbufs, cufftComplex* d_distance_factor,
	cudaStream_t* streams, float* omegas, int* hrtf_indices, cudaEvent_t fft_in) {
	size_t buf_size = PAD_LEN + 2;
	size_t complex_buf_size = buf_size / 2;
	int numThreads = 128;
	int numBlocks = (buf_size + numThreads - 1) / numThreads;
	for (int i = 0; i < 4; i++) {
		checkCudaErrors(cudaStreamWaitEvent(streams[i], fft_in, 0));
	}
	/*The azi & ele falls exactly on an hrtf resolution*/
	if (hrtf_indices[0] == hrtf_indices[1] && hrtf_indices[1] == hrtf_indices[2] && hrtf_indices[2] == hrtf_indices[3]) {
		caseOneConvolve(d_input, d_output, d_convbufs, d_distance_factor, streams, hrtf_indices);
	}
	/*If the elevation falls on the resolution, interpolate the azimuth*/
	else if (hrtf_indices[0] == hrtf_indices[2] && hrtf_indices[1] == hrtf_indices[3]) {
		caseTwoConvolve(d_input, d_output, d_convbufs, d_distance_factor, streams, hrtf_indices, omegas);

	}
	/*If the azimuth falls on the resolution, interpolate the elevation*/
	else if (hrtf_indices[0] == hrtf_indices[1] && hrtf_indices[0] != hrtf_indices[2]) {
		caseThreeConvolve(d_input, d_output, d_convbufs, d_distance_factor, streams, hrtf_indices, omegas);
	}
	/*Worst case scenario*/
	else {
		caseFourConvolve(d_input, d_output, d_convbufs, d_distance_factor, streams, hrtf_indices, omegas);
	}
}

/*blockNo MUST be modulo FLIGHT_NUM*/
void GPUSoundSource::interpolateConvolve() {
	cufftComplex* d_convbufs2 = this->d_convbufs + 8 * (PAD_LEN + 2);
	float scale = 1.0f / ((float)PAD_LEN);
	int buf_size = PAD_LEN + 2;
	int numThreads = 128;
	int numBlocks = (buf_size + numThreads - 1) / numThreads;
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
	fillWithZeroesKernel(d_output, 2 * (PAD_LEN + 2), streams[0]);
	gpuCalculateDistanceFactor(streams[1]);
	if (xfade) {
		fillWithZeroesKernel(d_output2, 2 * (PAD_LEN + 2), streams[2]);
	}
	CHECK_CUFFT_ERRORS(cufftSetStream(plans[0], streams[0]));
	CHECK_CUFFT_ERRORS(cufftSetStream(plans[1], streams[0]));
	CHECK_CUFFT_ERRORS(cufftSetStream(plans[2], streams[4]));
	CHECK_CUFFT_ERRORS(cufftExecR2C(plans[0], (cufftReal*)d_input, (cufftComplex*)d_input));

	MyFloatScale << < numBlocks, numThreads, 0, streams[0] >> > (d_input, scale, buf_size);
	checkCudaErrors(cudaEventRecord(fft_events, streams[0]));
	checkCudaErrors(cudaMemcpy(d_input + buf_size, d_input, buf_size * sizeof(float), cudaMemcpyDeviceToDevice));
	if (!xfade) {
		allKernels(d_input, d_output, d_convbufs, d_distance_factor, streams, omegas, hrtf_indices, fft_events);
		for (int i = 1; i < 4; i++) {
			checkCudaErrors(cudaEventRecord(kernel_launches[i], streams[i]));
		}
		for (int i = 1; i < 4; i++) {
			checkCudaErrors(cudaStreamWaitEvent(streams[0], kernel_launches[i], 0));
		}
		CHECK_CUFFT_ERRORS(cufftExecC2R(plans[1], (cufftComplex*)d_output, d_output));

	}
	else {
		allKernels(d_input, d_output, d_convbufs, d_distance_factor, streams, old_omegas, old_hrtf_indices, fft_events);
		allKernels(d_input, d_output2, d_convbufs2, d_distance_factor, streams + 4, omegas, hrtf_indices, fft_events);
		for (int i = 0; i < 8; i++) {
			checkCudaErrors(cudaEventRecord(kernel_launches[i], streams[i]));
		}
		for (int i = 1; i < 4; i++) {
			checkCudaErrors(cudaStreamWaitEvent(streams[0], kernel_launches[i], 0));
		}
		for (int i = 5; i < 8; i++) {
			checkCudaErrors(cudaStreamWaitEvent(streams[4], kernel_launches[i], 0));
		}
		//checkCudaErrors(cudaDeviceSynchronize());
		CHECK_CUFFT_ERRORS(cufftExecC2R(plans[1], (cufftComplex*)d_output, d_output));
		CHECK_CUFFT_ERRORS(cufftExecC2R(plans[2], (cufftComplex*)d_output2, d_output2));
		checkCudaErrors(cudaEventRecord(fft_events, streams[4]));
		checkCudaErrors(cudaStreamWaitEvent(streams[0], fft_events, 0));
		
		int numThreads = FRAMES_PER_BUFFER;
		crossFade << <numThreads, 1, 0, streams[0] >> > (
			d_output + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
			d_output2 + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
			FRAMES_PER_BUFFER);
	}
	old_azi = azi;
	old_ele = ele;
}
void GPUSoundSource::fftConvolve() {
	cudaEvent_t start, stop;
	/*cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);*/

	float scale = 1.0f / ((float)PAD_LEN);
	CHECK_CUFFT_ERRORS(cufftSetStream(plans[0], streams[0]));
	CHECK_CUFFT_ERRORS(cufftSetStream(plans[1], streams[0]));
	CHECK_CUFFT_ERRORS(cufftExecR2C(plans[0], (cufftReal*)d_input, (cufftComplex*)d_input));
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
	CHECK_CUFFT_ERRORS(cufftExecC2R(plans[1], (cufftComplex*)d_output, d_output));

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
void GPUSoundSource::sendBlock() {
	checkCudaErrors(cudaMemcpyAsync(
		d_input,
		x,
		PAD_LEN * sizeof(float),
		cudaMemcpyHostToDevice,
		streams[0])
	);

}
void GPUSoundSource::receiveBlock() {
	checkCudaErrors(cudaMemcpyAsync(
		intermediate,
		d_output + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
		FRAMES_PER_BUFFER * 2 * sizeof(float),
		cudaMemcpyDeviceToHost,
		streams[0])
	);
}
void GPUSoundSource::chunkProcess() {
	overlapSave();
	copyIncomingBlock();
	sendBlock();
	/*Process*/
	interpolateConvolve();
	/*Receive*/
	receiveBlock();
}
/*Must send un-modded block number*/
void GPUSoundSource::overlapSave() {
	/*if (blockNo == 0) {
		return;
	}*/
	/*checkCudaErrors(cudaStreamWaitEvent(streams[0], incomingTransfers[0], 0));
	checkCudaErrors(cudaStreamWaitEvent(streams[0], incomingTransfers[1], 0));*/
	/*Overlap-save, put the function in a stream*/

	callback_data_blocks[1].input = x + FRAMES_PER_BUFFER;
	callback_data_blocks[1].output = x;
	callback_data_blocks[1].size = PAD_LEN - FRAMES_PER_BUFFER;
	cudaHostFn_t fn = memcpyCallback;
	checkCudaErrors(cudaLaunchHostFunc(streams[1], fn, &callback_data_blocks[1]));
	checkCudaErrors(cudaEventRecord(incomingTransfers[0], streams[0]));
	//checkCudaErrors(cudaEventRecord(incomingTransfers[1], streams[1]));
	checkCudaErrors(cudaDeviceSynchronize());
	//memcpy(x[moddedBlockNo], x[(blockNo - 1) % FLIGHT_NUM] + FRAMES_PER_BUFFER, (PAD_LEN - FRAMES_PER_BUFFER) * sizeof(float));

}
/*Must send modulo'd block number*/
void GPUSoundSource::copyIncomingBlock() {
	checkCudaErrors(cudaStreamWaitEvent(streams[0], incomingTransfers[0], 0));
	/*Copy into curr_source->x pinned memory*/
	if (count + FRAMES_PER_BUFFER < length) {
		callback_data_blocks[0].input = buf + count;
		callback_data_blocks[0].output = x + (PAD_LEN - FRAMES_PER_BUFFER);
		callback_data_blocks[0].size = FRAMES_PER_BUFFER;
		cudaHostFn_t fn = memcpyCallback;
		checkCudaErrors(cudaLaunchHostFunc(streams[0], fn, &callback_data_blocks[0]));
		checkCudaErrors(cudaEventRecord(incomingTransfers[1], streams[0]));
		count += FRAMES_PER_BUFFER;
	}
	else {
		int rem = length - count;
		callback_data_blocks[0].input = buf + count;
		callback_data_blocks[0].output = x + (PAD_LEN - FRAMES_PER_BUFFER);
		callback_data_blocks[0].size = rem;
		cudaHostFn_t fn = memcpyCallback;
		checkCudaErrors(cudaLaunchHostFunc(streams[0], fn, &callback_data_blocks[0]));
		memcpy(
			x + (PAD_LEN - FRAMES_PER_BUFFER),
			buf + count,
			rem * sizeof(float));
		memcpy(
			x + (PAD_LEN - FRAMES_PER_BUFFER) + rem,
			buf,
			(FRAMES_PER_BUFFER - rem) * sizeof(float));
		callback_data_blocks[2].input = buf;
		callback_data_blocks[2].output = x + (PAD_LEN - FRAMES_PER_BUFFER) + rem;
		callback_data_blocks[2].size = FRAMES_PER_BUFFER - rem;
		fn = memcpyCallback;
		checkCudaErrors(cudaLaunchHostFunc(streams[0], fn, &callback_data_blocks[2]));
		checkCudaErrors(cudaEventRecord(incomingTransfers[0], streams[0]));
		count = FRAMES_PER_BUFFER - rem;
	}
	checkCudaErrors(cudaDeviceSynchronize());
}
void GPUSoundSource::process(processes type) {
	switch (type) {
	case GPU_FD_COMPLEX:
		chunkProcess();
		break;
	case GPU_FD_BASIC:
		fftConvolve();
		break;
	case GPU_TD:
		gpuTDConvolve(
			d_input + PAD_LEN - FRAMES_PER_BUFFER,
			d_output + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
			FRAMES_PER_BUFFER,
			gain, streams);
		break;
	}

}
GPUSoundSource::~GPUSoundSource() {
	free(buf);
	checkCudaErrors(cudaFreeHost(x));
	checkCudaErrors(cudaFree(d_input));
	checkCudaErrors(cudaFree(d_output));
	checkCudaErrors(cudaFree(distance_factor));
	for (int j = 0; j < STREAMS_PER_FLIGHT; j++) {
		checkCudaErrors(cudaStreamDestroy(streams[j]));
		checkCudaErrors(cudaEventDestroy(kernel_launches[j]));
	}
	checkCudaErrors(cudaEventDestroy(fft_events));
	checkCudaErrors(cudaFree(d_convbufs));
	checkCudaErrors(cudaFreeHost(intermediate));
	CHECK_CUFFT_ERRORS(cufftDestroy(plans[0]));
	CHECK_CUFFT_ERRORS(cufftDestroy(plans[1]));
	CHECK_CUFFT_ERRORS(cufftDestroy(plans[2]));

	delete[] streams;
	delete[] fft_events;
	delete[] kernel_launches;
}

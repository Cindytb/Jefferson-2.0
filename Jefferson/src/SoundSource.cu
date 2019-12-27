#include "SoundSource.cuh"
void pointwiseAddition(fftwf_complex* a, fftwf_complex* b, int size) {
#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		a[i][0] += b[i][0];
		a[i][1] += b[i][1];
	}
}
void pointwiseAddition(fftwf_complex* a, fftwf_complex* b, fftwf_complex* c, int size) {
#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		c[i][0] = a[i][0] + b[i][0];
		c[i][1] = a[i][1] + b[i][1];
	}
}
void pointwiseMultiplication(fftwf_complex* a, fftwf_complex* b, int size) {
#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		fftwf_complex temp;
		temp[0] = a[i][0];
		temp[1] = a[i][1];
		a[i][0] = temp[0] * b[i][0] - temp[1] * b[i][1];
		a[i][1] = temp[0] * b[i][1] + temp[1] * b[i][0];
	}
}
void complexScaling(fftwf_complex* f_x, float scale, int size) {
#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		f_x[i][0] *= scale;
		f_x[i][1] *= scale;
	}
}

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

SoundSource::SoundSource() {
	count = 0;
	length = 0;
	gain = 0.99074;
	hrtf_idx = 314;
	coordinates.x = 0;
	coordinates.y = 0;
	coordinates.z = 0.5;
	azi = 3;
	ele = 5;
	r = 0.5;
	
	streams = new cudaStream_t[FLIGHT_NUM * STREAMS_PER_FLIGHT];
	for (int i = 0; i < FLIGHT_NUM; i++) {
		checkCudaErrors(cudaEventCreate(&incomingTransfers[i * 3]));
		checkCudaErrors(cudaEventCreate(&incomingTransfers[i * 3 + 1]));
		checkCudaErrors(cudaEventCreate(&incomingTransfers[i * 3 + 2]));
		/*Allocating pinned memory for incoming transfer*/
		checkCudaErrors(cudaMallocHost(x + i, (PAD_LEN + 2) * sizeof(float)));
		/*Allocating memory for the inputs*/
		checkCudaErrors(cudaMalloc(d_input + i, (PAD_LEN + 2) * sizeof(float)));
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
	}
	for (int i = 0; i < FLIGHT_NUM; i++) {
		for (int j = 0; j < PAD_LEN + 2; j++) {
			x[i][j] = 0.0f;
		}
	}
	fftw_intermediate = fftwf_alloc_complex(4 * (PAD_LEN + 2));
	fftw_conv_bufs = fftwf_alloc_complex(8 * (PAD_LEN + 2));
	fftw_distance_factor = fftwf_alloc_complex(PAD_LEN / 2 + 1);
	CHECK_CUFFT_ERRORS(cufftPlan1d(&in_plan, PAD_LEN, CUFFT_R2C, 1));
	/*cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n,
		int *inembed, int istride, int idist,
		int *onembed, int ostride, int odist,
		cufftType type, int batch);*/
		/*stride = skip length for interleaving. Ex 1 = every element, 2 = every other element*/
			/*use for interleaving*/
		/*idist/odist is space between batches of transforms*/
			/*need to check if odist is in terms of complex numbers or floats*/
		/*inembed/onembed are for 2D/3D, num elements per dimension*/
	/*This type of cufft plan will take 2 mono channels located contiguously in memory, take the IFFT, and interleave them*/
	int n = PAD_LEN;
	CHECK_CUFFT_ERRORS(
		cufftPlanMany(
			&out_plan, 1, &n,
			&n, 1, n / 2 + 1,
			&n, 2, 1,
			CUFFT_C2R, 2)
	);
	CHECK_CUFFT_ERRORS(
		cufftPlanMany(
			&out_plan2, 1, &n,
			&n, 1, n / 2 + 1,
			&n, 2, 1,
			CUFFT_C2R, 2)
	);
	fftw_in_plan = fftwf_plan_dft_r2c_1d(PAD_LEN, x[0], fftw_intermediate, FFTW_ESTIMATE);
	fftw_out_plan = fftwf_plan_many_dft_c2r(
		1, &PAD_LEN, 2, 
		fftw_intermediate, NULL, 
		1, PAD_LEN / 2 + 1, 
		(float*)fftw_intermediate, NULL, 
		2, 1, FFTW_ESTIMATE
	);
	
	old_azi = azi;
	old_ele = ele;
}
void SoundSource::updateFromCartesian() {
	updateFromCartesian(coordinates);
}
void SoundSource::updateFromCartesian(float3 point) {
	coordinates.x = point.x;
	coordinates.y = point.y;
	coordinates.z = point.z;
	/*Calculate the radius, distance, elevation, and azimuth*/
	r = std::sqrt(coordinates.x * coordinates.x + coordinates.z * coordinates.z + coordinates.y * coordinates.y);
	float horizR = std::sqrt(coordinates.x * coordinates.x + coordinates.z * coordinates.z);
	ele = (float)atan2(coordinates.y, horizR) * 180.0f / PI;

	azi = atan2(-coordinates.x / r, -coordinates.z / r) * 180.0f / PI;
	if (azi < 0.0f) {
		azi += 360;
	}
	ele = round(ele);
	azi = round(azi);
	hrtf_idx = pick_hrtf(ele, azi);
}
void SoundSource::updateFromSpherical() {
	updateFromSpherical(ele, azi, r);
}

void SoundSource::updateFromSpherical(float ele, float azi, float r) {
	this->ele = round(ele);
	this->azi = round(azi);
	this->r = r;
	ele = this->ele;
	azi = this->azi;
	coordinates.x = r * sin(azi * PI / 180.0f);
	coordinates.z = r * -cos(azi * PI / 180.0f);
	coordinates.y = r * sin(ele * PI / 180.0f);
	/*coordinates.x = r * sin(ele * PI / 180.0f) * cos((azi + 90) * PI / 180.0f);
	coordinates.y = r * sin(ele * PI / 180.0f); 
	coordinates.z = r * sin(ele * PI / 180.0f) * cos((azi + 90) * PI / 180.0f);*/
	hrtf_idx = pick_hrtf(ele, azi);
}
void SoundSource::drawWaveform() {
	float rotateVBO_y = atan2(-coordinates.z, coordinates.x) * 180.0f / PI;

	if (rotateVBO_y < 0) {
		rotateVBO_y += 360;
	}
	waveform->averageNum = 100;
	waveform->update();
	waveform->draw(rotateVBO_y, ele, 0.0f);
}
void SoundSource::interpolationCalculations(float ele, float azi, int* hrtf_indices, float* omegas) {
	float omegaA, omegaB, omegaC, omegaD, omegaE, omegaF;
	int phi[2];
	int theta[4];
	float deltaTheta1, deltaTheta2;
	phi[0] = int(ele) / 10 * 10; /*hard coded 10 because the elevation increments in the KEMAR HRTF database is 10 degrees*/
	phi[1] = int(ele + 9) / 10 * 10;
	omegaE = (ele - phi[0]) / 10.0f;
	omegaF = (phi[1] - ele) / 10.0f;

	for (int i = 0; i < NUM_ELEV; i++) {
		if (phi[0] == elevation_pos[i]) {
			deltaTheta1 = azimuth_inc[i];
		}
		if (phi[1] == elevation_pos[i]) {
			deltaTheta2 = azimuth_inc[i];
			break;
		}
	}
	theta[0] = int(azi / deltaTheta1) * deltaTheta1;
	theta[1] = int((azi + deltaTheta1 - 1) / deltaTheta1) * deltaTheta1;
	theta[2] = int(azi / deltaTheta2) * deltaTheta2;
	theta[3] = int((azi + deltaTheta2 - 1) / deltaTheta2) * deltaTheta2;
	omegaA = (azi - theta[0]) / deltaTheta1;
	omegaB = (theta[1] - azi) / deltaTheta1;
	omegaC = (azi - theta[2]) / deltaTheta2;
	omegaD = (theta[3] - azi) / deltaTheta2;

	hrtf_indices[0] = pick_hrtf(phi[0], theta[0]);
	hrtf_indices[1] = pick_hrtf(phi[0], theta[1]);
	hrtf_indices[2] = pick_hrtf(phi[1], theta[2]);
	hrtf_indices[3] = pick_hrtf(phi[1], theta[3]);

	omegas[0] = omegaA;
	omegas[1] = omegaB;
	omegas[2] = omegaC;
	omegas[3] = omegaD;
	omegas[4] = omegaE;
	omegas[5] = omegaF;

}
/*
	R(r) = (1 / (1 + (fs / vs) (r - r0)^2) ) * e^ ((-j2PI (fs/vs) * (r - r0) *k) / N)
			|----------FRAC-----------------|	  |------------exponent------------------|

	FRAC * e^(exponent)
	FRAC * (cosine(exponent) - sine(exponent))
	R[r].x = FRAC * cosine(exponent)
	R[r].y = -FRAC * sine(exponent)
	*/
void SoundSource::gpuCalculateDistanceFactor(int blockNo, cudaStream_t stream){
	cufftComplex* d_distance_factor = this->distance_factor[blockNo % FLIGHT_NUM];
	cudaStream_t* streams = this->streams + (blockNo % FLIGHT_NUM) * STREAMS_PER_FLIGHT;
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
void SoundSource::cpuCalculateDistanceFactor() {
	float r = std::sqrt(
		coordinates.x * coordinates.x +
		coordinates.y * coordinates.y +
		coordinates.z * coordinates.z
	);
	r /= 5;
	float fsvs = 44100.0 / 343.0;
	float frac = 1 + fsvs * pow(r, 2);
	float N = PAD_LEN / 2 + 1;
	#pragma omp parallel for
	for (int i = 0; i < N; i++) {
		fftw_distance_factor[i][0] = cos(2 * PI * fsvs * r * i / N) / frac;
		fftw_distance_factor[i][1] = -sin(2 * PI * fsvs * r * i / N) / frac;
	}
}
/*
This method is a slightly tweaked implementation of Jose Belloch's
"Headphone-Based Virtual Spatialization of Sound with a GPU Accelerator"
paper from the Journal of the Audio Engineering Society,
Volume 61, No 7/8, 2013, July/August
*/
void SoundSource::allKernels(float* d_input, float* d_output, 
	cufftComplex* d_convbufs, cufftComplex* d_distance_factor, 
	cudaStream_t* streams, float* omegas, int* hrtf_indices, cudaEvent_t fft_in){
	

	float scale = 1.0f / ((float)PAD_LEN);
	
	int numThreads = 64;
	int numBlocks = (PAD_LEN / 2 + numThreads) / numThreads;
	size_t buf_size = PAD_LEN + 2;
	for (int i = 0; i < 8; i++) {
		checkCudaErrors(cudaStreamWaitEvent(streams[i], fft_in, 0));
	}
	/*The azi & ele falls exactly on an hrtf resolution*/
	if (hrtf_indices[0] == hrtf_indices[1] && hrtf_indices[1] == hrtf_indices[2] && hrtf_indices[2] == hrtf_indices[3]) {
		/*+ Theta 1 Left*/
		ComplexPointwiseMulAndScaleOutPlace << < numBlocks, numThreads, 0, streams[0] >> > (
			(cufftComplex*)d_input,
			(cufftComplex*)(d_hrtf + hrtf_indices[0] * (PAD_LEN + 2) * HRTF_CHN),
			d_convbufs,
			PAD_LEN / 2 + 1,
			scale
			);
		ComplexPointwiseMulInPlace << < numBlocks, numThreads, 0, streams[0] >> > (
			d_distance_factor, 
			d_convbufs, 
			PAD_LEN / 2 + 1
			);
		ComplexPointwiseAdd << < numBlocks, numThreads, 0, streams[0] >> > (
			d_convbufs,
			(cufftComplex*)d_output,
			PAD_LEN / 2 + 1
			);
		/*+ Theta 1 Right*/
		ComplexPointwiseMulAndScaleOutPlace << < numBlocks, numThreads, 0, streams[1] >> > (
			(cufftComplex*)d_input,
			(cufftComplex*)(d_hrtf + hrtf_indices[0] * (PAD_LEN + 2) * HRTF_CHN + PAD_LEN + 2),
			d_convbufs + buf_size / 2,
			PAD_LEN / 2 + 1,
			scale
			);
		ComplexPointwiseMulInPlace << < numBlocks, numThreads, 0, streams[1] >> > (
			d_distance_factor, 
			d_convbufs + buf_size / 2, 
			PAD_LEN / 2 + 1
			);
		
		ComplexPointwiseAdd << < numBlocks, numThreads, 0, streams[1] >> > (
			d_convbufs + buf_size / 2,
			(cufftComplex*)(d_output + buf_size),
			PAD_LEN / 2 + 1
			);
	}
	/*If the elevation falls on the resolution, interpolate the azimuth*/
	else if (hrtf_indices[0] == hrtf_indices[2]) {
		for (int buf_no = 0; buf_no < 4; buf_no++) {
			/*Even buf numbers are the left channel, odd ones are the right channel*/
			float curr_scale;
			if (buf_no < 2)
				curr_scale = scale * omegas[1];
			else {
				curr_scale = scale * omegas[0];
			}
			int hrtf_index;
			if (buf_no < 2)
				hrtf_index = hrtf_indices[0];
			else
				hrtf_index = hrtf_indices[1];
			ComplexPointwiseMulAndScaleOutPlace << < numBlocks, numThreads, 0, streams[buf_no] >> > (
				(cufftComplex*)d_input,
				(cufftComplex*)(d_hrtf + hrtf_index * (PAD_LEN + 2) * HRTF_CHN + ((buf_no % 2) * (PAD_LEN + 2))),
				d_convbufs + buf_size / 2 * buf_no,
				PAD_LEN / 2 + 1,
				curr_scale
				);
			ComplexPointwiseMulInPlace << < numBlocks, numThreads, 0, streams[buf_no] >> > (
				d_distance_factor, 
				d_convbufs + buf_size / 2 * buf_no,
				PAD_LEN / 2 + 1
				);
			ComplexPointwiseAdd << < numBlocks, numThreads, 0, streams[buf_no] >> > (
				d_convbufs + buf_size / 2 * buf_no,
				(cufftComplex*)(d_output + buf_size * (buf_no % 2)),
				PAD_LEN / 2 + 1
				);
		}

	}
	/*If the azimuth falls on the resolution, interpolate the elevation*/
	else if (hrtf_indices[0] == hrtf_indices[1] && hrtf_indices[0] != hrtf_indices[2]) {
		for (int buf_no = 0; buf_no < 4; buf_no++) {
			/*Even buf numbers are the left channel, odd ones are the right channel*/
			float curr_scale;
			int hrtf_index;
			switch (buf_no) {
			case 0:
			case 1:
				curr_scale = scale * omegas[5]; //double check these scalings
				hrtf_index = 0;
				break;
			case 2:
			case 3:
				curr_scale = scale * omegas[4];
				hrtf_index = 2;
				break;
			}

			ComplexPointwiseMulAndScaleOutPlace << < numBlocks, numThreads, 0, streams[buf_no] >> > (
				(cufftComplex*)d_input,
				(cufftComplex*)(d_hrtf + hrtf_indices[hrtf_index] * (PAD_LEN + 2) * HRTF_CHN + ((buf_no % 2) * (PAD_LEN + 2))),
				d_convbufs + buf_size / 2 * buf_no,
				PAD_LEN / 2 + 1,
				curr_scale
				);
			ComplexPointwiseMulInPlace << < numBlocks, numThreads, 0, streams[buf_no] >> > (
				d_distance_factor,
				d_convbufs + buf_size / 2 * buf_no,
				PAD_LEN / 2 + 1
				);
			ComplexPointwiseAdd << < numBlocks, numThreads, 0, streams[buf_no] >> > (
				d_convbufs + buf_size / 2 * buf_no,
				(cufftComplex*)(d_output + buf_size * (buf_no % 2)),
				PAD_LEN / 2 + 1
				);
		}
	}
	/*Worst case scenario*/
	else {
		for (int buf_no = 0; buf_no < 8; buf_no++) {
			/*Even buf numbers are the left channel, odd ones are the right channel*/
			float curr_scale;
			int hrtf_index = buf_no / 2;
			switch (hrtf_index) {
			case 0:
				curr_scale = scale * omegas[5] * omegas[1];
				break;
			case 1:
				curr_scale = scale * omegas[5] * omegas[0];
				break;
			case 2:
				curr_scale = scale * omegas[4] * omegas[3];
				break;
			case 3:
				curr_scale = scale * omegas[4] * omegas[2];
				break;
			}
			int blah = buf_size / 2 * buf_no;
			int blah2 = buf_size * (buf_no % 2);
			ComplexPointwiseMulAndScaleOutPlace << < numBlocks, numThreads, 0, streams[buf_no] >> > (
				(cufftComplex*)d_input,
				(cufftComplex*)(d_hrtf + hrtf_indices[hrtf_index] * (PAD_LEN + 2) * HRTF_CHN + ((buf_no % 2) * (PAD_LEN + 2))),
				d_convbufs + buf_size / 2 * buf_no,
				PAD_LEN / 2 + 1,
				curr_scale
				);
			ComplexPointwiseMulInPlace << < numBlocks, numThreads, 0, streams[buf_no] >> > (
				d_distance_factor,
				d_convbufs + buf_size / 2 * buf_no,
				PAD_LEN / 2 + 1
				);
			ComplexPointwiseAdd << < numBlocks, numThreads, 0, streams[buf_no] >> > (
				d_convbufs + buf_size / 2 * buf_no,
				(cufftComplex*)(d_output + buf_size * (buf_no % 2)),
				PAD_LEN / 2 + 1
				);
		}
	}
}

/*blockNo MUST be modulo FLIGHT_NUM*/
void SoundSource::interpolateConvolve(int blockNo) {
	cufftComplex* d_distance_factor = this->distance_factor[blockNo];
	float* d_input = this->d_input[blockNo];
	float* d_output = this->d_output[blockNo];
	float* d_output2 = this->d_output2[blockNo];
	cufftComplex* d_convbufs = this->d_convbufs[blockNo];
	cufftComplex* d_convbufs2 = this->d_convbufs[blockNo] + 8 * (PAD_LEN + 2);
	cudaStream_t* streams = this->streams + blockNo * STREAMS_PER_FLIGHT;
	cudaEvent_t fft_in_event, xfade_fft_out;
	cudaEvent_t* kernel_launches = new cudaEvent_t[STREAMS_PER_FLIGHT];
	for (int i = 0; i < STREAMS_PER_FLIGHT; i++) {
		checkCudaErrors(cudaEventCreate(kernel_launches + i));
	}
	checkCudaErrors(cudaEventCreate(&fft_in_event));
	checkCudaErrors(cudaEventCreate(&xfade_fft_out));
	int old_hrtf_indices[4];
	float old_omegas[6];
	interpolationCalculations(ele, azi, hrtf_indices, omegas);
	bool xfade = false;
	if (old_azi != azi || old_ele != ele) {
		/*printf("Crossfading!\n");
		printf("Theta1: %f theta2: %f", old_azi, azi);
		printf("Phi1:   %f phi2:   %f", old_ele, ele);*/
		xfade = true;
		interpolationCalculations(old_ele, old_azi, old_hrtf_indices, old_omegas);
	}
	gpuCalculateDistanceFactor(blockNo, streams[1]);
	fillWithZeroesKernel(d_output, 2 * (PAD_LEN + 2), streams[2]);
	if (xfade) {
		fillWithZeroesKernel(d_output2, 2 * (PAD_LEN + 2), streams[3]);
	}

	CHECK_CUFFT_ERRORS(cufftSetStream(in_plan, streams[0]));
	CHECK_CUFFT_ERRORS(cufftExecR2C(in_plan, (cufftReal*)d_input, (cufftComplex*)d_input));
	checkCudaErrors(cudaEventRecord(fft_in_event, streams[0]));
	if (!xfade) {
		allKernels(d_input, d_output, d_convbufs, d_distance_factor, streams, omegas, hrtf_indices, fft_in_event);
		for (int i = 1; i < 8; i++) {
			checkCudaErrors(cudaEventRecord(kernel_launches[i], streams[i]));
		}
		CHECK_CUFFT_ERRORS(cufftSetStream(out_plan, streams[0]));
		for (int i = 1; i < 8; i++) {
			checkCudaErrors(cudaStreamWaitEvent(streams[0], kernel_launches[i], 0));
		}
		CHECK_CUFFT_ERRORS(cufftExecC2R(out_plan, (cufftComplex*)d_output, d_output));
	}
	else {
		allKernels(d_input, d_output, d_convbufs, d_distance_factor, streams, omegas, hrtf_indices, fft_in_event);
		allKernels(d_input, d_output2, d_convbufs2, d_distance_factor, streams + 8, old_omegas, old_hrtf_indices, fft_in_event);
		for (int i = 0; i < 16; i++) {
			checkCudaErrors(cudaEventRecord(kernel_launches[i], streams[i]));
		}
		CHECK_CUFFT_ERRORS(cufftSetStream(out_plan, streams[0]));
		CHECK_CUFFT_ERRORS(cufftSetStream(out_plan2, streams[8]));
		for (int i = 1; i < 8; i++) {
			checkCudaErrors(cudaStreamWaitEvent(streams[0], kernel_launches[i], 0));
		}
		for (int i = 1; i < 8; i++) {
			checkCudaErrors(cudaStreamWaitEvent(streams[8], kernel_launches[i + 8], 0));
		}
		CHECK_CUFFT_ERRORS(cufftExecC2R(out_plan, (cufftComplex*)d_output, d_output));
		CHECK_CUFFT_ERRORS(cufftExecC2R(out_plan2, (cufftComplex*)d_output2, d_output2));
		checkCudaErrors(cudaEventRecord(xfade_fft_out, streams[8]));
		int numThreads = FRAMES_PER_BUFFER;
		checkCudaErrors(cudaStreamWaitEvent(streams[0], xfade_fft_out, 0));
		crossFade << <numThreads, 1, 0, streams[0] >> > (
			d_output + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
			d_output2 + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
			FRAMES_PER_BUFFER);
	}
	old_azi = azi;
	old_ele = ele;
}
void SoundSource::fftConvolve(int blockNo) {
	cudaEvent_t start, stop;
	/*cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);*/

	float* d_input = this->d_input[blockNo % FLIGHT_NUM];
	float* d_output = this->d_output[blockNo % FLIGHT_NUM];
	cudaStream_t* streams = this->streams + (blockNo % FLIGHT_NUM) * STREAMS_PER_FLIGHT;
	if (gain > 1)
		gain = 1;
	float scale = 1.0f / ((float) PAD_LEN);
	CHECK_CUFFT_ERRORS(cufftSetStream(in_plan, streams[0]));
	CHECK_CUFFT_ERRORS(cufftSetStream(out_plan, streams[0]));
	CHECK_CUFFT_ERRORS(cufftExecR2C(in_plan, (cufftReal*)d_input, (cufftComplex*)d_input));
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
	CHECK_CUFFT_ERRORS(cufftExecC2R(out_plan, (cufftComplex*)d_output, d_output));

}
void SoundSource::gpuTDConvolve(float* input, float* d_output, int outputLen, float gain, cudaStream_t* streams) {
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
void SoundSource::sendBlock(int blockNo) {
	cudaStream_t* streams = this->streams + blockNo * STREAMS_PER_FLIGHT;
	checkCudaErrors(cudaStreamWaitEvent(streams[0], incomingTransfers[blockNo * 3], 0));
	checkCudaErrors(cudaStreamWaitEvent(streams[0], incomingTransfers[blockNo * 3 + 1], 0));
	/*Send*/
	checkCudaErrors(cudaMemcpyAsync(
		d_input[blockNo],
		x[blockNo],
		PAD_LEN * sizeof(float),
		cudaMemcpyHostToDevice,
		streams[0])
	);
	
}
void SoundSource::receiveBlock(int blockNo) {
	cudaStream_t* streams = this->streams + blockNo * STREAMS_PER_FLIGHT;
	checkCudaErrors(cudaMemcpyAsync(
		intermediate[blockNo],
		d_output[blockNo] + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
		FRAMES_PER_BUFFER * 2 * sizeof(float),
		cudaMemcpyDeviceToHost,
		streams[0])
	);
}
void SoundSource::chunkProcess(int blockNo) {
	copyIncomingBlock(blockNo % FLIGHT_NUM);
	overlapSave(blockNo);
	sendBlock(blockNo % FLIGHT_NUM);
	/*Process*/
	process(blockNo % FLIGHT_NUM);
	/*Receive*/
	receiveBlock(blockNo % FLIGHT_NUM);
}
/*Must send un-modded block number*/
void SoundSource::overlapSave(int blockNo) {
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
	
}
/*Must send modulo'd block number*/
void SoundSource::copyIncomingBlock(int blockNo) {
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
}
void SoundSource::process(int blockNo){
#ifdef RT_GPU_INTERPOLATE
	 interpolateConvolve(blockNo);
#endif
#ifdef RT_GPU_BASIC
	fftConvolve(blockNo);
#endif
#ifdef RT_GPU_TD
	/*Process*/
	gpuTDConvolve(
		d_input[blockNo % FLIGHT_NUM] + PAD_LEN - FRAMES_PER_BUFFER,
		d_output[blockNo % FLIGHT_NUM] + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
		FRAMES_PER_BUFFER,
		gain, streams + blockNo * STREAMS_PER_FLIGHT);

#endif
#ifdef CPU_FD_BASIC
	 cpuFFTConvolve();
#endif
#ifdef CPU_FD_COMPLEX
	 cpuFFTInterpolate();
#endif
}

/////////////////////////////////////////////////////////////////
/*CPU Implementations*/
/////////////////////////////////////////////////////////////////
void SoundSource::cpuTDConvolve(float* input, float* output, int outputLen, float gain) {
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	//cudaEventRecord(start);
	float* l_hrtf = hrtf + hrtf_idx * HRTF_CHN * (PAD_LEN + 2);
	float* r_hrtf = hrtf + hrtf_idx * HRTF_CHN * (PAD_LEN + 2) + PAD_LEN + 2;
	if (gain > 1)
		gain = 1;

	/* zero output buffer */
	#pragma omp parallel for
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
void SoundSource::cpuFFTConvolve() {
	cudaEvent_t start, stop;
	/*cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);*/
	float* output = intermediate[0];
	fftwf_execute(fftw_in_plan); /*FFT on x[0] --> fftw_intermediate*/
	complexScaling(fftw_intermediate, 1.0 / PAD_LEN, PAD_LEN / 2 + 1);
	/*Copying over for both channels*/
	#pragma omp for parallel
	for (int i = 0; i < PAD_LEN / 2 + 1; i++) {
		fftw_intermediate[i + PAD_LEN / 2 + 1][0] = fftw_intermediate[i][0];
		fftw_intermediate[i + PAD_LEN / 2 + 1][1] = fftw_intermediate[i][1];
	}
	/*Doing both channels at once since they're contiguous in memory*/
	pointwiseMultiplication(fftw_intermediate,
		fft_hrtf + hrtf_idx * HRTF_CHN * (PAD_LEN / 2 + 1),
		PAD_LEN + 2);
	fftwf_execute(fftw_out_plan);

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

void SoundSource::cpuInterpolateLoops(fftwf_complex* output, fftwf_complex* convbufs, int *hrtf_indices, float* omegas) {
	if (hrtf_indices[0] == hrtf_indices[1] && hrtf_indices[1] == hrtf_indices[2] && hrtf_indices[2] == hrtf_indices[3]) {
		pointwiseMultiplication(output,
			fft_hrtf + hrtf_indices[0] * HRTF_CHN * (PAD_LEN / 2 + 1),
			PAD_LEN + 2);
	}
	/*If the elevation falls on the resolution, interpolate the azimuth*/
	else if (hrtf_indices[0] == hrtf_indices[2]) {
		memcpy(convbufs, output, (PAD_LEN + 2) * sizeof(fftwf_complex));
		memcpy(convbufs + (PAD_LEN + 2), output, (PAD_LEN + 2) * sizeof(fftwf_complex));
		pointwiseMultiplication(convbufs,
			fft_hrtf + hrtf_indices[0] * HRTF_CHN * (PAD_LEN / 2 + 1),
			PAD_LEN + 2
		);
		pointwiseMultiplication(convbufs,
			fft_hrtf + hrtf_indices[1] * HRTF_CHN * (PAD_LEN / 2 + 1),
			PAD_LEN + 2
		);
		#pragma omp parallel for
		for (int i = 0; i < PAD_LEN / 2 + 1; i++) {
			convbufs[i][0] *= omegas[1];
			convbufs[(PAD_LEN + 2) + i][0] *= omegas[0];
		}
		pointwiseMultiplication(
			convbufs,
			fftw_distance_factor,
			PAD_LEN + 2
		);
		pointwiseAddition(
			convbufs, 
			convbufs + (PAD_LEN + 2), 
			output, 
		PAD_LEN + 2);
		

	}
	/*If the azimuth falls on the resolution, interpolate the elevation*/
	else if (hrtf_indices[0] == hrtf_indices[1] && hrtf_indices[0] != hrtf_indices[2]) {
		memcpy(convbufs, output, (PAD_LEN + 2) * sizeof(fftwf_complex));
		memcpy(convbufs + (PAD_LEN + 2), output, (PAD_LEN + 2) * sizeof(fftwf_complex));
		pointwiseMultiplication(convbufs,
			fft_hrtf + hrtf_indices[0] * HRTF_CHN * (PAD_LEN / 2 + 1),
			PAD_LEN + 2
		);
		pointwiseMultiplication(convbufs + (PAD_LEN + 2),
			fft_hrtf + hrtf_indices[2] * HRTF_CHN * (PAD_LEN / 2 + 1),
			PAD_LEN + 2
		);
		#pragma omp parallel for
		for (int i = 0; i < PAD_LEN / 2 + 1; i++) {
			convbufs[i][0] *= omegas[5];
			convbufs[(PAD_LEN + 2) + i][0] *= omegas[4];
		}
		pointwiseMultiplication(
			convbufs,
			fftw_distance_factor,
			PAD_LEN + 2
		);
		pointwiseMultiplication(
			convbufs + (PAD_LEN + 2),
			fftw_distance_factor,
			PAD_LEN + 2
		);
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
			memcpy(convbufs + (PAD_LEN + 2) * i, output, (PAD_LEN + 2) * sizeof(fftwf_complex));
			pointwiseMultiplication(convbufs + (PAD_LEN + 2) * i,
				fft_hrtf + hrtf_indices[i] * HRTF_CHN * (PAD_LEN / 2 + 1),
				PAD_LEN + 2
			);
			pointwiseMultiplication(
				convbufs + (PAD_LEN + 2) * i,
				fftw_distance_factor,
				PAD_LEN + 2
			);
		}
		#pragma omp parallel for
		for (int i = 0; i < PAD_LEN / 2 + 1; i++) {
			convbufs[i][0] *= omegas[5] * omegas[1];
			convbufs[(PAD_LEN + 2) + i][0] *= omegas[5] * omegas[0];
			convbufs[((PAD_LEN + 2)) * 2 + i][0] *= omegas[5] * omegas[0];
			convbufs[((PAD_LEN + 2)) * 3 + i][0] *= omegas[4] * omegas[2];
		}
		
		pointwiseAddition(
			convbufs,
			convbufs + (PAD_LEN + 2),
			output,
			PAD_LEN + 2);
		#pragma omp parallel for
		for (int i = 1; i < 4; i++) {
			pointwiseAddition(output,
				convbufs + (PAD_LEN + 2) * i,
				PAD_LEN + 2);
		}
	}
}
void SoundSource::cpuFFTInterpolate(){
	/*cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);*/
	float* output = intermediate[0];
	fftwf_execute(fftw_in_plan); /*FFT on x[0] --> fftw_intermediate*/
	complexScaling(fftw_intermediate, 1.0 / PAD_LEN, PAD_LEN / 2 + 1);
	/*Copying over for both channels*/
	memcpy(
		fftw_intermediate + PAD_LEN / 2 + 1, 
		fftw_intermediate, 
		(PAD_LEN / 2 + 1) * sizeof(fftwf_complex)
	);
	
	int old_hrtf_indices[4];
	float old_omegas[6];
	interpolationCalculations(ele, azi, hrtf_indices, omegas);
	bool xfade = false;
	if (old_azi != azi || old_ele != ele) {
		xfade = true;
		interpolationCalculations(old_ele, old_azi, old_hrtf_indices, old_omegas);
	}
	cpuCalculateDistanceFactor();
	cpuInterpolateLoops(fftw_intermediate, fftw_conv_bufs, hrtf_indices, omegas);
	
	if(xfade){
		memcpy(
			fftw_intermediate + PAD_LEN + 2,
			fftw_intermediate,
			(PAD_LEN + 2) * sizeof(fftwf_complex)
		);
		cpuInterpolateLoops(fftw_intermediate + PAD_LEN + 2, fftw_conv_bufs + (PAD_LEN + 2) * 4, old_hrtf_indices, old_omegas);
	}
	fftwf_execute(fftw_out_plan);
	if (xfade) {
		fftwf_execute_dft_c2r(
			fftw_out_plan, 
			fftw_intermediate + (PAD_LEN + 2), 
			(float*)(fftw_intermediate + (PAD_LEN + 2))
		);
		float* out1 = ((float*)fftw_intermediate) + 2 * (PAD_LEN - FRAMES_PER_BUFFER);
		float* out2 = ((float*)fftw_intermediate) + PAD_LEN + 2 + 2 * (PAD_LEN - FRAMES_PER_BUFFER);
		#pragma omp parallel for
		for(int i = 0; i < FRAMES_PER_BUFFER; i++){
			float fn = float(i) / (FRAMES_PER_BUFFER - 1.0f);
			out1[i * 2] = out1[i * 2] * (1.0f - fn) + out2[i * 2] * fn;
			out1[i * 2 + 1] = out1[i * 2 + 1] * (1.0f - fn) + out2[i * 2 + 1] * fn;
		}
	}
	num_calls++;
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
SoundSource::~SoundSource() {
	free(buf);
	CHECK_CUFFT_ERRORS(cufftDestroy(in_plan));
	CHECK_CUFFT_ERRORS(cufftDestroy(out_plan));

	for (int i = 0; i < FLIGHT_NUM; i++) {
		checkCudaErrors(cudaFreeHost(x[i]));
		checkCudaErrors(cudaFree(d_input[i]));
		checkCudaErrors(cudaFree(d_output[i]));
		checkCudaErrors(cudaFree(distance_factor[i]));
		for (int j = 0; j < STREAMS_PER_FLIGHT; j++) {
			checkCudaErrors(cudaStreamDestroy(streams[i * STREAMS_PER_FLIGHT + j]));
		}
		checkCudaErrors(cudaFree(d_convbufs[i]));
		checkCudaErrors(cudaFreeHost(intermediate[i]));
	}
	free(streams);
	fftwf_free(fftw_intermediate);
	fftwf_free(fftw_conv_bufs);
	fftwf_free(fftw_distance_factor);
}

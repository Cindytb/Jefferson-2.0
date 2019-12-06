#include "SoundSource.cuh"

SoundSource::SoundSource() {
	count = 0;
	length = 0;
	gain = 0.99074;
	hrtf_idx = 314;
	streams = new cudaStream_t[FLIGHT_NUM * 2];
	for (int i = 0; i < FLIGHT_NUM; i++) {
		/*Allocating pinned memory for incoming transfer*/
		checkCudaErrors(cudaMallocHost(x + i, (PAD_LEN + 2) * sizeof(float)));
		/*Allocating memory for the inputs*/
		checkCudaErrors(cudaMalloc(d_input + i, (PAD_LEN + 2) * sizeof(float)));
		/*Allocating memory for raw, uncropped convolution output*/
		checkCudaErrors(cudaMalloc(d_uninterleaved + i, 2 * (PAD_LEN + 2) * sizeof(float)));
		/*Allocating memory for the outputs*/
		checkCudaErrors(cudaMalloc(d_output + i, 2 * (PAD_LEN + 2) * sizeof(float)));
		/*Creating the streams*/
		checkCudaErrors(cudaStreamCreate(streams + i * 2));
		checkCudaErrors(cudaStreamCreate(streams + i * 2 + 1));
		/*Allocating pinned memory for outgoing transfer*/
		checkCudaErrors(cudaMallocHost(intermediate + i, (FRAMES_PER_BUFFER * HRTF_CHN) * sizeof(float)));
	}
	for (int i = 0; i < FLIGHT_NUM; i++) {
		for (int j = 0; j < PAD_LEN + 2; j++) {
			x[i][j] = 0.0f;
		}
	}
	CHECK_CUFFT_ERRORS(cufftPlan1d(&in_plan, PAD_LEN, CUFFT_R2C, 1));
	/*cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n,
		int *inembed, int istride, int idist,
		int *onembed, int ostride, int odist,
		cufftType type, int batch);*/
		/*stride = skip length. Ex 1 = every element, 2 = every other element*/
			/*use for interleaving???*/
		/*idist/odist is space between batches of transforms*/
			/*need to check if odist is in terms of complex numbers or floats*/
		/*inembed/onembed are for 2D/3D, num elements per dimension*/
	//CHECK_CUFFT_ERRORS(cufftPlan1d(&out_plan, PAD_LEN, CUFFT_C2R, 1));
	//int n = PAD_LEN;
	CHECK_CUFFT_ERRORS(cufftPlan1d(&out_plan, PAD_LEN, CUFFT_C2R, 2));
	//CHECK_CUFFT_ERRORS(
	//	cufftPlanMany(
	//		&out_plan, 1, &n,
	//		&n, 1, n / 2 + 1,
	//		&n, 2, 1,
	//		CUFFT_C2R, 2)
	//);
}
void SoundSource::updateInfo() {
	/*Calculate the radius, distance, elevation, and azimuth*/
	float r = std::sqrt(coordinates.x * coordinates.x + coordinates.z * coordinates.z + coordinates.y * coordinates.y);
	float horizR = std::sqrt(coordinates.x * coordinates.x + coordinates.z * coordinates.z);
	ele = (float)atan2(coordinates.y, horizR) * 180.0f / PI;

	azi = atan2(-coordinates.x / r, coordinates.z / r) * 180.0f / PI;
	if (azi < 0.0f) {
		azi += 360;
	}
	float newR = r / 100 + 1;
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

void SoundSource::fftConvolve(int blockNo) {
	float* local_d_input = d_input[blockNo % FLIGHT_NUM];
	float* local_d_output = d_output[blockNo % FLIGHT_NUM];
	float* local_d_uninterleaved = d_uninterleaved[blockNo % FLIGHT_NUM];
	cudaStream_t* local_streams = streams + (blockNo * 2 % FLIGHT_NUM);
	if (gain > 1)
		gain = 1;

	CHECK_CUFFT_ERRORS(cufftExecR2C(in_plan, (cufftReal*)local_d_input, (cufftComplex*)local_d_input));
	int numThreads = 256;
	int numBlocks = (PAD_LEN + numThreads - 1) / numThreads;
	ComplexPointwiseMul << < numBlocks, numThreads >> > (
		(cufftComplex*)local_d_input,
		(cufftComplex*)d_hrtf + hrtf_idx * (PAD_LEN + 2) * HRTF_CHN,
		(cufftComplex*)local_d_uninterleaved,
		PAD_LEN / 2 + 1
		);

	ComplexPointwiseMul << < numBlocks, numThreads >> > (
		(cufftComplex*)local_d_input,
		(cufftComplex*)d_hrtf + hrtf_idx * (PAD_LEN + 2) * HRTF_CHN + PAD_LEN + 2,
		(cufftComplex*)local_d_uninterleaved,
		PAD_LEN / 2 + 1
		);
	checkCudaErrors(cudaStreamSynchronize(local_streams[0]));
	checkCudaErrors(cudaStreamSynchronize(local_streams[1]));
	CHECK_CUFFT_ERRORS(cufftExecC2R(out_plan, (cufftComplex*)local_d_uninterleaved, local_d_uninterleaved));

	interleave << < numBlocks, numThreads >> (
		local_d_uninterleaved,
		local_d_output,
		PAD_LEN
	);
	checkCudaErrors(cudaDeviceSynchronize());
}

SoundSource::~SoundSource() {
	CHECK_CUFFT_ERRORS(cufftDestroy(in_plan));
	CHECK_CUFFT_ERRORS(cufftDestroy(out_plan));

	/*Allocating pinned memory for incoming transfer*/
	checkCudaErrors(cudaMallocHost(x + i, (PAD_LEN + 2) * sizeof(float)));
	/*Allocating memory for the inputs*/
	checkCudaErrors(cudaMalloc(d_input + i, (PAD_LEN + 2) * sizeof(float)));
	/*Allocating memory for raw, uncropped convolution output*/
	checkCudaErrors(cudaMalloc(d_uninterleaved + i, 2 * (PAD_LEN + 2) * sizeof(float)));
	/*Allocating memory for the outputs*/
	checkCudaErrors(cudaMalloc(d_output + i, 2 * (PAD_LEN + 2) * sizeof(float)));
	/*Creating the streams*/
	checkCudaErrors(cudaStreamCreate(streams + i * 2));
	checkCudaErrors(cudaStreamCreate(streams + i * 2 + 1));
	/*Allocating pinned memory for outgoing transfer*/
	checkCudaErrors(cudaMallocHost(intermediate + i, (FRAMES_PER_BUFFER * HRTF_CHN) * sizeof(float)));
}
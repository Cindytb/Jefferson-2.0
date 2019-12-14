#include "SoundSource.cuh"

SoundSource::SoundSource() {
	count = 0;
	length = 0;
	gain = 0.99074;
	hrtf_idx = 314;
	azi = 270;
	ele = 0;
	streams = new cudaStream_t[FLIGHT_NUM * 2];
	for (int i = 0; i < FLIGHT_NUM; i++) {
		/*Allocating pinned memory for incoming transfer*/
		checkCudaErrors(cudaMallocHost(x + i, (PAD_LEN + 2) * sizeof(float)));
		/*Allocating memory for the inputs*/
		checkCudaErrors(cudaMalloc(d_input + i, (PAD_LEN + 2) * sizeof(float)));
		/*Allocating memory for the outputs*/
		checkCudaErrors(cudaMalloc(d_output + i, HRTF_CHN * (PAD_LEN + 2) * sizeof(float)));
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
	float* d_input = this->d_input[blockNo % FLIGHT_NUM];
	float* d_output = this->d_output[blockNo % FLIGHT_NUM];
	cudaStream_t* streams = this->streams + (blockNo * 2 % FLIGHT_NUM);
	if (gain > 1)
		gain = 1;
	float scale = 1.0f / ((float) PAD_LEN);
	CHECK_CUFFT_ERRORS(cufftExecR2C(in_plan, (cufftReal*)d_input, (cufftComplex*)d_input));
	int numThreads = 128;
	int numBlocks = (PAD_LEN + numThreads - 1) / numThreads;
	//interpolateConvolve(blockNo);
	ComplexPointwiseMulAndScaleOutPlace << < numBlocks, numThreads, 0, streams[0] >> > (
		(cufftComplex*)d_input,
		(cufftComplex*)(d_hrtf + hrtf_idx * (PAD_LEN + 2) * HRTF_CHN),
		(cufftComplex*)d_output,
		PAD_LEN / 2 + 1,
		scale
	);

	ComplexPointwiseMulAndScaleOutPlace << < numBlocks, numThreads, 0, streams[1] >> > (
		(cufftComplex*)d_input,
		(cufftComplex*)(d_hrtf + hrtf_idx * (PAD_LEN + 2) * HRTF_CHN + PAD_LEN + 2),
		(cufftComplex*)(d_output + PAD_LEN + 2),
		PAD_LEN / 2 + 1,
		scale
	);
	checkCudaErrors(cudaStreamSynchronize(streams[0]));
	checkCudaErrors(cudaStreamSynchronize(streams[1]));
	CHECK_CUFFT_ERRORS(cufftExecC2R(out_plan, (cufftComplex*)d_output, d_output));

}

SoundSource::~SoundSource() {
	CHECK_CUFFT_ERRORS(cufftDestroy(in_plan));
	CHECK_CUFFT_ERRORS(cufftDestroy(out_plan));
	for (int i = 0; i < FLIGHT_NUM; i++) {
		cudaFreeHost(x[i]);
		cudaFree(d_input[i]);
		cudaFree(d_output[i]);
		cudaStreamDestroy(streams[i * 2]);
		cudaStreamDestroy(streams[i * 2 + 1]);
		cudaFreeHost(intermediate[i]);
	}
}
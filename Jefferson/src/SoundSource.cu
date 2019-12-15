#include "SoundSource.cuh"

SoundSource::SoundSource() {
	count = 0;
	length = 0;
	gain = 0.99074;
	hrtf_idx = 314;
	coordinates.x = 1;
	coordinates.y = 0;
	coordinates.z = 0;
	azi = 270;
	ele = 0;
	streams = new cudaStream_t[FLIGHT_NUM * STREAMS_PER_FLIGHT];
	for (int i = 0; i < FLIGHT_NUM; i++) {
		/*Allocating pinned memory for incoming transfer*/
		checkCudaErrors(cudaMallocHost(x + i, (PAD_LEN + 2) * sizeof(float)));
		/*Allocating memory for the inputs*/
		checkCudaErrors(cudaMalloc(d_input + i, (PAD_LEN + 2) * sizeof(float)));
		/*Allocating memory for the outputs*/
		checkCudaErrors(cudaMalloc(d_output + i, HRTF_CHN * (PAD_LEN + 2) * sizeof(float)));
		/*Allocating memory for the distance factor*/
		checkCudaErrors(cudaMalloc(distance_factor + i, (PAD_LEN / 2 + 1) * sizeof(cufftComplex)));
		/*Creating the streams*/
		for(int j = 0; j < STREAMS_PER_FLIGHT; j++){
			checkCudaErrors(cudaStreamCreate(streams + i * STREAMS_PER_FLIGHT + j));
		}
		checkCudaErrors(cudaMalloc(d_convbufs + i, 4 * HRTF_CHN * (PAD_LEN / 2 + 1) * sizeof(cufftComplex)));

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
	ele = round(ele);
	azi = round(azi);
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
void SoundSource::interpolationCalculations(int* hrtf_indices, float* omegas) {
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
void SoundSource::calculateDistanceFactor(int blockNo){
	cufftComplex* d_distance_factor = this->distance_factor[blockNo % FLIGHT_NUM];
	cudaStream_t* streams = this->streams + (blockNo * 2 % FLIGHT_NUM);
	float r = std::sqrt(
		coordinates.x * coordinates.x + 
		coordinates.y * coordinates.y + 
		coordinates.z * coordinates.z
	);
	r /= 5;
	float fsvs = 44100.0 / 343.0;
	float frac = 1 + fsvs * pow(r, 2);
	float N = PAD_LEN / 2 + 1;
	int numThreads = 256;
	int numBlocks = (PAD_LEN / 2 + numThreads ) / numThreads;
	generateDistanceFactor << < numThreads, numBlocks, 0, streams[1] >> > (d_distance_factor, frac, fsvs, r, N);

}
/*
This method is a slightly tweaked implementation of Jose Belloch's 
"Headphone-Based Virtual Spatialization of Sound with a GPU Accelerator"
paper from the Journal of the Audio Engineering Society,
Volume 61, No 7/8, 2013, July/August
*/
void SoundSource::interpolateConvolve(int blockNo) {
	int hrtf_indices[4];
	float omegas[6];
	interpolationCalculations(hrtf_indices, omegas);
	calculateDistanceFactor(blockNo % FLIGHT_NUM);
	
	cufftComplex* d_distance_factor = this->distance_factor[blockNo % FLIGHT_NUM];
	float* d_input = this->d_input[blockNo % FLIGHT_NUM];
	float* d_output = this->d_output[blockNo % FLIGHT_NUM];
	cufftComplex* d_convbufs = this ->d_convbufs[blockNo % FLIGHT_NUM];
	cudaStream_t* streams = this->streams + (blockNo * 2 % FLIGHT_NUM);


	fillWithZeroes(&d_output, 0, 2 * (PAD_LEN + 2));

	CHECK_CUFFT_ERRORS(cufftSetStream(in_plan, streams[0]));
	CHECK_CUFFT_ERRORS(cufftSetStream(out_plan, streams[0]));
	float scale = 1.0f / ((float)PAD_LEN);
	CHECK_CUFFT_ERRORS(cufftExecR2C(in_plan, (cufftReal*)d_input, (cufftComplex*)d_input));
	checkCudaErrors(cudaStreamSynchronize(streams[0]));
	int numThreads = 256;
	int numBlocks = (PAD_LEN / 2 + numThreads) / numThreads;
	size_t buf_size = PAD_LEN + 2;
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
				curr_scale = scale * omegas[4];
				hrtf_index = 0;
				break;
			case 2:
			case 3:
				curr_scale = scale * omegas[5];
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
	for (int i = 1; i < STREAMS_PER_FLIGHT; i++) {
		checkCudaErrors(cudaStreamSynchronize(streams[i]));
	}
	CHECK_CUFFT_ERRORS(cufftExecC2R(out_plan, (cufftComplex*)d_output, d_output));
}
void SoundSource::fftConvolve(int blockNo) {
	float* d_input = this->d_input[blockNo % FLIGHT_NUM];
	float* d_output = this->d_output[blockNo % FLIGHT_NUM];
	cudaStream_t* streams = this->streams + (blockNo * 2 % FLIGHT_NUM);
	if (gain > 1)
		gain = 1;
	float scale = 1.0f / ((float) PAD_LEN);
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
		checkCudaErrors(cudaFreeHost(x[i]));
		checkCudaErrors(cudaFree(d_input[i]));
		checkCudaErrors(cudaFree(d_output[i]));
		for(int j = 0; j < STREAMS_PER_FLIGHT; j++){
			checkCudaErrors(cudaStreamDestroy(streams[i * STREAMS_PER_FLIGHT + j]));
		}
		checkCudaErrors(cudaFree(d_convbufs[i]));
		checkCudaErrors(cudaFreeHost(intermediate[i]));
	}
}
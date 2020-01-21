#include "precision_test.cuh"


void precisionTest(Data* p) {
	SoundSource* src = (SoundSource*)&(p->all_sources[0]);
	GPUSoundSource* gsrc = &(p->all_sources[0]);
	CPUSoundSource* csrc = (CPUSoundSource*)&(p->all_sources[0]);
	float* gpu_output = new float[FRAMES_PER_BUFFER * 2];
	float* cpu_output = new float[FRAMES_PER_BUFFER * 2];

	float scale = 1.0f / ((float)PAD_LEN);
	size_t buf_size = PAD_LEN + 2;
	size_t complex_buf_size = buf_size / 2;

	float* gpu_distance = new float[buf_size];
	float* gpu_fft_in = new float[buf_size];
	float* gpu_ifft = new float[2 * buf_size];
	float* gpu_fft_scaled = new float[buf_size];
	float* gpu_conv = new float[2 * buf_size];
	/////////////////////////////////////////////////////////
	// COPY INCOMING BLOCK
	/////////////////////////////////////////////////////////

	// GPU
	p->blockNo = 0;
	src->count = 0;
	for (int i = 0; i < PAD_LEN + 2; i++) {
		gsrc->x[i] = 0.0f;
		csrc->x[i] = 0.0f;
	}
	fillWithZeroesKernel(gsrc->d_output, 2 * buf_size, gsrc->streams[0]);
	gsrc->copyIncomingBlock(0);
	gsrc->sendBlock(0);
	// CPU
	src->count = 0;
	memcpy(
		csrc->x + (PAD_LEN - FRAMES_PER_BUFFER),  /*Go to the end and work backwards*/
		csrc->buf,
		FRAMES_PER_BUFFER * sizeof(float));


	checkCudaErrors(cudaMemcpy(gpu_fft_in, gsrc->d_input, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_fft_in, (float*)csrc->x, PAD_LEN)) {
		printf("ERROR: Inaccurate Input Copy\n");
	}
	else {
		printf("Accurate Input Copy\n");
	}

	/////////////////////////////////////////////////////////
	// CALCULATE WEIGHTS
	/////////////////////////////////////////////////////////
	int blockNo = 0;
	float ele = src->ele;
	float azi = src->azi;
	int hrtf_indices[4];
	float omegas[6];
	src->interpolationCalculations(ele, azi, hrtf_indices, omegas);

	/////////////////////////////////////////////////////////
	// CALCULATE DISTANCE FACTOR
	/////////////////////////////////////////////////////////

	// GPU
	gsrc->gpuCalculateDistanceFactor(blockNo);

	// CPU
	csrc->calculateDistanceFactor();

	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_distance, gsrc->d_distance_factor, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_distance, (float*)csrc->distance_factor, PAD_LEN + 2)) {
		printf("ERROR: Inaccurate Distance calculations\n");
	}
	else {
		printf("Distance calculations successfull\n");
	}



	/////////////////////////////////////////////////////////
	// FFT IN
	/////////////////////////////////////////////////////////

	// GPU
	cufftComplex* d_distance_factor = gsrc->d_distance_factor;
	float* d_input = gsrc->d_input;
	float* d_output = gsrc->d_output;
	cufftComplex* d_convbufs = gsrc->d_convbufs;
	CHECK_CUFFT_ERRORS(cufftExecR2C(gsrc->plans[0], (cufftReal*)d_input, (cufftComplex*)d_input));


	// CPU
	fftwf_execute(csrc->in_plan); /*FFT on x --> intermediate*/

	// Precision Check
	checkCudaErrors(cudaMemcpy(gpu_fft_in, d_input, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_fft_in, (float*)csrc->intermediate, PAD_LEN + 2)) {
		printf("ERROR: Inaccurate FFT\n");
	}
	else {
		printf("FFT successful\n");
	}


	/////////////////////////////////////////////////////////
	// SCALING
	/////////////////////////////////////////////////////////

	// GPU
	int numThreads = 128;
	int numBlocks = (buf_size + numThreads - 1) / numThreads;

	MyFloatScale << < numBlocks, numThreads >> > (d_input, scale, buf_size);
	// CPU
	complexScaling(csrc->intermediate, scale, complex_buf_size);

	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_fft_scaled, d_input, buf_size * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_fft_scaled, (float*)csrc->intermediate, buf_size)) {
		printf("ERROR: Inaccurate Scaling\n");
	}
	else {
		printf("Scaling successful\n");
	}

	/*Copying over for both channels*/
	checkCudaErrors(cudaMemcpy(d_input + buf_size, d_input, buf_size * sizeof(float), cudaMemcpyDeviceToDevice));
	memcpy(
		csrc->intermediate + complex_buf_size,
		csrc->intermediate,
		complex_buf_size * sizeof(fftwf_complex)
	);
	/////////////////////////////////////////////////////////
	// CASE 1 CONVOLUTIONS
	/////////////////////////////////////////////////////////
	gsrc->caseOneConvolve(d_input, d_output, d_convbufs, d_distance_factor, gsrc->streams, hrtf_indices);
	//CPU
	csrc->caseOneConvolve(csrc->intermediate, hrtf_indices);

	// Precision Check
	checkCudaErrors(cudaMemcpy(gpu_conv, d_output, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)csrc->intermediate, buf_size)) {
		printf("ERROR: Inaccurate Case 1 Convolutions\n");
	}
	else {
		printf("Case 1 Convolutions successful\n");
	}

	/////////////////////////////////////////////////////////
	// FFT OUT
	/////////////////////////////////////////////////////////
	CHECK_CUFFT_ERRORS(cufftExecC2R(gsrc->plans[1], (cufftComplex*)d_output, d_output));
	fftwf_execute(csrc->out_plan);
	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_ifft, d_output, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_ifft, (float*)csrc->intermediate, buf_size)) {
		printf("ERROR: Inaccurate Case 1 IFFT\n");
	}
	else {
		printf("Case 1 IFFT successful\n");
	}


	/*Receive*/
	gsrc->receiveBlock(0);
	memcpy(gpu_output, gsrc->intermediate, FRAMES_PER_BUFFER * 2 * sizeof(float));
	memcpy(cpu_output, ((float*)csrc->intermediate) + 2 * (PAD_LEN - FRAMES_PER_BUFFER), FRAMES_PER_BUFFER * 2 * sizeof(float));
	if (precisionChecking(gpu_output, cpu_output, FRAMES_PER_BUFFER * 2)) {
		printf("ERROR: Inaccurate case 1 output\n");
	}
	else {
		printf("Successfully accurate case 1 output\n");
	}
	/////////////////////////////////////////////////////////
	// CASE 2 CONVOLUTIONS - Interpolate Azimuth
	/////////////////////////////////////////////////////////
	fftwf_execute(csrc->in_plan); /*FFT on x --> intermediate*/
	complexScaling(csrc->intermediate, scale, complex_buf_size);
	memcpy(
		csrc->intermediate + complex_buf_size,
		csrc->intermediate,
		complex_buf_size * sizeof(fftwf_complex)
	);
	//checkCudaErrors(cudaMemcpy(d_input + buf_size, d_input, buf_size * sizeof(float), cudaMemcpyDeviceToDevice));
	// Precision Check
	checkCudaErrors(cudaMemcpy(gpu_fft_in, d_input, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_fft_in, (float*)csrc->intermediate, buf_size)) {
		printf("ERROR: Inaccurate Scaled FFT\n");
	}
	else {
		printf("Scaled FFT successful\n");
	}
	checkCudaErrors(cudaMemcpy(gpu_fft_in, d_input + buf_size, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_fft_in, (float*)(csrc->intermediate + complex_buf_size), buf_size)) {
		printf("ERROR: Inaccurate Scaled FFT\n");
	}
	else {
		printf("Scaled FFT successful\n");
	}
	azi = 3;
	src->interpolationCalculations(ele, azi, hrtf_indices, omegas);
	checkCudaErrors(cudaMemcpy(gpu_fft_in, d_fft_hrtf + hrtf_indices[0] * complex_buf_size * HRTF_CHN, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_fft_in, (float*)(fft_hrtf + hrtf_indices[0] * HRTF_CHN * complex_buf_size), buf_size, 1e-6)) {
		printf("ERROR: Inaccurate HRTF\n");
	}
	else {
		printf("Accurate HRTF \n");
	}
	//GPU
	fillWithZeroesKernel(d_output, 2 * buf_size, gsrc->streams[0]);
	checkCudaErrors(cudaDeviceSynchronize());
	gsrc->caseTwoConvolve(d_input, d_output, d_convbufs, d_distance_factor, gsrc->streams, hrtf_indices, omegas);

	//CPU
	csrc->caseTwoConvolve(csrc->intermediate, csrc->conv_bufs, hrtf_indices, omegas);


	checkCudaErrors(cudaMemcpy(gpu_conv, d_convbufs, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)csrc->conv_bufs, buf_size)) {
		printf("ERROR: Inaccurate Case 2 Convbuf 1\n");
	}
	else {
		printf("Case 2 Convolutions Convbuf 1 successful\n");
	}
	checkCudaErrors(cudaMemcpy(gpu_conv, d_convbufs + buf_size, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->conv_bufs + buf_size), 2 * buf_size)) {
		printf("ERROR: Inaccurate Case 2 Convbuf 2\n");
	}
	else {
		printf("Case 2 Convolutions Convbuf 2 successful\n");
	}
	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_conv, d_output, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)csrc->intermediate, 2 * buf_size)) {
		printf("ERROR: Inaccurate Case 2 Convolution output\n");
	}
	else {
		printf("Case 2 Convolution output successful\n");
	}
	/////////////////////////////////////////////////////////
	// FFT OUT
	/////////////////////////////////////////////////////////
	CHECK_CUFFT_ERRORS(cufftExecC2R(gsrc->plans[1], (cufftComplex*)d_output, d_output));
	fftwf_execute(csrc->out_plan);
	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_ifft, d_output, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_ifft, (float*)csrc->intermediate, buf_size)) {
		printf("ERROR: Inaccurate Case 2 IFFT\n");
	}
	else {
		printf("Case 2 IFFT successful\n");
	}

	/*Receive*/
	gsrc->receiveBlock(0);
	memcpy(gpu_output, gsrc->intermediate, FRAMES_PER_BUFFER * 2 * sizeof(float));
	memcpy(cpu_output, ((float*)csrc->intermediate) + 2 * (PAD_LEN - FRAMES_PER_BUFFER), FRAMES_PER_BUFFER * 2 * sizeof(float));
	if (precisionChecking(gpu_output, cpu_output, FRAMES_PER_BUFFER * 2)) {
		printf("ERROR: Inaccurate Case 2 output\n");
	}
	else {
		printf("Successful Case 2 accurate output\n");
	}

	/////////////////////////////////////////////////////////
	// CASE 3 CONVOLUTIONS - Interpolate Elevation
	/////////////////////////////////////////////////////////
	fftwf_execute(csrc->in_plan); /*FFT on x --> intermediate*/
	complexScaling(csrc->intermediate, scale, complex_buf_size);
	memcpy(
		csrc->intermediate + complex_buf_size,
		csrc->intermediate,
		complex_buf_size * sizeof(fftwf_complex)
	);
	ele = 5;
	azi = 0;
	src->interpolationCalculations(ele, azi, hrtf_indices, omegas);
	checkCudaErrors(cudaMemcpy(gpu_fft_in, d_fft_hrtf + hrtf_indices[0] * complex_buf_size * HRTF_CHN, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_fft_in, (float*)(fft_hrtf + hrtf_indices[0] * HRTF_CHN * complex_buf_size), buf_size, 1e-6)) {
		printf("ERROR: Inaccurate HRTF\n");
	}
	else {
		printf("Accurate HRTF \n");
	}
	//GPU
	fillWithZeroesKernel(d_output, 2 * buf_size, gsrc->streams[0]);
	checkCudaErrors(cudaDeviceSynchronize());
	gsrc->caseThreeConvolve(d_input, d_output, d_convbufs, d_distance_factor, gsrc->streams, hrtf_indices, omegas);

	//CPU
	csrc->caseThreeConvolve(csrc->intermediate, csrc->conv_bufs, hrtf_indices, omegas);
	checkCudaErrors(cudaMemcpy(gpu_conv, d_convbufs, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)csrc->conv_bufs, buf_size)) {
		printf("ERROR: Inaccurate Case 3 Convbuf 1\n");
	}
	else {
		printf("Case 3 Convolutions Convbuf 1 successful\n");
	}
	checkCudaErrors(cudaMemcpy(gpu_conv, d_convbufs + buf_size, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->conv_bufs + buf_size), 2 * buf_size)) {
		printf("ERROR: Inaccurate Case 3 Convbuf 2\n");
	}
	else {
		printf("Case 3 Convolutions Convbuf 2 successful\n");
	}
	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_conv, d_output, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)csrc->intermediate, 2 * buf_size)) {
		printf("ERROR: Inaccurate Case 3 Convolution output\n");
	}
	else {
		printf("Case 3 Convolution output successful\n");
	}
	/////////////////////////////////////////////////////////
	// FFT OUT
	/////////////////////////////////////////////////////////
	CHECK_CUFFT_ERRORS(cufftExecC2R(gsrc->plans[1], (cufftComplex*)d_output, d_output));
	fftwf_execute(csrc->out_plan);
	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_ifft, d_output, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_ifft, (float*)csrc->intermediate, buf_size)) {
		printf("ERROR: Inaccurate Case 3 IFFT\n");
	}
	else {
		printf("Case 3 IFFT successful\n");
	}

	/*Receive*/
	gsrc->receiveBlock(0);
	memcpy(gpu_output, gsrc->intermediate, FRAMES_PER_BUFFER * 2 * sizeof(float));
	memcpy(cpu_output, ((float*)csrc->intermediate) + 2 * (PAD_LEN - FRAMES_PER_BUFFER), FRAMES_PER_BUFFER * 2 * sizeof(float));
	if (precisionChecking(gpu_output, cpu_output, FRAMES_PER_BUFFER * 2)) {
		printf("ERROR: Inaccurate Case 3 output\n");
	}
	else {
		printf("Successful Case 3 accurate output\n");
	}

	/////////////////////////////////////////////////////////
	// CASE 4 CONVOLUTIONS - Worst Case Scenario
	/////////////////////////////////////////////////////////
	fftwf_execute(csrc->in_plan); /*FFT on x --> intermediate*/
	complexScaling(csrc->intermediate, scale, complex_buf_size);
	memcpy(
		csrc->intermediate + complex_buf_size,
		csrc->intermediate,
		complex_buf_size * sizeof(fftwf_complex)
	);
	ele = 5;
	azi = 3;
	src->interpolationCalculations(ele, azi, hrtf_indices, omegas);

	//GPU
	fillWithZeroesKernel(d_output, 2 * buf_size, gsrc->streams[0]);
	checkCudaErrors(cudaDeviceSynchronize());
	gsrc->caseFourConvolve(d_input, d_output, d_convbufs, d_distance_factor, gsrc->streams, hrtf_indices, omegas);

	//CPU
	csrc->caseFourConvolve(csrc->intermediate, csrc->conv_bufs, hrtf_indices, omegas);

	checkCudaErrors(cudaMemcpy(gpu_conv, d_convbufs, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)csrc->conv_bufs, buf_size)) {
		printf("ERROR: Inaccurate Case 4 Convbuf 1\n");
	}
	else {
		printf("Case 4 Convolutions Convbuf 1 successful\n");
	}
	checkCudaErrors(cudaMemcpy(gpu_conv, d_convbufs + buf_size, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->conv_bufs + buf_size), 2 * buf_size)) {
		printf("ERROR: Inaccurate Case 4 Convbuf 2\n");
	}
	else {
		printf("Case 4 Convolutions Convbuf 2 successful\n");
	}
	checkCudaErrors(cudaMemcpy(gpu_conv, d_convbufs + 2 * buf_size, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->conv_bufs + 2 * buf_size), 2 * buf_size)) {
		printf("ERROR: Inaccurate Case 4 Convbuf 3\n");
	}
	else {
		printf("Case 4 Convolutions Convbuf 3 successful\n");
	}
	checkCudaErrors(cudaMemcpy(gpu_conv, d_convbufs + 3 * buf_size, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->conv_bufs + 3 * buf_size), 2 * buf_size)) {
		printf("ERROR: Inaccurate Case 4 Convbuf 4\n");
	}
	else {
		printf("Case 4 Convolutions Convbuf 4 successful\n");
	}
	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_conv, d_output, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)csrc->intermediate, 2 * buf_size)) {
		printf("ERROR: Inaccurate Case 4 Convolution output\n");
	}
	else {
		printf("Case 4 Convolution output successful\n");
	}
	/////////////////////////////////////////////////////////
	// FFT OUT
	/////////////////////////////////////////////////////////
	CHECK_CUFFT_ERRORS(cufftExecC2R(gsrc->plans[1], (cufftComplex*)d_output, d_output));
	fftwf_execute(csrc->out_plan);
	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_ifft, d_output, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_ifft, (float*)csrc->intermediate, buf_size)) {
		printf("ERROR: Inaccurate Case 4 IFFT\n");
	}
	else {
		printf("Case 4 IFFT successful\n");
	}

	/*Receive*/
	gsrc->receiveBlock(0);
	memcpy(gpu_output, gsrc->intermediate, FRAMES_PER_BUFFER * 2 * sizeof(float));
	memcpy(cpu_output, ((float*)csrc->intermediate) + 2 * (PAD_LEN - FRAMES_PER_BUFFER), FRAMES_PER_BUFFER * 2 * sizeof(float));
	if (precisionChecking(gpu_output, cpu_output, FRAMES_PER_BUFFER * 2)) {
		printf("ERROR: Inaccurate Case 4 output\n");
	}
	else {
		printf("Successful Case 4 accurate output\n");
	}
	for (int i = 0; i < PAD_LEN + 2; i++) {
		gsrc->x[i] = 0.0f;
		csrc->x[i] = 0.0f;
	}
	for (int i = 0; i < FRAMES_PER_BUFFER * 2; i++) {
		gsrc->intermediate[i] = 0.0f;
	}
	p->blockNo = 0;
	src->count = 0;
	delete[] gpu_output;
	delete[] cpu_output;
	delete[] gpu_conv;
	delete[] gpu_distance;
	delete[] gpu_ifft;
	delete[] gpu_fft_in;
	delete[] gpu_fft_scaled;

}

void xfadePrecisionTest(Data* p) {
	SoundSource* src = (SoundSource*)&(p->all_sources[0]);
	GPUSoundSource* gsrc = &(p->all_sources[0]);
	CPUSoundSource* csrc = (CPUSoundSource*)&(p->all_sources[0]);
	float* gpu_output = new float[FRAMES_PER_BUFFER * 2];
	float* cpu_output = new float[FRAMES_PER_BUFFER * 2];

	float scale = 1.0f / ((float)PAD_LEN);
	size_t buf_size = PAD_LEN + 2;
	size_t complex_buf_size = buf_size / 2;

	float* gpu_distance = new float[buf_size];
	float* gpu_fft_in = new float[buf_size];
	float* gpu_ifft = new float[2 * buf_size];
	float* gpu_fft_scaled = new float[buf_size];
	float* gpu_conv = new float[2 * buf_size];

	/////////////////////////////////////////////////////////
	// COPY INCOMING BLOCK
	/////////////////////////////////////////////////////////

	// GPU
	p->blockNo = 0;
	src->count = 0;
	for (int i = 0; i < PAD_LEN + 2; i++) {
		gsrc->x[i] = 0.0f;
	}
	fillWithZeroesKernel(gsrc->d_output, 2 * buf_size, gsrc->streams[0]);
	gsrc->copyIncomingBlock(0);
	gsrc->sendBlock(0);
	// CPU
	src->count = 0;
	memcpy(
		csrc->x + (PAD_LEN - FRAMES_PER_BUFFER),  /*Go to the end and work backwards*/
		csrc->buf,
		FRAMES_PER_BUFFER * sizeof(float));


	checkCudaErrors(cudaMemcpy(gpu_fft_in, gsrc->d_input, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_fft_in, (float*)csrc->x, PAD_LEN)) {
		printf("ERROR: Inaccurate Input Copy\n");
	}
	else {
		printf("Accurate Input Copy\n");
	}

	/////////////////////////////////////////////////////////
	// CALCULATE WEIGHTS
	/////////////////////////////////////////////////////////
	int blockNo = 0;
	src->old_azi = 0;
	src->old_ele = 0;
	src->ele = 10;
	src->azi = 5;
	float ele = src->ele;
	float azi = src->azi;
	int hrtf_indices[4];
	float omegas[6];
	int old_hrtf_indices[4];
	float old_omegas[6];
	src->interpolationCalculations(ele, azi, hrtf_indices, omegas);
	src->interpolationCalculations(src->old_ele, src->old_azi, old_hrtf_indices, old_omegas);

	/////////////////////////////////////////////////////////
	// CALCULATE DISTANCE FACTOR
	/////////////////////////////////////////////////////////

	// GPU
	gsrc->gpuCalculateDistanceFactor(blockNo);

	// CPU
	csrc->calculateDistanceFactor();

	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_distance, gsrc->d_distance_factor, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_distance, (float*)csrc->distance_factor, PAD_LEN + 2)) {
		printf("ERROR: Inaccurate Distance calculations\n");
	}
	else {
		printf("Distance calculations successfull\n");
	}



	/////////////////////////////////////////////////////////
	// FFT IN
	/////////////////////////////////////////////////////////

	// GPU
	cufftComplex* d_distance_factor = gsrc->d_distance_factor;
	float* d_input = gsrc->d_input;
	float* d_output = gsrc->d_output;
	float* d_output2 = gsrc->d_output2;
	cufftComplex* d_convbufs = gsrc->d_convbufs;
	cufftComplex* d_convbufs2 = gsrc->d_convbufs + 4 * (PAD_LEN + 2);
	CHECK_CUFFT_ERRORS(cufftExecR2C(gsrc->plans[blockNo * 3], (cufftReal*)d_input, (cufftComplex*)d_input));


	// CPU
	fftwf_execute(csrc->in_plan); /*FFT on x --> intermediate*/

	// Precision Check
	checkCudaErrors(cudaMemcpy(gpu_fft_in, d_input, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_fft_in, (float*)csrc->intermediate, PAD_LEN + 2)) {
		printf("ERROR: Inaccurate FFT\n");
	}
	else {
		printf("FFT successful\n");
	}


	/////////////////////////////////////////////////////////
	// SCALING
	/////////////////////////////////////////////////////////

	// GPU
	int numThreads = 128;
	int numBlocks = (buf_size + numThreads - 1) / numThreads;

	MyFloatScale << < numBlocks, numThreads >> > (d_input, scale, buf_size);
	// CPU
	complexScaling(csrc->intermediate, scale, complex_buf_size);

	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_fft_scaled, d_input, buf_size * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_fft_scaled, (float*)csrc->intermediate, buf_size)) {
		printf("ERROR: Inaccurate Scaling\n");
	}
	else {
		printf("Scaling successful\n");
	}

	/*Copying over for both channels*/
	checkCudaErrors(cudaMemcpy(d_input + buf_size, d_input, buf_size * sizeof(float), cudaMemcpyDeviceToDevice));
	memcpy(
		csrc->intermediate + complex_buf_size,
		csrc->intermediate,
		complex_buf_size * sizeof(fftwf_complex)
	);
	/*Copying over for xfade*/
	memcpy(
		csrc->intermediate + buf_size,
		csrc->intermediate,
		(PAD_LEN + 2) * sizeof(fftwf_complex)
	);
	/////////////////////////////////////////////////////////
	// CASE 1 CONVOLUTIONS
	/////////////////////////////////////////////////////////
	gsrc->caseOneConvolve(d_input, d_output, d_convbufs, d_distance_factor, gsrc->streams, old_hrtf_indices);
	gsrc->caseOneConvolve(d_input, d_output2, d_convbufs2, d_distance_factor, gsrc->streams, hrtf_indices);
	//CPU
	csrc->caseOneConvolve(csrc->intermediate, old_hrtf_indices);
	csrc->caseOneConvolve(csrc->intermediate + buf_size, hrtf_indices);

	// Precision Check
	checkCudaErrors(cudaMemcpy(gpu_conv, d_output, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)csrc->intermediate, buf_size)) {
		printf("ERROR: Inaccurate Case 1 Convolutions\n");
	}
	else {
		printf("Case 1 Convolutions successful\n");
	}

	checkCudaErrors(cudaMemcpy(gpu_conv, d_output2, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->intermediate + buf_size), buf_size)) {
		printf("ERROR: Inaccurate Case 1 Convolutions\n");
	}
	else {
		printf("Case 1 Convolutions successful\n");
	}
	/////////////////////////////////////////////////////////
	// FFT OUT
	/////////////////////////////////////////////////////////
	CHECK_CUFFT_ERRORS(cufftExecC2R(gsrc->plans[1], (cufftComplex*)d_output, d_output));
	CHECK_CUFFT_ERRORS(cufftExecC2R(gsrc->plans[2], (cufftComplex*)d_output2, d_output2));

	fftwf_execute(csrc->out_plan);
	fftwf_execute_dft_c2r(
		csrc->out_plan,
		csrc->intermediate + (PAD_LEN + 2),
		(float*)(csrc->intermediate + PAD_LEN + 2)
	);
	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_ifft, d_output, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_ifft, (float*)csrc->intermediate, buf_size)) {
		printf("ERROR: Inaccurate Case 1 IFFT\n");
	}
	else {
		printf("Case 1 IFFT successful\n");
	}
	checkCudaErrors(cudaMemcpy(gpu_ifft, d_output2, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_ifft, (float*)(csrc->intermediate + buf_size), buf_size)) {
		printf("ERROR: Inaccurate Case 1 IFFT\n");
	}
	else {
		printf("Case 1 IFFT successful\n");
	}

	/////////////////////////////////////////////////////////
	// CROSSFADE
	/////////////////////////////////////////////////////////
	numThreads = FRAMES_PER_BUFFER;
	crossFade << <numThreads, 1 >> > (
		d_output + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
		d_output2 + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
		FRAMES_PER_BUFFER);

	float* out1 = ((float*)csrc->intermediate) + 2 * (PAD_LEN - FRAMES_PER_BUFFER);
	float* out2 = ((float*)csrc->intermediate) + 2 * buf_size + 2 * (PAD_LEN - FRAMES_PER_BUFFER);//PAD_LEN + 2 + 2 * (PAD_LEN - FRAMES_PER_BUFFER);
#pragma omp parallel for
	for (int i = 0; i < FRAMES_PER_BUFFER; i++) {
		float fn = float(i) / (FRAMES_PER_BUFFER - 1.0f);
		out1[i * 2] = out1[i * 2] * fn + out2[i * 2] * (1.0f - fn);
		out1[i * 2 + 1] = out1[i * 2 + 1] * fn + out2[i * 2 + 1] * (1.0f - fn);
	}

	checkCudaErrors(cudaMemcpy(gpu_ifft, d_output, buf_size * 2 * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_ifft, (float*)(csrc->intermediate), buf_size * 2)) {
		printf("ERROR: Inaccurate Crossfade\n");
	}
	else {
		printf("Case 1 Crossfade successful\n");
	}
	/*Receive*/
	gsrc->receiveBlock(0);
	memcpy(gpu_output, gsrc->intermediate, FRAMES_PER_BUFFER * 2 * sizeof(float));
	memcpy(cpu_output, ((float*)csrc->intermediate) + 2 * (PAD_LEN - FRAMES_PER_BUFFER), FRAMES_PER_BUFFER * 2 * sizeof(float));
	if (precisionChecking(gpu_output, cpu_output, FRAMES_PER_BUFFER * 2)) {
		printf("ERROR: Inaccurate case 1 output\n");
	}
	else {
		printf("Successfully accurate case 1 output\n");
	}
	/////////////////////////////////////////////////////////
	// CASE 2 CONVOLUTIONS - Interpolate Azimuth
	/////////////////////////////////////////////////////////
	fftwf_execute(csrc->in_plan); /*FFT on x --> intermediate*/
	complexScaling(csrc->intermediate, scale, complex_buf_size);
	memcpy(
		csrc->intermediate + complex_buf_size,
		csrc->intermediate,
		complex_buf_size * sizeof(fftwf_complex)
	);
	/*Copying over for xfade*/
	memcpy(
		csrc->intermediate + buf_size,
		csrc->intermediate,
		buf_size * sizeof(fftwf_complex)
	);

	// Precision Check
	checkCudaErrors(cudaMemcpy(gpu_fft_in, d_input, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_fft_in, (float*)csrc->intermediate, buf_size)) {
		printf("ERROR: Inaccurate Scaled FFT\n");
	}
	else {
		printf("Scaled FFT successful\n");
	}
	checkCudaErrors(cudaMemcpy(gpu_fft_in, d_input + buf_size, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_fft_in, (float*)(csrc->intermediate + complex_buf_size), buf_size)) {
		printf("ERROR: Inaccurate Scaled FFT\n");
	}
	else {
		printf("Scaled FFT successful\n");
	}
	src->interpolationCalculations(0, 8, hrtf_indices, omegas);
	src->interpolationCalculations(0, 3, old_hrtf_indices, old_omegas);
	for (int i = 0; i < 4; i++) {
		checkCudaErrors(cudaMemcpy(gpu_fft_in, d_fft_hrtf + hrtf_indices[i] * complex_buf_size * HRTF_CHN, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
		if (precisionChecking(gpu_fft_in, (float*)(fft_hrtf + hrtf_indices[i] * HRTF_CHN * complex_buf_size), buf_size, 1e-6)) {
			printf("ERROR: Inaccurate HRTF\n");
		}
		else {
			printf("Accurate HRTF \n");
		}
	}
	for (int i = 0; i < 4; i++) {
		checkCudaErrors(cudaMemcpy(gpu_fft_in, d_fft_hrtf + old_hrtf_indices[i] * complex_buf_size * HRTF_CHN, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
		if (precisionChecking(gpu_fft_in, (float*)(fft_hrtf + old_hrtf_indices[i] * HRTF_CHN * complex_buf_size), buf_size, 1e-6)) {
			printf("ERROR: Inaccurate HRTF\n");
		}
		else {
			printf("Accurate HRTF \n");
		}
	}
	//GPU
	fillWithZeroesKernel(d_output, 2 * buf_size, gsrc->streams[0]);
	fillWithZeroesKernel(d_output2, 2 * buf_size, gsrc->streams[0]);
	checkCudaErrors(cudaDeviceSynchronize());
	gsrc->caseTwoConvolve(d_input, d_output, d_convbufs, d_distance_factor, gsrc->streams, old_hrtf_indices, old_omegas);
	gsrc->caseTwoConvolve(d_input, d_output2, d_convbufs2, d_distance_factor, gsrc->streams, hrtf_indices, omegas);
	//CPU
	csrc->caseTwoConvolve(csrc->intermediate, csrc->conv_bufs, old_hrtf_indices, old_omegas);
	csrc->caseTwoConvolve(csrc->intermediate + buf_size, csrc->conv_bufs + buf_size * 2, hrtf_indices, omegas);


	//OLD HRTFS
	checkCudaErrors(cudaMemcpy(gpu_conv, d_convbufs, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)csrc->conv_bufs, 2 * buf_size)) {
		printf("ERROR: Inaccurate Case 2 Convbuf 1\n");
	}
	else {
		printf("Case 2 Convolutions Convbuf 1 successful\n");
	}
	checkCudaErrors(cudaMemcpy(gpu_conv, d_convbufs + buf_size, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->conv_bufs + buf_size), 2 * buf_size)) {
		printf("ERROR: Inaccurate Case 2 Convbuf 2\n");
	}
	else {
		printf("Case 2 Convolutions Convbuf 2 successful\n");
	}
	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_conv, d_output, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)csrc->intermediate, 2 * buf_size)) {
		printf("ERROR: Inaccurate Case 2 Convolution output\n");
	}
	else {
		printf("Case 2 Convolution output successful\n");
	}

	//NEW HRTFS
	checkCudaErrors(cudaMemcpy(gpu_conv, d_convbufs2, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->conv_bufs + buf_size * 2), buf_size)) {
		printf("ERROR: Inaccurate Case 2 Convbuf 1\n");
	}
	else {
		printf("Case 2 Convolutions Convbuf 1 successful\n");
	}
	checkCudaErrors(cudaMemcpy(gpu_conv, d_convbufs2 + buf_size, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->conv_bufs + 3 * buf_size), 2 * buf_size)) {
		printf("ERROR: Inaccurate Case 2 Convbuf 2\n");
	}
	else {
		printf("Case 2 Convolutions Convbuf 2 successful\n");
	}
	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_conv, d_output2, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->intermediate + buf_size), 2 * buf_size)) {
		printf("ERROR: Inaccurate Case 2 Convolution output\n");
	}
	else {
		printf("Case 2 Convolution output successful\n");
	}

	// Precision Check
	checkCudaErrors(cudaMemcpy(gpu_conv, d_output2, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->intermediate + buf_size), buf_size)) {
		printf("ERROR: Inaccurate Case 2 Convolutions\n");
	}
	else {
		printf("Case 2 Convolutions successful\n");
	}

	checkCudaErrors(cudaMemcpy(gpu_conv, d_output2, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->intermediate + buf_size), buf_size)) {
		printf("ERROR: Inaccurate Case 2 Convolutions\n");
	}
	else {
		printf("Case 2 Convolutions successful\n");
	}
	/////////////////////////////////////////////////////////
	// FFT OUT
	/////////////////////////////////////////////////////////
	CHECK_CUFFT_ERRORS(cufftExecC2R(gsrc->plans[1], (cufftComplex*)d_output, d_output));
	CHECK_CUFFT_ERRORS(cufftExecC2R(gsrc->plans[2], (cufftComplex*)d_output2, d_output2));

	fftwf_execute(csrc->out_plan);
	fftwf_execute_dft_c2r(
		csrc->out_plan,
		csrc->intermediate + (PAD_LEN + 2),
		(float*)(csrc->intermediate + PAD_LEN + 2)
	);
	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_ifft, d_output, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_ifft, (float*)csrc->intermediate, buf_size)) {
		printf("ERROR: Inaccurate Case 2 IFFT\n");
	}
	else {
		printf("Case 2 IFFT successful\n");
	}
	checkCudaErrors(cudaMemcpy(gpu_ifft, d_output2, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_ifft, (float*)(csrc->intermediate + buf_size), buf_size)) {
		printf("ERROR: Inaccurate Case 2 IFFT\n");
	}
	else {
		printf("Case 2 IFFT successful\n");
	}

	/////////////////////////////////////////////////////////
	// CROSSFADE
	/////////////////////////////////////////////////////////
	numThreads = FRAMES_PER_BUFFER;
	crossFade << <numThreads, 1 >> > (
		d_output + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
		d_output2 + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
		FRAMES_PER_BUFFER);

	out1 = ((float*)csrc->intermediate) + 2 * (PAD_LEN - FRAMES_PER_BUFFER);
	out2 = ((float*)csrc->intermediate) + 2 * PAD_LEN + 4 + 2 * (PAD_LEN - FRAMES_PER_BUFFER);//PAD_LEN + 2 + 2 * (PAD_LEN - FRAMES_PER_BUFFER);
#pragma omp parallel for
	for (int i = 0; i < FRAMES_PER_BUFFER; i++) {
		float fn = float(i) / (FRAMES_PER_BUFFER - 1.0f);
		out1[i * 2] = out1[i * 2] * fn + out2[i * 2] * (1.0f - fn);
		out1[i * 2 + 1] = out1[i * 2 + 1] * fn + out2[i * 2 + 1] * (1.0f - fn);
	}

	checkCudaErrors(cudaMemcpy(gpu_ifft, d_output, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_ifft, (float*)(csrc->intermediate), buf_size)) {
		printf("ERROR: Inaccurate Crossfade\n");
	}
	else {
		printf("Case 2 Crossfade successful\n");
	}

	/*Receive*/
	gsrc->receiveBlock(0);
	memcpy(gpu_output, gsrc->intermediate, FRAMES_PER_BUFFER * 2 * sizeof(float));
	memcpy(cpu_output, ((float*)csrc->intermediate) + 2 * (PAD_LEN - FRAMES_PER_BUFFER), FRAMES_PER_BUFFER * 2 * sizeof(float));
	if (precisionChecking(gpu_output, cpu_output, FRAMES_PER_BUFFER * 2)) {
		printf("ERROR: Inaccurate Case 2 output\n");
	}
	else {
		printf("Successful Case 2 accurate output\n");
	}

	/////////////////////////////////////////////////////////
	// CASE 3 CONVOLUTIONS - Interpolate Elevation
	/////////////////////////////////////////////////////////
	fftwf_execute(csrc->in_plan); /*FFT on x --> intermediate*/
	complexScaling(csrc->intermediate, scale, complex_buf_size);
	memcpy(
		csrc->intermediate + complex_buf_size,
		csrc->intermediate,
		complex_buf_size * sizeof(fftwf_complex)
	);
	/*Copying over for xfade*/
	memcpy(
		csrc->intermediate + buf_size,
		csrc->intermediate,
		buf_size * sizeof(fftwf_complex)
	);

	// Precision Check
	checkCudaErrors(cudaMemcpy(gpu_fft_in, d_input, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_fft_in, (float*)csrc->intermediate, buf_size)) {
		printf("ERROR: Inaccurate Scaled FFT\n");
	}
	else {
		printf("Scaled FFT successful\n");
	}
	checkCudaErrors(cudaMemcpy(gpu_fft_in, d_input + buf_size, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_fft_in, (float*)(csrc->intermediate + complex_buf_size), buf_size)) {
		printf("ERROR: Inaccurate Scaled FFT\n");
	}
	else {
		printf("Scaled FFT successful\n");
	}
	src->interpolationCalculations(5, 15, hrtf_indices, omegas);
	src->interpolationCalculations(-5, 10, old_hrtf_indices, old_omegas);
	for (int i = 0; i < 4; i++) {
		checkCudaErrors(cudaMemcpy(gpu_fft_in, d_fft_hrtf + hrtf_indices[i] * complex_buf_size * HRTF_CHN, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
		if (precisionChecking(gpu_fft_in, (float*)(fft_hrtf + hrtf_indices[i] * HRTF_CHN * complex_buf_size), buf_size, 1e-6)) {
			printf("ERROR: Inaccurate HRTF\n");
		}
		else {
			printf("Accurate HRTF \n");
		}
	}
	for (int i = 0; i < 4; i++) {
		checkCudaErrors(cudaMemcpy(gpu_fft_in, d_fft_hrtf + old_hrtf_indices[i] * complex_buf_size * HRTF_CHN, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
		if (precisionChecking(gpu_fft_in, (float*)(fft_hrtf + old_hrtf_indices[i] * HRTF_CHN * complex_buf_size), buf_size, 1e-6)) {
			printf("ERROR: Inaccurate HRTF\n");
		}
		else {
			printf("Accurate HRTF \n");
		}
	}
	//GPU
	fillWithZeroesKernel(d_output, 2 * buf_size, gsrc->streams[0]);
	fillWithZeroesKernel(d_output2, 2 * buf_size, gsrc->streams[0]);
	checkCudaErrors(cudaDeviceSynchronize());
	gsrc->caseThreeConvolve(d_input, d_output, d_convbufs, d_distance_factor, gsrc->streams, old_hrtf_indices, old_omegas);
	gsrc->caseThreeConvolve(d_input, d_output2, d_convbufs2, d_distance_factor, gsrc->streams, hrtf_indices, omegas);
	//CPU
	csrc->caseThreeConvolve(csrc->intermediate, csrc->conv_bufs, old_hrtf_indices, old_omegas);
	csrc->caseThreeConvolve(csrc->intermediate + buf_size, csrc->conv_bufs + buf_size * 2, hrtf_indices, omegas);


	//OLD HRTFS
	checkCudaErrors(cudaMemcpy(gpu_conv, d_convbufs, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)csrc->conv_bufs, 2 * buf_size)) {
		printf("ERROR: Inaccurate Case 3 Convbuf 1\n");
	}
	else {
		printf("Case 3 Convolutions Convbuf 1 successful\n");
	}
	checkCudaErrors(cudaMemcpy(gpu_conv, d_convbufs + buf_size, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->conv_bufs + buf_size), 2 * buf_size)) {
		printf("ERROR: Inaccurate Case 3 Convbuf 2\n");
	}
	else {
		printf("Case 3 Convolutions Convbuf 2 successful\n");
	}
	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_conv, d_output, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)csrc->intermediate, 2 * buf_size)) {
		printf("ERROR: Inaccurate Case 3 Convolution output\n");
	}
	else {
		printf("Case 3 Convolution output successful\n");
	}

	//NEW HRTFS
	checkCudaErrors(cudaMemcpy(gpu_conv, d_convbufs2, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->conv_bufs + buf_size * 2), buf_size)) {
		printf("ERROR: Inaccurate Case 3 Convbuf 1\n");
	}
	else {
		printf("Case 3 Convolutions Convbuf 1 successful\n");
	}
	checkCudaErrors(cudaMemcpy(gpu_conv, d_convbufs2 + buf_size, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->conv_bufs + 3 * buf_size), 2 * buf_size)) {
		printf("ERROR: Inaccurate Case 3 Convbuf 2\n");
	}
	else {
		printf("Case 3 Convolutions Convbuf 2 successful\n");
	}
	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_conv, d_output2, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->intermediate + buf_size), 2 * buf_size)) {
		printf("ERROR: Inaccurate Case 3 Convolution output\n");
	}
	else {
		printf("Case 3 Convolution output successful\n");
	}

	// Precision Check
	checkCudaErrors(cudaMemcpy(gpu_conv, d_output2, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->intermediate + buf_size), buf_size)) {
		printf("ERROR: Inaccurate Case 3 Convolutions\n");
	}
	else {
		printf("Case 3 Convolutions successful\n");
	}

	checkCudaErrors(cudaMemcpy(gpu_conv, d_output2, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->intermediate + buf_size), buf_size)) {
		printf("ERROR: Inaccurate Case 3 Convolutions\n");
	}
	else {
		printf("Case 3 Convolutions successful\n");
	}
	/////////////////////////////////////////////////////////
	// FFT OUT
	/////////////////////////////////////////////////////////
	CHECK_CUFFT_ERRORS(cufftExecC2R(gsrc->plans[1], (cufftComplex*)d_output, d_output));
	CHECK_CUFFT_ERRORS(cufftExecC2R(gsrc->plans[2], (cufftComplex*)d_output2, d_output2));

	fftwf_execute(csrc->out_plan);
	fftwf_execute_dft_c2r(
		csrc->out_plan,
		csrc->intermediate + (PAD_LEN + 2),
		(float*)(csrc->intermediate + PAD_LEN + 2)
	);
	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_ifft, d_output, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_ifft, (float*)csrc->intermediate, buf_size)) {
		printf("ERROR: Inaccurate Case 3 IFFT\n");
	}
	else {
		printf("Case 3 IFFT successful\n");
	}
	checkCudaErrors(cudaMemcpy(gpu_ifft, d_output2, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_ifft, (float*)(csrc->intermediate + buf_size), buf_size)) {
		printf("ERROR: Inaccurate Case 3 IFFT\n");
	}
	else {
		printf("Case 3 IFFT successful\n");
	}

	/////////////////////////////////////////////////////////
	// CROSSFADE
	/////////////////////////////////////////////////////////
	numThreads = FRAMES_PER_BUFFER;
	crossFade << <numThreads, 1 >> > (
		d_output + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
		d_output2 + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
		FRAMES_PER_BUFFER);

	out1 = ((float*)csrc->intermediate) + 2 * (PAD_LEN - FRAMES_PER_BUFFER);
	out2 = ((float*)csrc->intermediate) + 2 * PAD_LEN + 4 + 2 * (PAD_LEN - FRAMES_PER_BUFFER);//PAD_LEN + 2 + 2 * (PAD_LEN - FRAMES_PER_BUFFER);
#pragma omp parallel for
	for (int i = 0; i < FRAMES_PER_BUFFER; i++) {
		float fn = float(i) / (FRAMES_PER_BUFFER - 1.0f);
		out1[i * 2] = out1[i * 2] * fn + out2[i * 2] * (1.0f - fn);
		out1[i * 2 + 1] = out1[i * 2 + 1] * fn + out2[i * 2 + 1] * (1.0f - fn);
	}

	checkCudaErrors(cudaMemcpy(gpu_ifft, d_output, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_ifft, (float*)(csrc->intermediate), buf_size)) {
		printf("ERROR: Inaccurate Crossfade\n");
	}
	else {
		printf("Case 3 Crossfade successful\n");
	}

	/*Receive*/
	gsrc->receiveBlock(0);
	memcpy(gpu_output, gsrc->intermediate, FRAMES_PER_BUFFER * 2 * sizeof(float));
	memcpy(cpu_output, ((float*)csrc->intermediate) + 2 * (PAD_LEN - FRAMES_PER_BUFFER), FRAMES_PER_BUFFER * 2 * sizeof(float));
	if (precisionChecking(gpu_output, cpu_output, FRAMES_PER_BUFFER * 2)) {
		printf("ERROR: Inaccurate Case 3 output\n");
	}
	else {
		printf("Successful Case 3 accurate output\n");
	}

	/////////////////////////////////////////////////////////
	// CASE 4 CONVOLUTIONS - Worst Case Scenario
	/////////////////////////////////////////////////////////
	fftwf_execute(csrc->in_plan); /*FFT on x --> intermediate*/
	complexScaling(csrc->intermediate, scale, complex_buf_size);
	memcpy(
		csrc->intermediate + complex_buf_size,
		csrc->intermediate,
		complex_buf_size * sizeof(fftwf_complex)
	);
	/*Copying over for xfade*/
	memcpy(
		csrc->intermediate + buf_size,
		csrc->intermediate,
		buf_size * sizeof(fftwf_complex)
	);
	src->interpolationCalculations(3, 23, hrtf_indices, omegas);
	src->interpolationCalculations(8, 18, old_hrtf_indices, old_omegas);
	for (int i = 0; i < 4; i++) {
		checkCudaErrors(cudaMemcpy(gpu_fft_in, d_fft_hrtf + hrtf_indices[i] * complex_buf_size * HRTF_CHN, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
		if (precisionChecking(gpu_fft_in, (float*)(fft_hrtf + hrtf_indices[i] * HRTF_CHN * complex_buf_size), buf_size, 1e-6)) {
			printf("ERROR: Inaccurate HRTF\n");
		}
		else {
			printf("Accurate HRTF \n");
		}
	}
	for (int i = 0; i < 4; i++) {
		checkCudaErrors(cudaMemcpy(gpu_fft_in, d_fft_hrtf + old_hrtf_indices[i] * complex_buf_size * HRTF_CHN, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
		if (precisionChecking(gpu_fft_in, (float*)(fft_hrtf + old_hrtf_indices[i] * HRTF_CHN * complex_buf_size), buf_size, 1e-6)) {
			printf("ERROR: Inaccurate HRTF\n");
		}
		else {
			printf("Accurate HRTF \n");
		}
	}

	//GPU
	fillWithZeroesKernel(d_output, 2 * buf_size, gsrc->streams[0]);
	fillWithZeroesKernel(d_output2, 2 * buf_size, gsrc->streams[0]);
	checkCudaErrors(cudaDeviceSynchronize());
	gsrc->caseFourConvolve(d_input, d_output, d_convbufs, d_distance_factor, gsrc->streams, old_hrtf_indices, old_omegas);
	gsrc->caseFourConvolve(d_input, d_output2, d_convbufs + buf_size * 4, d_distance_factor, gsrc->streams, hrtf_indices, omegas);
	//CPU
	csrc->caseFourConvolve(csrc->intermediate, csrc->conv_bufs, old_hrtf_indices, old_omegas);
	csrc->caseFourConvolve(csrc->intermediate + buf_size, csrc->conv_bufs + buf_size * 4, hrtf_indices, omegas);


	for (int i = 0; i < 8; i++) {
		checkCudaErrors(cudaMemcpy(gpu_conv, d_convbufs + i * buf_size, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));
		if (precisionChecking(gpu_conv, (float*)(csrc->conv_bufs + i * buf_size), 2 * buf_size)) {
			printf("ERROR: Inaccurate Case 4 Convbuf %i\n", i);
		}
		else {
			printf("Case 4 Convolutions Convbuf %i successful\n", i);
		}
	}
	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_conv, d_output2, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->intermediate + buf_size), 2 * buf_size)) {
		printf("ERROR: Inaccurate Case 4 Convolution output\n");
	}
	else {
		printf("Case 4 Convolution output successful\n");
	}

	// Precision Check
	checkCudaErrors(cudaMemcpy(gpu_conv, d_output2, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->intermediate + buf_size), buf_size)) {
		printf("ERROR: Inaccurate Case 4 Convolutions\n");
	}
	else {
		printf("Case 4 Convolutions successful\n");
	}

	checkCudaErrors(cudaMemcpy(gpu_conv, d_output2, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->intermediate + buf_size), buf_size)) {
		printf("ERROR: Inaccurate Case 4 Convolutions\n");
	}
	else {
		printf("Case 4 Convolutions successful\n");
	}


	/////////////////////////////////////////////////////////
	// FFT OUT
	/////////////////////////////////////////////////////////
	CHECK_CUFFT_ERRORS(cufftExecC2R(gsrc->plans[1], (cufftComplex*)d_output, d_output));
	fftwf_execute(csrc->out_plan);
	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_ifft, d_output, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_ifft, (float*)csrc->intermediate, buf_size)) {
		printf("ERROR: Inaccurate Case 4 IFFT\n");
	}
	else {
		printf("Case 4 IFFT successful\n");
	}
	/////////////////////////////////////////////////////////
	// CROSSFADE
	/////////////////////////////////////////////////////////
	numThreads = FRAMES_PER_BUFFER;
	crossFade << <numThreads, 1 >> > (
		d_output + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
		d_output2 + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
		FRAMES_PER_BUFFER);

	out1 = ((float*)csrc->intermediate) + 2 * (PAD_LEN - FRAMES_PER_BUFFER);
	out2 = ((float*)csrc->intermediate) + 2 * PAD_LEN + 4 + 2 * (PAD_LEN - FRAMES_PER_BUFFER);//PAD_LEN + 2 + 2 * (PAD_LEN - FRAMES_PER_BUFFER);
#pragma omp parallel for
	for (int i = 0; i < FRAMES_PER_BUFFER; i++) {
		float fn = float(i) / (FRAMES_PER_BUFFER - 1.0f);
		out1[i * 2] = out1[i * 2] * fn + out2[i * 2] * (1.0f - fn);
		out1[i * 2 + 1] = out1[i * 2 + 1] * fn + out2[i * 2 + 1] * (1.0f - fn);
	}

	checkCudaErrors(cudaMemcpy(gpu_ifft, d_output, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_ifft, (float*)(csrc->intermediate), buf_size)) {
		printf("ERROR: Inaccurate Crossfade\n");
	}
	else {
		printf("Case 4 Crossfade successful\n");
	}
	/*Receive*/
	gsrc->receiveBlock(0);
	memcpy(gpu_output, gsrc->intermediate, FRAMES_PER_BUFFER * 2 * sizeof(float));
	memcpy(cpu_output, ((float*)csrc->intermediate) + 2 * (PAD_LEN - FRAMES_PER_BUFFER), FRAMES_PER_BUFFER * 2 * sizeof(float));
	if (precisionChecking(gpu_output, cpu_output, FRAMES_PER_BUFFER * 2)) {
		printf("ERROR: Inaccurate Case 4 output\n");
	}
	else {
		printf("Successful Case 4 accurate output\n");
	}
	for (int i = 0; i < PAD_LEN + 2; i++) {
		gsrc->x[i] = 0.0f;

		csrc->x[i] = 0.0f;
	}
	for (int i = 0; i < FRAMES_PER_BUFFER * 2; i++) {
		gsrc->intermediate[i] = 0.0f;
	}
	p->blockNo = 0;
	src->count = 0;
	delete[] gpu_output;
	delete[] cpu_output;
	delete[] gpu_conv;
	delete[] gpu_distance;
	delete[] gpu_ifft;
	delete[] gpu_fft_in;
	delete[] gpu_fft_scaled;
}



void xfadePrecisionCallbackTest(Data* p) {
	SoundSource* src = (SoundSource*)&(p->all_sources[0]);
	GPUSoundSource* gsrc = &(p->all_sources[0]);
	CPUSoundSource* csrc = (CPUSoundSource*)&(p->all_sources[0]);
	float* gpu_output = new float[FRAMES_PER_BUFFER * 2];
	float* cpu_output = new float[FRAMES_PER_BUFFER * 2];

	float scale = 1.0f / ((float)PAD_LEN);
	size_t buf_size = PAD_LEN + 2;
	size_t complex_buf_size = buf_size / 2;

	float* gpu_distance = new float[buf_size];
	float* gpu_fft_in = new float[buf_size];
	float* gpu_ifft = new float[2 * buf_size];
	float* gpu_fft_scaled = new float[buf_size];
	float* gpu_conv = new float[2 * buf_size];

	/////////////////////////////////////////////////////////
	// COPY INCOMING BLOCK
	/////////////////////////////////////////////////////////

	// GPU
	p->blockNo = 0;
	src->count = 0;
	for (int i = 0; i < PAD_LEN + 2; i++) {
		gsrc->x[i] = 0.0f;
	}
	fillWithZeroesKernel(gsrc->d_output, 2 * buf_size, gsrc->streams[0]);
	gsrc->copyIncomingBlock(0);
	gsrc->sendBlock(0);
	// CPU
	src->count = 0;
	memcpy(
		csrc->x + (PAD_LEN - FRAMES_PER_BUFFER),  /*Go to the end and work backwards*/
		csrc->buf,
		FRAMES_PER_BUFFER * sizeof(float));


	checkCudaErrors(cudaMemcpy(gpu_fft_in, gsrc->d_input, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_fft_in, (float*)csrc->x, PAD_LEN)) {
		printf("ERROR: Inaccurate Input Copy\n");
	}
	else {
		printf("Accurate Input Copy\n");
	}

	/////////////////////////////////////////////////////////
	// CALCULATE WEIGHTS
	/////////////////////////////////////////////////////////
	int blockNo = 0;
	src->old_azi = 18;
	src->old_ele = 8;
	src->ele = 23;
	src->azi = 18;
	float ele = src->ele;
	float azi = src->azi;
	int hrtf_indices[4];
	float omegas[6];
	int old_hrtf_indices[4];
	float old_omegas[6];

	src->interpolationCalculations(3, 23, hrtf_indices, omegas);
	src->interpolationCalculations(8, 18, old_hrtf_indices, old_omegas);


	/////////////////////////////////////////////////////////
	// CALCULATE DISTANCE FACTOR
	/////////////////////////////////////////////////////////

	// GPU
	gsrc->gpuCalculateDistanceFactor(blockNo);

	// CPU
	csrc->calculateDistanceFactor();

	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_distance, gsrc->d_distance_factor, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_distance, (float*)csrc->distance_factor, PAD_LEN + 2)) {
		printf("ERROR: Inaccurate Distance calculations\n");
	}
	else {
		printf("Distance calculations successfull\n");
	}



	/////////////////////////////////////////////////////////
	// FFT IN
	/////////////////////////////////////////////////////////

	// GPU
	cufftComplex* d_distance_factor = gsrc->d_distance_factor;
	float* d_input = gsrc->d_input;
	float* d_output = gsrc->d_output;
	float* d_output2 = gsrc->d_output2;
	cufftComplex* d_convbufs = gsrc->d_convbufs;
	cufftComplex* d_convbufs2 = gsrc->d_convbufs + 4 * (PAD_LEN + 2);
	CHECK_CUFFT_ERRORS(cufftExecR2C(gsrc->plans[0], (cufftReal*)d_input, (cufftComplex*)d_input));


	// CPU
	fftwf_execute(csrc->in_plan); /*FFT on x --> intermediate*/

	// Precision Check
	checkCudaErrors(cudaMemcpy(gpu_fft_in, d_input, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_fft_in, (float*)csrc->intermediate, PAD_LEN + 2)) {
		printf("ERROR: Inaccurate FFT\n");
	}
	else {
		printf("FFT successful\n");
	}


	/////////////////////////////////////////////////////////
	// SCALING
	/////////////////////////////////////////////////////////

	// GPU
	int numThreads = 128;
	int numBlocks = (buf_size + numThreads - 1) / numThreads;

	MyFloatScale << < numBlocks, numThreads >> > (d_input, scale, buf_size);
	// CPU
	complexScaling(csrc->intermediate, scale, complex_buf_size);

	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_fft_scaled, d_input, buf_size * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_fft_scaled, (float*)csrc->intermediate, buf_size)) {
		printf("ERROR: Inaccurate Scaling\n");
	}
	else {
		printf("Scaling successful\n");
	}

	/*Copying over for both channels*/
	checkCudaErrors(cudaMemcpy(d_input + buf_size, d_input, buf_size * sizeof(float), cudaMemcpyDeviceToDevice));
	memcpy(
		csrc->intermediate + complex_buf_size,
		csrc->intermediate,
		complex_buf_size * sizeof(fftwf_complex)
	);
	/*Copying over for xfade*/
	memcpy(
		csrc->intermediate + buf_size,
		csrc->intermediate,
		(PAD_LEN + 2) * sizeof(fftwf_complex)
	);


	/////////////////////////////////////////////////////////
	// CASE 4 CONVOLUTIONS - Worst Case Scenario
	/////////////////////////////////////////////////////////
	for (int i = 0; i < 4; i++) {
		checkCudaErrors(cudaMemcpy(gpu_fft_in, d_fft_hrtf + hrtf_indices[i] * complex_buf_size * HRTF_CHN, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
		if (precisionChecking(gpu_fft_in, (float*)(fft_hrtf + hrtf_indices[i] * HRTF_CHN * complex_buf_size), buf_size, 1e-6)) {
			printf("ERROR: Inaccurate HRTF\n");
		}
		else {
			printf("Accurate HRTF \n");
		}
	}
	for (int i = 0; i < 4; i++) {
		checkCudaErrors(cudaMemcpy(gpu_fft_in, d_fft_hrtf + old_hrtf_indices[i] * complex_buf_size * HRTF_CHN, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
		if (precisionChecking(gpu_fft_in, (float*)(fft_hrtf + old_hrtf_indices[i] * HRTF_CHN * complex_buf_size), buf_size, 1e-6)) {
			printf("ERROR: Inaccurate HRTF\n");
		}
		else {
			printf("Accurate HRTF \n");
		}
	}

	//GPU
	fillWithZeroesKernel(d_output, 2 * buf_size, gsrc->streams[0]);
	fillWithZeroesKernel(d_output2, 2 * buf_size, gsrc->streams[0]);
	checkCudaErrors(cudaDeviceSynchronize());
	gsrc->caseFourConvolve(d_input, d_output, d_convbufs, d_distance_factor, gsrc->streams, old_hrtf_indices, old_omegas);
	gsrc->caseFourConvolve(d_input, d_output2, d_convbufs + buf_size * 4, d_distance_factor, gsrc->streams, hrtf_indices, omegas);
	//CPU
	csrc->caseFourConvolve(csrc->intermediate, csrc->conv_bufs, old_hrtf_indices, old_omegas);
	csrc->caseFourConvolve(csrc->intermediate + buf_size, csrc->conv_bufs + buf_size * 4, hrtf_indices, omegas);


	for (int i = 0; i < 8; i++) {
		checkCudaErrors(cudaMemcpy(gpu_conv, d_convbufs + i * buf_size, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));
		if (precisionChecking(gpu_conv, (float*)(csrc->conv_bufs + i * buf_size), 2 * buf_size)) {
			printf("ERROR: Inaccurate Case 4 Convbuf %i\n", i);
		}
		else {
			printf("Case 4 Convolutions Convbuf %i successful\n", i);
		}
	}
	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_conv, d_output2, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->intermediate + buf_size), 2 * buf_size)) {
		printf("ERROR: Inaccurate Case 4 Convolution output\n");
	}
	else {
		printf("Case 4 Convolution output successful\n");
	}

	// Precision Check
	checkCudaErrors(cudaMemcpy(gpu_conv, d_output2, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->intermediate + buf_size), buf_size)) {
		printf("ERROR: Inaccurate Case 4 Convolutions\n");
	}
	else {
		printf("Case 4 Convolutions successful\n");
	}

	checkCudaErrors(cudaMemcpy(gpu_conv, d_output2, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->intermediate + buf_size), buf_size)) {
		printf("ERROR: Inaccurate Case 4 Convolutions\n");
	}
	else {
		printf("Case 4 Convolutions successful\n");
	}


	/////////////////////////////////////////////////////////
	// FFT OUT
	/////////////////////////////////////////////////////////
	CHECK_CUFFT_ERRORS(cufftExecC2R(gsrc->plans[1], (cufftComplex*)d_output, d_output));
	fftwf_execute(csrc->out_plan);
	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_ifft, d_output, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_ifft, (float*)csrc->intermediate, buf_size)) {
		printf("ERROR: Inaccurate Case 4 IFFT\n");
	}
	else {
		printf("Case 4 IFFT successful\n");
	}
	/////////////////////////////////////////////////////////
	// CROSSFADE
	/////////////////////////////////////////////////////////
	numThreads = FRAMES_PER_BUFFER;
	crossFade << <numThreads, 1 >> > (
		d_output + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
		d_output2 + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
		FRAMES_PER_BUFFER);

	float* out1 = ((float*)csrc->intermediate) + 2 * (PAD_LEN - FRAMES_PER_BUFFER);
	float* out2 = ((float*)csrc->intermediate) + 2 * PAD_LEN + 4 + 2 * (PAD_LEN - FRAMES_PER_BUFFER);//PAD_LEN + 2 + 2 * (PAD_LEN - FRAMES_PER_BUFFER);
#pragma omp parallel for
	for (int i = 0; i < FRAMES_PER_BUFFER; i++) {
		float fn = float(i) / (FRAMES_PER_BUFFER - 1.0f);
		out1[i * 2] = out1[i * 2] * fn + out2[i * 2] * (1.0f - fn);
		out1[i * 2 + 1] = out1[i * 2 + 1] * fn + out2[i * 2 + 1] * (1.0f - fn);
	}

	checkCudaErrors(cudaMemcpy(gpu_ifft, d_output, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_ifft, (float*)(csrc->intermediate), buf_size)) {
		printf("ERROR: Inaccurate Crossfade\n");
	}
	else {
		printf("Case 4 Crossfade successful\n");
	}
	/*Receive*/
	gsrc->receiveBlock(0);
	memcpy(gpu_output, gsrc->intermediate, FRAMES_PER_BUFFER * 2 * sizeof(float));
	memcpy(cpu_output, ((float*)csrc->intermediate) + 2 * (PAD_LEN - FRAMES_PER_BUFFER), FRAMES_PER_BUFFER * 2 * sizeof(float));
	if (precisionChecking(gpu_output, cpu_output, FRAMES_PER_BUFFER * 2)) {
		printf("ERROR: Inaccurate Case 4 output\n");
	}
	else {
		printf("Successful Case 4 accurate output\n");
	}


	// ROUND TWO!
	/*Overlap-save*/
	printf("Round 2\n");
	memmove(
		csrc->x,
		csrc->x + FRAMES_PER_BUFFER,
		sizeof(float) * (PAD_LEN - FRAMES_PER_BUFFER)
	);

	p->blockNo++;
	blockNo = p->blockNo;
	src->count = 128;
	gsrc->overlapSave(blockNo);
	gsrc->copyIncomingBlock(blockNo);
	gsrc->sendBlock(blockNo);
	// CPU
	src->count = 128;
	memcpy(
		csrc->x + (PAD_LEN - FRAMES_PER_BUFFER),  /*Go to the end and work backwards*/
		csrc->buf + src->count,
		FRAMES_PER_BUFFER * sizeof(float));


	checkCudaErrors(cudaMemcpy(gpu_fft_in, gsrc->d_input, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_fft_in, (float*)csrc->x, PAD_LEN)) {
		printf("ERROR: Inaccurate Input Copy\n");
	}
	else {
		printf("Accurate Input Copy\n");
	}
	/////////////////////////////////////////////////////////
	// CALCULATE DISTANCE FACTOR
	/////////////////////////////////////////////////////////

	// GPU
	gsrc->gpuCalculateDistanceFactor(blockNo);

	// CPU
	csrc->calculateDistanceFactor();

	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_distance, gsrc->d_distance_factor, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_distance, (float*)csrc->distance_factor, PAD_LEN + 2)) {
		printf("ERROR: Inaccurate Distance calculations\n");
	}
	else {
		printf("Distance calculations successfull\n");
	}



	/////////////////////////////////////////////////////////
	// FFT IN
	/////////////////////////////////////////////////////////

	// GPU
	d_distance_factor = gsrc->d_distance_factor;
	d_input = gsrc->d_input;
	d_output = gsrc->d_output;
	d_output2 = gsrc->d_output2;
	d_convbufs = gsrc->d_convbufs;
	d_convbufs2 = gsrc->d_convbufs + 4 * (PAD_LEN + 2);
	CHECK_CUFFT_ERRORS(cufftExecR2C(gsrc->plans[0], (cufftReal*)d_input, (cufftComplex*)d_input));


	// CPU
	fftwf_execute(csrc->in_plan); /*FFT on x --> intermediate*/

	// Precision Check
	checkCudaErrors(cudaMemcpy(gpu_fft_in, d_input, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_fft_in, (float*)csrc->intermediate, PAD_LEN + 2, 1e-6)) {
		printf("ERROR: Inaccurate FFT\n");
	}
	else {
		printf("FFT successful\n");
	}


	/////////////////////////////////////////////////////////
	// SCALING
	/////////////////////////////////////////////////////////

	// GPU
	numThreads = 128;
	numBlocks = (buf_size + numThreads - 1) / numThreads;

	MyFloatScale << < numBlocks, numThreads >> > (d_input, scale, buf_size);
	// CPU
	complexScaling(csrc->intermediate, scale, complex_buf_size);

	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_fft_scaled, d_input, buf_size * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_fft_scaled, (float*)csrc->intermediate, buf_size)) {
		printf("ERROR: Inaccurate Scaling\n");
	}
	else {
		printf("Scaling successful\n");
	}

	/*Copying over for both channels*/
	checkCudaErrors(cudaMemcpy(d_input + buf_size, d_input, buf_size * sizeof(float), cudaMemcpyDeviceToDevice));
	memcpy(
		csrc->intermediate + complex_buf_size,
		csrc->intermediate,
		complex_buf_size * sizeof(fftwf_complex)
	);
	/*Copying over for xfade*/
	memcpy(
		csrc->intermediate + buf_size,
		csrc->intermediate,
		(PAD_LEN + 2) * sizeof(fftwf_complex)
	);


	/////////////////////////////////////////////////////////
	// CASE 4 CONVOLUTIONS - Worst Case Scenario
	/////////////////////////////////////////////////////////
	for (int i = 0; i < 4; i++) {
		checkCudaErrors(cudaMemcpy(gpu_fft_in, d_fft_hrtf + hrtf_indices[i] * complex_buf_size * HRTF_CHN, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
		if (precisionChecking(gpu_fft_in, (float*)(fft_hrtf + hrtf_indices[i] * HRTF_CHN * complex_buf_size), buf_size, 1e-6)) {
			printf("ERROR: Inaccurate HRTF\n");
		}
		else {
			printf("Accurate HRTF \n");
		}
	}
	for (int i = 0; i < 4; i++) {
		checkCudaErrors(cudaMemcpy(gpu_fft_in, d_fft_hrtf + old_hrtf_indices[i] * complex_buf_size * HRTF_CHN, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
		if (precisionChecking(gpu_fft_in, (float*)(fft_hrtf + old_hrtf_indices[i] * HRTF_CHN * complex_buf_size), buf_size, 1e-6)) {
			printf("ERROR: Inaccurate HRTF\n");
		}
		else {
			printf("Accurate HRTF \n");
		}
	}

	//GPU
	fillWithZeroesKernel(d_output, 2 * buf_size, gsrc->streams[0]);
	checkCudaErrors(cudaDeviceSynchronize());
	fillWithZeroesKernel(d_output2, 2 * buf_size, gsrc->streams[0]);
	checkCudaErrors(cudaDeviceSynchronize());
	gsrc->caseFourConvolve(d_input, d_output, d_convbufs, d_distance_factor, gsrc->streams, old_hrtf_indices, old_omegas);
	gsrc->caseFourConvolve(d_input, d_output2, d_convbufs + buf_size * 4, d_distance_factor, gsrc->streams, hrtf_indices, omegas);
	//CPU
	csrc->caseFourConvolve(csrc->intermediate, csrc->conv_bufs, old_hrtf_indices, old_omegas);
	csrc->caseFourConvolve(csrc->intermediate + buf_size, csrc->conv_bufs + buf_size * 4, hrtf_indices, omegas);


	for (int i = 0; i < 8; i++) {
		checkCudaErrors(cudaMemcpy(gpu_conv, d_convbufs + i * buf_size, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));
		if (precisionChecking(gpu_conv, (float*)(csrc->conv_bufs + i * buf_size), 2 * buf_size)) {
			printf("ERROR: Inaccurate Case 4 Convbuf %i\n", i);
		}
		else {
			printf("Case 4 Convolutions Convbuf %i successful\n", i);
		}
	}
	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_conv, d_output2, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->intermediate + buf_size), 2 * buf_size)) {
		printf("ERROR: Inaccurate Case 4 Convolution output\n");
	}
	else {
		printf("Case 4 Convolution output successful\n");
	}

	// Precision Check
	checkCudaErrors(cudaMemcpy(gpu_conv, d_output2, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->intermediate + buf_size), buf_size)) {
		printf("ERROR: Inaccurate Case 4 Convolutions\n");
	}
	else {
		printf("Case 4 Convolutions successful\n");
	}

	checkCudaErrors(cudaMemcpy(gpu_conv, d_output2, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->intermediate + buf_size), buf_size)) {
		printf("ERROR: Inaccurate Case 4 Convolutions\n");
	}
	else {
		printf("Case 4 Convolutions successful\n");
	}


	/////////////////////////////////////////////////////////
	// FFT OUT
	/////////////////////////////////////////////////////////
	CHECK_CUFFT_ERRORS(cufftExecC2R(gsrc->plans[1], (cufftComplex*)d_output, d_output));
	fftwf_execute(csrc->out_plan);
	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_ifft, d_output, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_ifft, (float*)csrc->intermediate, buf_size)) {
		printf("ERROR: Inaccurate Case 4 IFFT\n");
	}
	else {
		printf("Case 4 IFFT successful\n");
	}
	/////////////////////////////////////////////////////////
	// CROSSFADE
	/////////////////////////////////////////////////////////
	numThreads = FRAMES_PER_BUFFER;
	crossFade << <numThreads, 1 >> > (
		d_output + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
		d_output2 + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
		FRAMES_PER_BUFFER);

	out1 = ((float*)csrc->intermediate) + 2 * (PAD_LEN - FRAMES_PER_BUFFER);
	out2 = ((float*)csrc->intermediate) + 2 * PAD_LEN + 4 + 2 * (PAD_LEN - FRAMES_PER_BUFFER);//PAD_LEN + 2 + 2 * (PAD_LEN - FRAMES_PER_BUFFER);
#pragma omp parallel for
	for (int i = 0; i < FRAMES_PER_BUFFER; i++) {
		float fn = float(i) / (FRAMES_PER_BUFFER - 1.0f);
		out1[i * 2] = out1[i * 2] * fn + out2[i * 2] * (1.0f - fn);
		out1[i * 2 + 1] = out1[i * 2 + 1] * fn + out2[i * 2 + 1] * (1.0f - fn);
	}

	checkCudaErrors(cudaMemcpy(gpu_ifft, d_output, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_ifft, (float*)(csrc->intermediate), buf_size)) {
		printf("ERROR: Inaccurate Crossfade\n");
	}
	else {
		printf("Case 4 Crossfade successful\n");
	}
	/*Receive*/
	gsrc->receiveBlock(blockNo);
	memcpy(gpu_output, gsrc->intermediate, FRAMES_PER_BUFFER * 2 * sizeof(float));
	memcpy(cpu_output, ((float*)csrc->intermediate) + 2 * (PAD_LEN - FRAMES_PER_BUFFER), FRAMES_PER_BUFFER * 2 * sizeof(float));
	if (precisionChecking(gpu_output, cpu_output, FRAMES_PER_BUFFER * 2)) {
		printf("ERROR: Inaccurate Case 4 output\n");
	}
	else {
		printf("Successful Case 4 accurate output\n");
	}

	////////////////////////////////
	//ROUND THREE!
	printf("Round 3\n");
	/*Overlap-save*/
	p->blockNo++;
	blockNo = p->blockNo;
	memmove(
		csrc->x,
		csrc->x + FRAMES_PER_BUFFER,
		sizeof(float) * (PAD_LEN - FRAMES_PER_BUFFER)
	);

	src->count = 256;
	gsrc->overlapSave(blockNo);
	gsrc->copyIncomingBlock(blockNo % FLIGHT_NUM);
	gsrc->sendBlock(blockNo % FLIGHT_NUM);
	// CPU
	src->count = 256;
	memcpy(
		csrc->x + (PAD_LEN - FRAMES_PER_BUFFER),  /*Go to the end and work backwards*/
		csrc->buf + src->count,
		FRAMES_PER_BUFFER * sizeof(float));

	checkCudaErrors(cudaMemcpy(gpu_fft_in, d_input, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_fft_in, (float*)csrc->x, PAD_LEN)) {
		printf("ERROR: Inaccurate Input Copy\n");
	}
	else {
		printf("Accurate Input Copy\n");
	}

	/////////////////////////////////////////////////////////
	// CALCULATE DISTANCE FACTOR
	/////////////////////////////////////////////////////////
	// GPU
	gsrc->gpuCalculateDistanceFactor(blockNo % FLIGHT_NUM);
	csrc->calculateDistanceFactor();

	// Precision Check
	//checkCudaErrors(cudaMemcpy(gpu_distance, d_distance_factor, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	/*if (precisionChecking(gpu_distance, (float*)csrc->distance_factor, PAD_LEN + 2)) {
		printf("ERROR: Inaccurate Distance calculations\n");
	}
	else {
		printf("Distance calculations successfull\n");
	}*/

	/////////////////////////////////////////////////////////
	// FFT IN
	/////////////////////////////////////////////////////////

	// GPU
	CHECK_CUFFT_ERRORS(cufftExecR2C(gsrc->plans[(blockNo % FLIGHT_NUM) * 3], d_input, (cufftComplex*)d_input));

	// CPU
	fftwf_execute(csrc->in_plan); /*FFT on x --> intermediate*/

	// Precision Check
	checkCudaErrors(cudaMemcpy(gpu_fft_in, d_input, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_fft_in, (float*)csrc->intermediate, PAD_LEN + 2, 1e-6)) {
		printf("ERROR: Inaccurate FFT\n");
	}
	else {
		printf("FFT successful\n");
	}


	/////////////////////////////////////////////////////////
	// SCALING
	/////////////////////////////////////////////////////////

	// GPU
	numThreads = 128;
	numBlocks = (buf_size + numThreads - 1) / numThreads;

	MyFloatScale << < numBlocks, numThreads >> > (d_input, scale, buf_size);
	// CPU
	complexScaling(csrc->intermediate, scale, complex_buf_size);

	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_fft_scaled, d_input, buf_size * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_fft_scaled, (float*)csrc->intermediate, buf_size)) {
		printf("ERROR: Inaccurate Scaling\n");
	}
	else {
		printf("Scaling successful\n");
	}

	/*Copying over for both channels*/
	checkCudaErrors(cudaMemcpy(d_input + buf_size, d_input, buf_size * sizeof(float), cudaMemcpyDeviceToDevice));
	memcpy(
		csrc->intermediate + complex_buf_size,
		csrc->intermediate,
		complex_buf_size * sizeof(fftwf_complex)
	);
	/*Copying over for xfade*/
	memcpy(
		csrc->intermediate + buf_size,
		csrc->intermediate,
		(PAD_LEN + 2) * sizeof(fftwf_complex)
	);


	/////////////////////////////////////////////////////////
	// CASE 4 CONVOLUTIONS - Worst Case Scenario
	/////////////////////////////////////////////////////////
	for (int i = 0; i < 4; i++) {
		checkCudaErrors(cudaMemcpy(gpu_fft_in, d_fft_hrtf + hrtf_indices[i] * complex_buf_size * HRTF_CHN, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
		if (precisionChecking(gpu_fft_in, (float*)(fft_hrtf + hrtf_indices[i] * HRTF_CHN * complex_buf_size), buf_size, 1e-6)) {
			printf("ERROR: Inaccurate HRTF\n");
		}
		else {
			printf("Accurate HRTF \n");
		}
	}
	for (int i = 0; i < 4; i++) {
		checkCudaErrors(cudaMemcpy(gpu_fft_in, d_fft_hrtf + old_hrtf_indices[i] * complex_buf_size * HRTF_CHN, buf_size * sizeof(float), cudaMemcpyDeviceToHost));
		if (precisionChecking(gpu_fft_in, (float*)(fft_hrtf + old_hrtf_indices[i] * HRTF_CHN * complex_buf_size), buf_size, 1e-6)) {
			printf("ERROR: Inaccurate HRTF\n");
		}
		else {
			printf("Accurate HRTF \n");
		}
	}

	//GPU
	fillWithZeroesKernel(d_output, 2 * buf_size, gsrc->streams[0]);
	fillWithZeroesKernel(d_output2, 2 * buf_size, gsrc->streams[0]);
	checkCudaErrors(cudaDeviceSynchronize());
	gsrc->caseFourConvolve(d_input, d_output, d_convbufs, d_distance_factor, gsrc->streams, old_hrtf_indices, old_omegas);
	gsrc->caseFourConvolve(d_input, d_output2, d_convbufs + buf_size * 4, d_distance_factor, gsrc->streams, hrtf_indices, omegas);
	//CPU
	csrc->caseFourConvolve(csrc->intermediate, csrc->conv_bufs, old_hrtf_indices, old_omegas);
	csrc->caseFourConvolve(csrc->intermediate + buf_size, csrc->conv_bufs + buf_size * 4, hrtf_indices, omegas);


	for (int i = 0; i < 8; i++) {
		checkCudaErrors(cudaMemcpy(gpu_conv, d_convbufs + i * buf_size, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));
		if (precisionChecking(gpu_conv, (float*)(csrc->conv_bufs + i * buf_size), 2 * buf_size)) {
			printf("ERROR: Inaccurate Case 4 Convbuf %i\n", i);
		}
		else {
			printf("Case 4 Convolutions Convbuf %i successful\n", i);
		}
	}
	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_conv, d_output2, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->intermediate + buf_size), 2 * buf_size)) {
		printf("ERROR: Inaccurate Case 4 Convolution output\n");
	}
	else {
		printf("Case 4 Convolution output successful\n");
	}

	// Precision Check
	checkCudaErrors(cudaMemcpy(gpu_conv, d_output2, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->intermediate + buf_size), buf_size)) {
		printf("ERROR: Inaccurate Case 4 Convolutions\n");
	}
	else {
		printf("Case 4 Convolutions successful\n");
	}

	checkCudaErrors(cudaMemcpy(gpu_conv, d_output2, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));
	if (precisionChecking(gpu_conv, (float*)(csrc->intermediate + buf_size), buf_size)) {
		printf("ERROR: Inaccurate Case 4 Convolutions\n");
	}
	else {
		printf("Case 4 Convolutions successful\n");
	}


	/////////////////////////////////////////////////////////
	// FFT OUT
	/////////////////////////////////////////////////////////
	CHECK_CUFFT_ERRORS(cufftExecC2R(gsrc->plans[1], (cufftComplex*)d_output, d_output));
	fftwf_execute(csrc->out_plan);
	// Precision Check

	checkCudaErrors(cudaMemcpy(gpu_ifft, d_output, 2 * (buf_size) * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_ifft, (float*)csrc->intermediate, buf_size)) {
		printf("ERROR: Inaccurate Case 4 IFFT\n");
	}
	else {
		printf("Case 4 IFFT successful\n");
	}
	/////////////////////////////////////////////////////////
	// CROSSFADE
	/////////////////////////////////////////////////////////
	numThreads = FRAMES_PER_BUFFER;
	crossFade << <numThreads, 1 >> > (
		d_output + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
		d_output2 + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
		FRAMES_PER_BUFFER);

	out1 = ((float*)csrc->intermediate) + 2 * (PAD_LEN - FRAMES_PER_BUFFER);
	out2 = ((float*)csrc->intermediate) + 2 * PAD_LEN + 4 + 2 * (PAD_LEN - FRAMES_PER_BUFFER);//PAD_LEN + 2 + 2 * (PAD_LEN - FRAMES_PER_BUFFER);
#pragma omp parallel for
	for (int i = 0; i < FRAMES_PER_BUFFER; i++) {
		float fn = float(i) / (FRAMES_PER_BUFFER - 1.0f);
		out1[i * 2] = out1[i * 2] * fn + out2[i * 2] * (1.0f - fn);
		out1[i * 2 + 1] = out1[i * 2 + 1] * fn + out2[i * 2 + 1] * (1.0f - fn);
	}

	checkCudaErrors(cudaMemcpy(gpu_ifft, d_output, 2 * buf_size * sizeof(float), cudaMemcpyDeviceToHost));

	if (precisionChecking(gpu_ifft, (float*)(csrc->intermediate), buf_size)) {
		printf("ERROR: Inaccurate Crossfade\n");
	}
	else {
		printf("Case 4 Crossfade successful\n");
	}
	/*Receive*/
	gsrc->receiveBlock(blockNo % FLIGHT_NUM);
	memcpy(gpu_output, gsrc->intermediate, FRAMES_PER_BUFFER * 2 * sizeof(float));
	memcpy(cpu_output, ((float*)csrc->intermediate) + 2 * (PAD_LEN - FRAMES_PER_BUFFER), FRAMES_PER_BUFFER * 2 * sizeof(float));
	if (precisionChecking(gpu_output, cpu_output, FRAMES_PER_BUFFER * 2)) {
		printf("ERROR: Inaccurate Case 4 output\n");
	}
	else {
		printf("Successful Case 4 accurate output\n");
	}
	for (int i = 0; i < PAD_LEN + 2; i++) {
		gsrc->x[i] = 0.0f;

		csrc->x[i] = 0.0f;
	}
	for (int i = 0; i < FRAMES_PER_BUFFER * 2; i++) {
		gsrc->intermediate[i] = 0.0f;
	}
	src->old_azi = 0.0f;
	src->old_ele = 0.0f;
	p->blockNo = 0;
	src->count = 0;
	delete[] gpu_output;
	delete[] cpu_output;
	delete[] gpu_conv;
	delete[] gpu_distance;
	delete[] gpu_ifft;
	delete[] gpu_fft_in;
	delete[] gpu_fft_scaled;
}
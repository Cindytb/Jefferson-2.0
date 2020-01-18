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


	float* gpu_cb_out = new float[FRAMES_PER_BUFFER * 2];
	for (int i = 0; i < FLIGHT_NUM; i++) {
		callback_func(gpu_cb_out, p);
	}
	callback_func(gpu_cb_out, p);
	/////////////////////////////////////////////////////////
	// COPY INCOMING BLOCK
	/////////////////////////////////////////////////////////

	// GPU
	p->blockNo = 0;
	src->count = 0;
	for (int i = 0; i < PAD_LEN + 2; i++) {
		gsrc->x[0][i] = 0.0f;
	}
	fillWithZeroesKernel(gsrc->d_output[0], 2 * buf_size, gsrc->streams[0]);
	gsrc->copyIncomingBlock(0);
	gsrc->sendBlock(0);
	// CPU
	src->count = 0;
	memcpy(
		csrc->x + (PAD_LEN - FRAMES_PER_BUFFER),  /*Go to the end and work backwards*/
		csrc->buf,
		FRAMES_PER_BUFFER * sizeof(float));


	checkCudaErrors(cudaMemcpy(gpu_fft_in, gsrc->d_input[0], buf_size * sizeof(float), cudaMemcpyDeviceToHost));
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

	checkCudaErrors(cudaMemcpy(gpu_distance, gsrc->distance_factor[0], buf_size * sizeof(float), cudaMemcpyDeviceToHost));
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
	cufftComplex* d_distance_factor = gsrc->distance_factor[blockNo];
	float* d_input = gsrc->d_input[blockNo];
	float* d_output = gsrc->d_output[blockNo];
	cufftComplex* d_convbufs = gsrc->d_convbufs[blockNo];
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

	//GPU
	checkCudaErrors(cudaMemcpy(d_input + buf_size, d_input, buf_size * sizeof(float), cudaMemcpyDeviceToDevice));
	/////////////////////////////////////////////////////////
	// CASE 1 CONVOLUTIONS
	/////////////////////////////////////////////////////////
	/*+ Theta 1 Left*/
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


	//CPU
	/*Copying over for both channels*/
	memcpy(
		csrc->intermediate + complex_buf_size,
		csrc->intermediate,
		complex_buf_size * sizeof(fftwf_complex)
	);
	pointwiseMultiplication(csrc->intermediate,
		fft_hrtf + hrtf_indices[0] * HRTF_CHN * complex_buf_size,
		buf_size);
	pointwiseMultiplication(
		csrc->intermediate,
		csrc->distance_factor,
		complex_buf_size
	);
	pointwiseMultiplication(
		csrc->intermediate + complex_buf_size,
		csrc->distance_factor,
		complex_buf_size
	);

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
	CHECK_CUFFT_ERRORS(cufftExecC2R(gsrc->plans[blockNo * 3 + 1], (cufftComplex*)d_output, d_output));
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
	memcpy(gpu_output, gsrc->intermediate[0], FRAMES_PER_BUFFER * 2 * sizeof(float));
	memcpy(cpu_output, ((float*)csrc->intermediate) + 2 * (PAD_LEN - FRAMES_PER_BUFFER), FRAMES_PER_BUFFER * 2 * sizeof(float));
	if (precisionChecking(gpu_output, cpu_output, FRAMES_PER_BUFFER * 2)) {
		printf("ERROR: Inaccurate case 1 output\n");
	}
	else {
		printf("Successfully accurate case 1 output\n");
	}

	if (precisionChecking(gpu_cb_out, gpu_output, FRAMES_PER_BUFFER * 2)) {
		printf("ERROR: Callback function is giving different results\n");
	}
	else {
		printf("Successfully accurate callback function\n");
	}

	if (precisionChecking(gpu_cb_out, cpu_output, FRAMES_PER_BUFFER * 2)) {
		printf("ERROR: Callback function is giving different results\n");
	}
	else {
		printf("Successfully accurate callback function\n");
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
	
	// CPU
	pointwiseMultiplication(
		csrc->intermediate,
		fft_hrtf + hrtf_indices[0] * HRTF_CHN * complex_buf_size,
		csrc->conv_bufs,
		buf_size
	);
	pointwiseMultiplication(
		csrc->intermediate,
		fft_hrtf + hrtf_indices[1] * HRTF_CHN * complex_buf_size,
		csrc->conv_bufs + buf_size,
		buf_size
	);
	complexScaling(csrc->conv_bufs, omegas[1], buf_size);
	complexScaling(csrc->conv_bufs + buf_size, omegas[0], buf_size);
	
	for (int i = 0; i < 4; i++) {
		pointwiseMultiplication(
			csrc->conv_bufs + complex_buf_size * i,
			csrc->distance_factor,
			complex_buf_size
		);
	}

	pointwiseAddition(
		csrc->conv_bufs,
		csrc->conv_bufs + buf_size,
		csrc->intermediate,
		buf_size);
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
	CHECK_CUFFT_ERRORS(cufftExecC2R(gsrc->plans[blockNo * 3 + 1], (cufftComplex*)d_output, d_output));
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
	memcpy(gpu_output, gsrc->intermediate[0], FRAMES_PER_BUFFER * 2 * sizeof(float));
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

	// CPU
	pointwiseMultiplication(
		csrc->intermediate,
		fft_hrtf + hrtf_indices[0] * HRTF_CHN * complex_buf_size,
		csrc->conv_bufs,
		buf_size
	);
	pointwiseMultiplication(
		csrc->intermediate,
		fft_hrtf + hrtf_indices[2] * HRTF_CHN * complex_buf_size,
		csrc->conv_bufs + buf_size,
		buf_size
	);
	complexScaling(csrc->conv_bufs, omegas[5], buf_size);
	complexScaling(csrc->conv_bufs + buf_size, omegas[4], buf_size);
	for (int i = 0; i < 4; i++) {
		pointwiseMultiplication(
			csrc->conv_bufs + complex_buf_size * i,
			csrc->distance_factor,
			complex_buf_size
		);
	}
	pointwiseAddition(
		csrc->conv_bufs,
		csrc->conv_bufs + buf_size,
		csrc->intermediate,
		buf_size);
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
	CHECK_CUFFT_ERRORS(cufftExecC2R(gsrc->plans[blockNo * 3 + 1], (cufftComplex*)d_output, d_output));
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
	memcpy(gpu_output, gsrc->intermediate[0], FRAMES_PER_BUFFER * 2 * sizeof(float));
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

	// CPU
	for (int i = 0; i < 4; i++) {
		pointwiseMultiplication(
			csrc->intermediate,
			fft_hrtf + hrtf_indices[i] * HRTF_CHN * complex_buf_size,
			csrc->conv_bufs + buf_size * i,
			buf_size
		);
		pointwiseMultiplication(
			csrc->conv_bufs + buf_size * i,
			csrc->distance_factor,
			complex_buf_size
		);
		pointwiseMultiplication(
			csrc->conv_bufs + buf_size * i + complex_buf_size,
			csrc->distance_factor,
			complex_buf_size
		);
	}
	complexScaling(csrc->conv_bufs, omegas[5] * omegas[1], buf_size);
	complexScaling(csrc->conv_bufs + buf_size, omegas[5] * omegas[0], buf_size);
	complexScaling(csrc->conv_bufs + 2 * buf_size, omegas[4] * omegas[3], buf_size);
	complexScaling(csrc->conv_bufs + 3 * buf_size, omegas[4] * omegas[2], buf_size);

	pointwiseAddition(
		csrc->conv_bufs,
		csrc->conv_bufs + buf_size,
		csrc->intermediate,
		PAD_LEN + 2);
	for (int i = 2; i < 4; i++) {
		pointwiseAddition(
			csrc->intermediate,
			csrc->conv_bufs + buf_size * i,
			buf_size);
	}
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
	CHECK_CUFFT_ERRORS(cufftExecC2R(gsrc->plans[blockNo * 3 + 1], (cufftComplex*)d_output, d_output));
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
	memcpy(gpu_output, gsrc->intermediate[0], FRAMES_PER_BUFFER * 2 * sizeof(float));
	memcpy(cpu_output, ((float*)csrc->intermediate) + 2 * (PAD_LEN - FRAMES_PER_BUFFER), FRAMES_PER_BUFFER * 2 * sizeof(float));
	if (precisionChecking(gpu_output, cpu_output, FRAMES_PER_BUFFER * 2)) {
		printf("ERROR: Inaccurate Case 4 output\n");
	}
	else {
		printf("Successful Case 4 accurate output\n");
	}
	for (int i = 0; i < PAD_LEN + 2; i++) {
		gsrc->x[0][i] = 0.0f;
		gsrc->x[1][i] = 0.0f;
		
		csrc->x[i] = 0.0f;
	}
	for (int i = 0; i < FRAMES_PER_BUFFER * 2; i++) {
		gsrc->intermediate[0][i] = 0.0f;
		gsrc->intermediate[1][i] = 0.0f;
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


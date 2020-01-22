#include "hrtf_signals.cuh"

float* d_hrtf;
cufftComplex* d_fft_hrtf;
float* hrtf;
fftwf_complex* fft_hrtf;
int elevation_pos[NUM_ELEV] =
{ -40,  -30,  -20,  -10,    0,   10,   20,   30,   40,   50,    60,    70,    80,  90 };
float azimuth_inc[NUM_ELEV] =
{ 6.43f, 6.00f, 5.00f, 5.00f, 5.00f, 5.00f, 5.00f, 6.00f, 6.43f, 8.00f, 10.00f, 15.00f, 30.00f, 361.0f };
//56	+ 60	+ 72 + 72	+ 72	+ 72  + 72		+ 60	+ 56 + 45 + 36		+ 24	+ 12	+ 1 = 710
int azimuth_offset[NUM_ELEV + 1];


/* on entry obj_ele and obj_azi are the new object position
* on exit hrtf_idx is set to the HRTF index of the closest HRTF position
*  hrtf_idx > 0 indicates to use right half-sphere HRTF
*  hrtf_idx < 0 indicates to create left half-sphere HRTF by exchanging L, R
*/
int pick_hrtf(float obj_ele, float obj_azi)
{
	int i, n, ele_idx, hrtf_idx = 0;
	float d, dmin;

	/* find closest elevation position */
	obj_ele = std::round(obj_ele / 10) * 10;
	dmin = 1e37f;
	for (i = 0; i < NUM_ELEV; i++) {
		d = obj_ele - elevation_pos[i];
		d = d > 0 ? d : -d;
		if (d < dmin) {
			dmin = d;
			ele_idx = i;
		}
	}
	/* find closest azimuth position */
	obj_azi = std::round(obj_azi);
	dmin = 1e37f;
	n = azimuth_offset[ele_idx + 1] - azimuth_offset[ele_idx];
	for (i = 0; i < n; i++) {
		d = obj_azi - i * azimuth_inc[ele_idx];
		d = d > 0 ? d : -d;
		if (d < dmin) {
			dmin = d;
			hrtf_idx = azimuth_offset[ele_idx] + i;
		}
	}

	/* return hrtf index */
	return(hrtf_idx);
}


/*HRTF Impulse reading for GPU*/
int read_and_error_check(char* input, float* hrtf) {
	/* sndfile data structures */
	SNDFILE* sndfile;
	SF_INFO sfinfo;
	/* zero libsndfile structures */
	memset(&sfinfo, 0, sizeof(sfinfo));
	/* open hrtf file */
	if ((sndfile = sf_open(input, SFM_READ, &sfinfo)) == NULL) {
		fprintf(stderr, "Error: could not open hrtf file:\n%s\n", input);
		fprintf(stderr, "%s\n", sf_strerror(sndfile));
		return -1;
	}
	/* check signal parameters */
	if (sfinfo.channels != 1) {
		fprintf(stderr, "ERROR: incorrect number of channels in HRTF\n");
		return -1;
	}
	if (sfinfo.samplerate != SAMP_RATE) {
		fprintf(stderr, "ERROR: incorrect sampling rate\n");
		return -1;
	}
	/* read HRTF signal */
	unsigned num_samples = sfinfo.frames * sfinfo.channels;

	if (sf_read_float(sndfile, hrtf, num_samples) != num_samples) {
		fprintf(stderr, "ERROR: cannot read HRTF signal\n");
		return -1;
	}

	/* close file */
	sf_close(sndfile);

	return 0;
}
void allocate_hrtf_buffers() {
	hrtf = fftwf_alloc_real(NUM_HRTF * HRTF_CHN * PAD_LEN);
	fft_hrtf = fftwf_alloc_complex(NUM_HRTF * HRTF_CHN * (PAD_LEN / 2 + 1));
	for (int i = 0; i < NUM_HRTF * HRTF_CHN * PAD_LEN; i++) {
		hrtf[i] = 0.0f;
	}
	size_t size = sizeof(float) * NUM_HRTF * HRTF_CHN * PAD_LEN;
	checkCudaErrors(cudaMalloc((void**)&d_hrtf, size));
	size = sizeof(cufftComplex) * NUM_HRTF * HRTF_CHN * (PAD_LEN / 2 + 1);
	checkCudaErrors(cudaMalloc((void**)&d_fft_hrtf, size));
}

void cleanup_hrtf_buffers() {
	fftwf_free(hrtf);
	fftwf_free(fft_hrtf);
	checkCudaErrors(cudaFree(d_hrtf));
	checkCudaErrors(cudaFree(d_fft_hrtf));
}
int read_hrtf_signals(void) {
	allocate_hrtf_buffers();
	char hrtf_file[PATH_LEN];
	float azi;
	int j = 0;
	azimuth_offset[0] = 0;
	int n[] = { PAD_LEN };
	fftwf_plan plan = fftwf_plan_many_dft_r2c(
		1, n, NUM_HRTF * 2,
		hrtf, NULL, 1, PAD_LEN,
		fft_hrtf, NULL, 1, PAD_LEN / 2 + 1,
		FFTW_ESTIMATE);
	for (int i = 0; i < NUM_ELEV; i++) {
		int ele = elevation_pos[i];
		for (azi = 0; azi < 360; azi += azimuth_inc[i]) {


			sprintf(hrtf_file, "%s/elev%d/L%de%03da.wav", HRTF_DIR, ele, ele, (int)round(azi));
			/* Print file information */
			//printf("%3d %3d %s\n", i, j, hrtf_file);
			if (read_and_error_check(hrtf_file, hrtf + j * HRTF_CHN * PAD_LEN)) {
				return -1;
			}

			sprintf(hrtf_file, "%s/elev%d/R%de%03da.wav", HRTF_DIR, ele, ele, (int)round(azi));
			//printf("%3d %3d %s\n", i, j, hrtf_file);
			if (read_and_error_check(hrtf_file, hrtf + j * HRTF_CHN * PAD_LEN + PAD_LEN)) {
				return -1;
			}
			j++;
		}

		azimuth_offset[i + 1] = j;
	}
	size_t size = sizeof(float) * NUM_HRTF * HRTF_CHN * PAD_LEN;
	checkCudaErrors(cudaMemcpy(d_hrtf, hrtf, size, cudaMemcpyHostToDevice));

	fftwf_execute(plan);
	fftwf_destroy_plan(plan);

	printf("\nHRTF index offsets for each elevation:\n");
	for (int i = 0; i < NUM_ELEV + 1; i++) {
		printf("%3d ", azimuth_offset[i]);
	}
	printf("\n");
	return 0;
}


void transform_hrtfs() {

	//// Precision Check for HRTFs

	//float* gpu_hrtfs = new float[NUM_HRTF * HRTF_CHN * (PAD_LEN + 2)];
	///*checkCudaErrors(cudaMemcpy(gpu_hrtfs, d_hrtf, NUM_HRTF * HRTF_CHN * PAD_LEN * sizeof(float), cudaMemcpyDeviceToHost));

	//if (precisionChecking(gpu_hrtfs, hrtf, NUM_HRTF * HRTF_CHN * PAD_LEN)) {
	//	printf("ERROR: Inaccurate HRIRs\n");
	//}
	//else {
	//	printf("Accurate HRIRs\n");
	//}*/
	//
	//cufftHandle plan;
	//CHECK_CUFFT_ERRORS(cufftPlan1d(&plan, PAD_LEN, CUFFT_R2C, NUM_HRTF * 2));
	//CHECK_CUFFT_ERRORS(cufftExecR2C(plan, d_hrtf, d_fft_hrtf));
	//CHECK_CUFFT_ERRORS(cufftDestroy(plan));

	//checkCudaErrors(cudaMemcpy(gpu_hrtfs, d_fft_hrtf, NUM_HRTF * HRTF_CHN * (PAD_LEN + 2) * sizeof(float), cudaMemcpyDeviceToHost));

	//if (precisionChecking(gpu_hrtfs, (float*)fft_hrtf, NUM_HRTF * HRTF_CHN * (PAD_LEN + 2), 1e-6)) {
	//	printf("ERROR: Inaccurate HRTF\n");
	//	//exit(EXIT_FAILURE);
	//}
	//else {
	//	printf("Accurate HRTFs\n");
	//}
	//printf("\n");
	//int batch = NUM_HRTF * 2;
	//float* d_blah1;
	//cufftComplex* d_blah2;
	//float* blah1 = fftwf_alloc_real(batch * PAD_LEN);
	//fftwf_complex* blah2 = fftwf_alloc_complex(batch * (PAD_LEN / 2 + 1));
	//fftwf_complex* blah3 = fftwf_alloc_complex(batch * (PAD_LEN / 2 + 1));
	//checkCudaErrors(cudaMalloc(&d_blah1, batch * PAD_LEN * sizeof(float)));
	//checkCudaErrors(cudaMalloc(&d_blah2, batch * (PAD_LEN + 2) * sizeof(float)));
	//checkCudaErrors(cudaMemcpy(d_blah1, hrtf, batch * PAD_LEN * sizeof(float), cudaMemcpyHostToDevice));
	//
	//CHECK_CUFFT_ERRORS(cufftPlan1d(&plan, PAD_LEN, CUFFT_R2C, batch));
	//CHECK_CUFFT_ERRORS(cufftExecR2C(plan, d_blah1, d_blah2));
	//CHECK_CUFFT_ERRORS(cufftDestroy(plan));

	//int n[] = { PAD_LEN };
	//fftwf_plan fftw_plan;
	//fftw_plan = fftwf_plan_many_dft_r2c(
	//	1, n, batch,
	//	blah1, NULL, 1, PAD_LEN,
	//	blah3, NULL, 1, PAD_LEN / 2 + 1,
	//	FFTW_ESTIMATE);
	//memcpy(blah1, hrtf, batch * PAD_LEN * sizeof(float));
	//fftwf_execute(fftw_plan);
	//fftwf_destroy_plan(fftw_plan);


	//fftw_plan = fftwf_plan_dft_r2c_1d(PAD_LEN, blah1, blah2, FFTW_ESTIMATE);
	//memcpy(blah1, hrtf, batch * PAD_LEN * sizeof(float));
	//fftwf_execute(fftw_plan);
	//for (int i = 1; i < batch; i++) {
	//	fftw_plan = fftwf_plan_dft_r2c_1d(PAD_LEN, blah1 + i * PAD_LEN, blah2 + i * (PAD_LEN / 2 + 1), FFTW_ESTIMATE);
	//	fftwf_execute(fftw_plan);
	//}

	//fftwf_destroy_plan(fftw_plan);

	//float* buf = new float[batch * (PAD_LEN + 2)];
	//checkCudaErrors(cudaMemcpy(buf, d_blah2, batch * (PAD_LEN + 2) * sizeof(float), cudaMemcpyDeviceToHost));
	//if (precisionChecking((float*)blah2, (float*)buf, batch * (PAD_LEN + 2), 1e-6)) {
	//	printf("Inccurate GPU vs manual batched FFTW precision check\n");
	//}
	//else {
	//	printf("Successful GPU vs manual batched FFTW precision check\n");
	//}
	//printf("\n");
	//if (precisionChecking((float*)blah3, (float*)blah2, batch * (PAD_LEN + 2))) {
	//	printf("FFTW Precision Error\n");
	//}
	//printf("\n");
	//if (precisionChecking((float*)blah3, (float*)buf, batch * (PAD_LEN + 2), 1e-6)) {
	//	printf("Inaccurate GPU vs advanced API FFTW check\n");
	//}
	//else {
	//	printf("Successful GPU vs advanced API FFTW check\n");
	//}
	//printf("\n");
	//checkCudaErrors(cudaFree(d_blah1));
	//checkCudaErrors(cudaFree(d_blah2));
	//fftwf_free(blah1);
	//fftwf_free(blah2);
	//fftwf_free(blah3);
	//delete[] gpu_hrtfs;
	//delete[] buf;
	checkCudaErrors(cudaMemcpy(d_fft_hrtf, fft_hrtf, NUM_HRTF * 2 * (PAD_LEN + 2) * sizeof(float), cudaMemcpyHostToDevice));
}
#include "hrtf_signals.cuh"

float hrtf[NUM_HRFT][HRTF_LEN*HRTF_CHN];   /* interleaved HRTF impulse responses */

int elevation_pos[NUM_ELEV] =
{ -40,  -30,  -20,  -10,    0,   10,   20,   30,   40,   50,    60,    70,    80,  90 };
float azimuth_inc[NUM_ELEV] =
{ 6.43, 6.00, 5.00, 5.00, 5.00, 5.00, 5.00, 6.00, 6.43, 8.00, 10.00, 15.00, 30.00, 181 };
int azimuth_offset[NUM_ELEV + 1];

/*Read hrtf signals on CPU/RAM*/
int read_hrtf_signals(void) {
	char hrtf_file[PATH_LEN];
	int i, j, ele, num_samples, count;
	float azi;
	/* sndfile data structures */
	SNDFILE *sndfile;
	SF_INFO sfinfo;

	j = 0;
	azimuth_offset[0] = 0;
	for (i = 0; i<NUM_ELEV; i++) {
		ele = elevation_pos[i];
		for (azi = 0; azi <= 180; azi += azimuth_inc[i]) {
			sprintf(hrtf_file, "%s/elev%d/H%de%03da.wav", HRTF_DIR, ele, ele, (int)round(azi));

			/* zero libsndfile structures */
			memset(&sfinfo, 0, sizeof(sfinfo));

			/* open hrtf file */
			if ((sndfile = sf_open(hrtf_file, SFM_READ, &sfinfo)) == NULL) {
				fprintf(stderr, "Error: could not open hrtf file:\n%3d %3d %s\n", i, j, hrtf_file);
				fprintf(stderr, "%s\n", sf_strerror(sndfile));
				return -1;
			}
			/* check signal parameters */
			if (sfinfo.channels != HRTF_CHN) {
				fprintf(stderr, "ERROR: incorrect number of channels in HRTF\n");
				return -1;
			}
			if (sfinfo.samplerate != SAMP_RATE) {
				fprintf(stderr, "ERROR: incorrect sampling rate\n");
				return -1;
			}
			/* Print file information */
			printf("%3d %3d %s Frames: %d, Channels: %d, Samplerate: %d\n",
				i, j, hrtf_file, (int)sfinfo.frames, sfinfo.channels, sfinfo.samplerate);

			/* read HRTF signal */
			num_samples = sfinfo.frames*sfinfo.channels;
			if ((count = sf_read_float(sndfile, hrtf[j], num_samples)) != num_samples) {
				fprintf(stderr, "ERROR: cannot read HRTF signal %3d\n", j);
				return -1;
			}

			/* close file */
			sf_close(sndfile);
			j++;
		}
		/* for given azimuth j, azimuth_offset[j] is where that set of hrtf begins
		* in set of NUM_HRTF
		*/
		azimuth_offset[i + 1] = j;
	}
	printf("\nHRTF index offsets for each elevation:\n");
	for (i = 0; i<NUM_ELEV + 1; i++) {
		printf("%3d ", azimuth_offset[i]);
	}
	printf("\n");
	return 0;
}

/* on entry obj_ele and obj_azi are the new object position
* on exit hrtf_idx is set to the HRTF index of the closest HRTF position
*  hrtf_idx > 0 indicates to use right half-sphere HRTF
*  hrtf_idx < 0 indicates to create left half-sphere HRTF by exchanging L, R
*/
int pick_hrtf(float obj_ele, float obj_azi)
{
	int i, n, ele_idx, obj_azi_sign, hrtf_idx;
	float d, dmin;

	/* save azimuth sign and force obj_azi to right half-sphere */
	obj_azi_sign = 1;
	if (obj_azi < 0) {
		obj_azi_sign = -1;
		obj_azi = -obj_azi;
	}

	/* find closest elevation position */
	obj_ele = std::round(obj_ele / 10) * 10;
	dmin = 1e37;
	for (i = 0; i<NUM_ELEV; i++) {
		d = obj_ele - elevation_pos[i];
		d = d > 0 ? d : -d;
		if (d < dmin) {
			dmin = d;
			ele_idx = i;
		}
	}
	/* find closest azimuth position */
	obj_azi = std::round(obj_azi);
	dmin = 1e37;
	n = azimuth_offset[ele_idx + 1] - azimuth_offset[ele_idx];
	for (i = 0; i<n; i++) {
		d = obj_azi - i*azimuth_inc[ele_idx];
		d = d > 0 ? d : -d;
		if (d < dmin) {
			dmin = d;
			hrtf_idx = azimuth_offset[ele_idx] + i;
		}
	}

	/* return hrtf index */
	return(hrtf_idx * obj_azi_sign);
}

/* convolve signal buffer with HRTF
* new signal starts at HRTF_LEN frames into x buffer
* x is mono input signal
* HRTF and y are interleaved by channel
* y_len is in frames
*/
int convolve_hrtf(float *input, int hrtf_idx, float *output, int outputLen, float gain) {
	int i, j, n, k, swap_chan, j_hrtf;
	float *p_hrtf;
	if (gain > 1)
		gain = 1;
	if (hrtf_idx >= 0) {
		swap_chan = false;
		p_hrtf = hrtf[hrtf_idx];
	}
	else {
		swap_chan = true;
		p_hrtf = hrtf[-hrtf_idx];
	}

	/* zero output buffer */
	for (i = 0; i<outputLen*HRTF_CHN; i++) {
		output[i] = 0.0;
	}
	for (n = 0; n < outputLen; n++) {
		for (k = 0; k < HRTF_LEN; k++) {
			for (j = 0; j < HRTF_CHN; j++) {
				/* outputLen and HRTF_LEN are n frames, output and hrtf are interleaved
				* input is mono
				*/
				j_hrtf = (swap_chan == false) ? j : (j == 0) ? 1 : 0;
				output[2 * n + j] += input[n - k] * p_hrtf[2 * k + j_hrtf];
			}
			output[2 * n] *= gain;
			output[2 * n + 1] *= gain;
		}
	}
	return 0;
}
////////////////////////////////////////////////////////////////////////////////
/*NOTE: GPU Convolution was not fast enough because of the large overhead
of FFT and IFFT. Keeping the code here for future purposes*/
/*HRTF Impulse reading for GPU/DRAM*/
//float *d_hrtf;
//int read_hrtf_signals(void) {
//	char hrtf_file[PATH_LEN];
//	int i, j, ele, num_samples, count;
//	float azi;
//	/* sndfile data structures */
//	SNDFILE *sndfile;
//	SF_INFO sfinfo;
//
//	j = 0;
//	azimuth_offset[0] = 0;
//	size_t size = sizeof(float) * NUM_HRFT * HRTF_LEN * HRTF_CHN;
//	checkCudaErrors(cudaMalloc((void**)&d_hrtf, size));
//
//	for (i = 0; i<NUM_ELEV; i++) {
//		ele = elevation_pos[i];
//		for (azi = 0; azi <= 180; azi += azimuth_inc[i]) {
//			sprintf(hrtf_file, "%s/elev%d/H%de%03da.wav", HRTF_DIR, ele, ele, (int)round(azi));
//
//			/* zero libsndfile structures */
//			memset(&sfinfo, 0, sizeof(sfinfo));
//
//			/* open hrtf file */
//			if ((sndfile = sf_open(hrtf_file, SFM_READ, &sfinfo)) == NULL) {
//				fprintf(stderr, "Error: could not open hrtf file:\n%3d %3d %s\n", i, j, hrtf_file);
//				fprintf(stderr, "%s\n", sf_strerror(sndfile));
//				return -1;
//			}
//			/* check signal parameters */
//			if (sfinfo.channels != HRTF_CHN) {
//				fprintf(stderr, "ERROR: incorrect number of channels in HRTF\n");
//				return -1;
//			}
//			if (sfinfo.samplerate != SAMP_RATE) {
//				fprintf(stderr, "ERROR: incorrect sampling rate\n");
//				return -1;
//			}
//			/* Print file information */
//			printf("%3d %3d %s Frames: %d, Channels: %d, Samplerate: %d\n",
//				i, j, hrtf_file, (int)sfinfo.frames, sfinfo.channels, sfinfo.samplerate);
//			/* read HRTF signal */
//			num_samples = sfinfo.frames*sfinfo.channels;
//			if ((count = sf_read_float(sndfile, hrtf[j], num_samples)) != num_samples) {
//				fprintf(stderr, "ERROR: cannot read HRTF signal %3d\n", j);
//				return -1;
//			}
//			/* close file */
//			sf_close(sndfile);
//			j++;
//		}
//		checkCudaErrors(cudaMemcpy(d_hrtf, hrtf, size, cudaMemcpyHostToDevice));
//		azimuth_offset[i + 1] = j;
//	}
//	printf("\nHRTF index offsets for each elevation:\n");
//	for (i = 0; i<NUM_ELEV + 1; i++) {
//		printf("%3d ", azimuth_offset[i]);
//	}
//	printf("\n");
//	return 0;
//}
////////////////////////////////////////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////////////
/*GPU Convolution was not fast enough because of the large overhead
of FFT and IFFT. Keeping the code here for future purposes*/
//void GPUconvolve_hrtf(float *x, int x_len, int hrtf_idx, float *output, int y_len, float gain) {
//	int i, j, n, k, swap_chan, j_hrtf;
//	float *p_hrtf;
//	if (gain > 1)
//		gain = 1;
//	if (hrtf_idx >= 0) {
//		swap_chan = false;
//		p_hrtf = hrtf[hrtf_idx];
//	}
//	else {
//		swap_chan = true;
//		p_hrtf = hrtf[-hrtf_idx];
//	}
//
//	/* zero output buffer */
//	/*for (i = 0; i<y_len*HRTF_CHN; i++) {
//		output[i] = 0.0;
//	}*/
//	convolveMe(output, x, x_len, p_hrtf, gain, d_hrtf);
//}
////////////////////////////////////////////////////////////////////////////////

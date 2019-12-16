#include "hrtf_signals.cuh"

float* d_hrtf;
float* hrtf;
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
	int i, n, ele_idx, hrtf_idx;
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


/*HRTF Impulse reading for GPU/DRAM*/
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
int read_hrtf_signals(void) {
	/*adding + 2 padding for the FFT*/
	hrtf = new float[NUM_HRTF * HRTF_CHN * (PAD_LEN + 2)];
	for (int i = 0; i < NUM_HRTF * HRTF_CHN * (PAD_LEN + 2); i++) {
		hrtf[i] = 0.0f;
	}
	char hrtf_file[PATH_LEN];
	float azi;
	int j = 0;
	azimuth_offset[0] = 0;
	size_t size = sizeof(float) * NUM_HRTF * HRTF_CHN * (PAD_LEN + 2);
	checkCudaErrors(cudaMalloc((void**)&d_hrtf, size));
	for (int i = 0; i < NUM_ELEV; i++) {
		int ele = elevation_pos[i];
		for (azi = 0; azi < 360; azi += azimuth_inc[i]) {


			sprintf(hrtf_file, "%s/elev%d/L%de%03da.wav", HRTF_DIR, ele, ele, (int)round(azi));
			/* Print file information */
			//printf("%3d %3d %s\n", i, j, hrtf_file);
			if (read_and_error_check(hrtf_file, hrtf + j * HRTF_CHN * (PAD_LEN + 2))) {
				return -1;
			}

			sprintf(hrtf_file, "%s/elev%d/R%de%03da.wav", HRTF_DIR, ele, ele, (int)round(azi));
			//printf("%3d %3d %s\n", i, j, hrtf_file);
			if (read_and_error_check(hrtf_file, hrtf + j * HRTF_CHN * (PAD_LEN + 2) + PAD_LEN + 2)) {
				return -1;
			}
			j++;
		}

		azimuth_offset[i + 1] = j;
	}
	
	checkCudaErrors(cudaMemcpy(d_hrtf, hrtf, size, cudaMemcpyHostToDevice));
	printf("\nHRTF index offsets for each elevation:\n");
	for (int i = 0; i < NUM_ELEV + 1; i++) {
		printf("%3d ", azimuth_offset[i]);
	}
	printf("\n");
	return 0;
}


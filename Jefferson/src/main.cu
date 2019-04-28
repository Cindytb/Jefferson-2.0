#include "main.cuh"
#include <chrono>
#include <thread>
#include <cuda.h>
#include <cuda_profiler_api.h>
int main(int argc, char *argv[]){
	if (argc > 3 ) {
		fprintf(stderr, "Usage: %s input.wav reverb.wav", argv[0]);
		return 0;
	}
	
	data.count = 0;
	data.length = 0;
	data.gain = 0.99074;
	data.hrtf_idx = 151;
	#if(DEBUGMODE != 1)
		/*Initialize & read files*/
		cudaFFT(argc, argv, p);
		// SNDFILE *test;
		// SF_INFO test1;
		// test = sf_open("scrap.wav", SFM_READ, &test1);
	
	////////////////////////////////////////////////////////////////////////////////
	///*NOTE: GPU Convolution was not fast enough because of the large overhead
	//of FFT and IFFT. Keeping the code here for future purposes*/
	//
	//checkCudaErrors(cudaMalloc((void**)&p->d_x, HRTF_LEN - 1 + FRAMES_PER_BUFFER));
	////////////////////////////////////////////////////////////////////////////////
	
		fprintf(stderr, "Opening and Reading HRTF signals\n");
		/*Open & read hrtf files*/

		if (read_hrtf_signals() != 0) {
			exit(EXIT_FAILURE);
		}
		p->hrtf_idx = 0;


		fprintf(stderr, "Opening output file\n");
		SF_INFO osfinfo;
		osfinfo.channels = 2;
		osfinfo.samplerate = 44100;
		osfinfo.format = SF_FORMAT_PCM_24 | SF_FORMAT_WAV;
		p->sndfile = sf_open("ofile.wav", SFM_WRITE, &osfinfo);
		p->count = 0;
		/*2019 version*/
		p->streams = (cudaStream_t *)malloc(10 * sizeof(cudaStream_t));
		for (int i = 0; i < 5; i++){
			/*Allocating memory for the inputs*/
			checkCudaErrors(cudaMalloc(&(p->d_input[i]), COPY_AMT * sizeof(float)));
			/*Allocating memory for the outputs*/
			checkCudaErrors(cudaMalloc(&(p->d_output[i]), FRAMES_PER_BUFFER * HRTF_CHN * sizeof(float)));
			/*Creating the streams*/
			checkCudaErrors(cudaStreamCreate(&(p->streams[i * 2])));
			checkCudaErrors(cudaStreamCreate(&(p->streams[i * 2 + 1])));
		}
		/*Allocating pinned memory for incoming transfer*/
		checkCudaErrors(cudaMallocHost(&(p->x), COPY_AMT * sizeof(float)));
		/*Allocating pinned memory for outgoing transfer*/
		checkCudaErrors(cudaMallocHost(&(p->intermediate), FRAMES_PER_BUFFER * HRTF_CHN * sizeof(float)));

		/*Setting initial input to 0*/
		for (int i = 0; i < HRTF_LEN - 1; i++){
			p->x[i] = 0.0f;
		}
		p->blockNo = 0;
		cudaProfilerStart();

		/*ROUND 1*/
		memcpy(p->x + HRTF_LEN - 1, p->buf + p->count, FRAMES_PER_BUFFER * sizeof(float));
		p->count += FRAMES_PER_BUFFER;
		/*Send B1*/
		checkCudaErrors(cudaMemcpyAsync(p->d_input[p->blockNo], p->x, COPY_AMT * sizeof(float), cudaMemcpyHostToDevice, p->streams[p->blockNo]));

		/*overlap-save*/
		memcpy(p->x, p->x + FRAMES_PER_BUFFER, (HRTF_LEN - 1) * sizeof(float));
		p->blockNo++;

		/*ROUND 2*/
		/*Copy new input chunk into pinned memory*/
		memcpy(p->x + HRTF_LEN - 1, p->buf + p->count, FRAMES_PER_BUFFER * sizeof(float));
		p->count += FRAMES_PER_BUFFER;
		
		/*Send B2*/
		checkCudaErrors(cudaMemcpyAsync(p->d_input[p->blockNo], p->x, COPY_AMT * sizeof(float), cudaMemcpyHostToDevice, p->streams[p->blockNo * 2]));
		/*Process B1*/
		GPUconvolve_hrtf(p->d_input[p->blockNo - 1] + HRTF_LEN, p->hrtf_idx, p->d_output[p->blockNo - 1], FRAMES_PER_BUFFER, p->gain, &(p->streams[(p->blockNo - 1) * 2]));

		/*overlap-save*/
		memcpy(p->x, p->x + FRAMES_PER_BUFFER, (HRTF_LEN - 1) * sizeof(float));
		p->blockNo++;

		/*ROUND 3*/
		/*Copy new input chunk into pinned memory*/
		memcpy(p->x + HRTF_LEN - 1, p->buf + p->count, FRAMES_PER_BUFFER * sizeof(float));
		p->count += FRAMES_PER_BUFFER;

		/*Send B3*/
		fprintf(stderr, "%i %i %i", p->blockNo, p->blockNo - 1, p->blockNo - 2);
		checkCudaErrors(cudaMemcpyAsync(p->d_input[p->blockNo], p->x, COPY_AMT * sizeof(float), cudaMemcpyHostToDevice, p->streams[p->blockNo * 2]));
		/*Process B2*/
		GPUconvolve_hrtf(p->d_input[p->blockNo - 1] + HRTF_LEN, p->hrtf_idx, p->d_output[p->blockNo - 1], FRAMES_PER_BUFFER, p->gain, &(p->streams[(p->blockNo - 1) * 2]));
		/*Idle B1*/

		/*overlap-save*/
		memcpy(p->x, p->x + FRAMES_PER_BUFFER, (HRTF_LEN - 1) * sizeof(float));
		p->blockNo++;
		
		/*ROUND 4*/
		/*Copy new input chunk into pinned memory*/
		memcpy(p->x + HRTF_LEN - 1, p->buf + p->count, FRAMES_PER_BUFFER * sizeof(float));
		p->count += FRAMES_PER_BUFFER;

		/*Send B4*/
		checkCudaErrors(cudaMemcpyAsync(p->d_input[p->blockNo % 5], p->x, COPY_AMT * sizeof(float), cudaMemcpyHostToDevice, p->streams[(p->blockNo) % 5 * 2]));
		/*Process B3*/
		GPUconvolve_hrtf(p->d_input[(p->blockNo - 1) % 5] + HRTF_LEN, p->hrtf_idx, p->d_output[p->blockNo - 1], FRAMES_PER_BUFFER, p->gain, &(p->streams[(p->blockNo - 1) % 5 * 2]));
		/*Idle B2*/

		/*Idle B1*/

		memcpy(p->x, p->x + FRAMES_PER_BUFFER, (HRTF_LEN - 1) * sizeof(float));
		p->blockNo++;

		/*ROUND 5*/
		/*Copy new input chunk into pinned memory*/
		memcpy(p->x + HRTF_LEN - 1, p->buf + p->count, FRAMES_PER_BUFFER * sizeof(float));
		p->count += FRAMES_PER_BUFFER;

		/*Send B5*/
		checkCudaErrors(cudaMemcpyAsync(p->d_input[p->blockNo % 5], p->x, COPY_AMT * sizeof(float), cudaMemcpyHostToDevice, p->streams[(p->blockNo) % 5 * 2]));
		/*Process B4*/
		GPUconvolve_hrtf(p->d_input[(p->blockNo - 1) % 5] + HRTF_LEN, p->hrtf_idx, p->d_output[p->blockNo - 1], FRAMES_PER_BUFFER, p->gain, &(p->streams[(p->blockNo - 1) % 5 * 2]));
		
		/*Idle B3
		/*Idle B2*/

		/*Return B1*/
		checkCudaErrors(cudaMemcpyAsync(p->intermediate, p->d_output[(p->blockNo - 4) % 5], FRAMES_PER_BUFFER * 2 * sizeof(float), cudaMemcpyDeviceToHost, p->streams[(p->blockNo - 4) % 5 * 2]));
		memcpy(p->x, p->x + FRAMES_PER_BUFFER, (HRTF_LEN - 1) * sizeof(float));
		p->blockNo++;
		
//		sf_close(test);
		checkCudaErrors(cudaDeviceSynchronize());
	#endif
	

#if(DEBUGMODE != 1)
	fprintf(stderr, "\n\n\n\nInitializing PortAudio\n\n\n\n");
	initializeAlsa(SAMPLE_RATE, p);
	printf("\n\n\n\nStarting playout\n");
	// fprintf(stderr, " %i %i %i %i %i\n", p->blockNo, p->blockNo - 1, p->blockNo - 2, p->blockNo - 3, p->blockNo - 4);
#endif
	///////////////////////////////////////////////////////////////////////////////////////////////////
	/*MAIN FUNCTIONAL LOOP*/
	/*Here to debug without graphics*/
#if DEBUGMODE == 2
	std::this_thread::sleep_for(std::chrono::seconds((p->length)/44100));
	//char merp = getchar();
#else
	graphicsMain(argc, argv, p);
#endif
	
	///////////////////////////////////////////////////////////////////////////////////////////////////
	// CUcontext pctx;
	// printf("%i\n", cuCtxPopCurrent(&pctx));
	/*THIS SECTION WILL NOT RUN IF GRAPHICS IS TURNED ON*/
	/*Placed here to properly close files when debugging without graphics*/
	cudaProfilerStop();
	/*Close output file*/
	sf_close(p->sndfile);
	closeAlsa();
	for (int i = 0; i < 3; i++){
		checkCudaErrors(cudaFree(p->d_input[i]));
		checkCudaErrors(cudaFree(p->d_output[i]));
		checkCudaErrors(cudaStreamDestroy(p->streams[i * 2]));
		checkCudaErrors(cudaStreamDestroy(p->streams[i * 2 + 1]));
	}
	checkCudaErrors(cudaFreeHost(p->intermediate));
	checkCudaErrors(cudaFreeHost(p->x));
	

	return 0;
}

/* This routine will be called by the PortAudio engine when audio is needed.
* It may called at interrupt level on some machines so don't do anything
* in the routine that requires significant resources.
*/
//static int paCallback(const void *inputBuffer, void *outputBuffer,
//	unsigned long framesPerBuffer,
//	const PaStreamCallbackTimeInfo* timeInfo,
//	PaStreamCallbackFlags statusFlags,
//	void *userData)
//{
//	/* Cast data passed through stream to our structure. */
//	Data *p = (Data *)userData;
//	float *output = (float *)outputBuffer;
//	//float *input = (float *)inputBuffer; /* input not used in this code */
//	float *px;
//	int i;
//	float *buf = (float*)malloc(sizeof(float) * 2 * framesPerBuffer - HRTF_LEN);
//
//	/*CPU/RAM Copy data loop*/
//	for (int i = 0; i < framesPerBuffer; i++) {
//		p->x[HRTF_LEN - 1 + i] = p->buf[p->count];
//		p->count++;
//		if (p->count == p->length) {
//			p->count = 0;
//		}
//	}
//	/*convolve with HRTF on CPU*/
//	convolve_hrtf(&p->x[HRTF_LEN], p->hrtf_idx, output, framesPerBuffer, p->gain);
//	
//	/*Enable pausing of audio*/
//	if (p->pauseStatus == true) {
//		memset((float*)output, 0.0f, framesPerBuffer * 2);
//		return 0;
//	}
//	
//	////////////////////////////////////////////////////////////////////////////////
//	////////////////////////////////////////////////////////////////////////////////
//	/*NOTE: GPU Convolution was not fast enough because of the large overhead
//	of FFT and IFFT. Keeping the code here for future purposes*/
//	/*CUDA Copy*/
//	//cudaThreadSynchronize();
//	//int blockSize = 256;
//	//int numBlocks = (framesPerBuffer + blockSize - 1) / blockSize;
//	//if(p->count + framesPerBuffer <= p->length) {
//	//	copyMe << < numBlocks, blockSize >> > (framesPerBuffer, p->d_x, &p->dbuf[p->count]);
//	//	cudaThreadSynchronize();
//	//	p->count += framesPerBuffer;
//	//}
//	//
//	//else {
//	//	int remainder = p->length - p->count - framesPerBuffer;
//	//	copyMe << < numBlocks, blockSize >> > (p->length - p->count, p->d_x, &p->dbuf[p->count]);
//	//	p->count = 0;
//	//	copyMe << < numBlocks, blockSize >> > (remainder, p->d_x, &p->dbuf[p->count]);
//	//	p->count += remainder;
//	//}
//	/*Convolve on GPU*/
//	//GPUconvolve_hrtf(p->d_x, framesPerBuffer, p->hrtf_idx, output, framesPerBuffer, p->gain);
//	////////////////////////////////////////////////////////////////////////////////
//	////////////////////////////////////////////////////////////////////////////////
//
//
//	/* copy last HRTF_LEN-1 samples of x data to "history" part of x for use next time */
//	px = p->x;
//	for (i = 0; i<HRTF_LEN - 1; i++) {
//		px[i] = px[framesPerBuffer + i];
//	}
//	//sf_writef_float(p->sndfile, output, framesPerBuffer);
//	return 0;
//}

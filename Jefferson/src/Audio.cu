#include "Audio.cuh"

PaStream *stream;

void initializePA(int fs) {
	PaError err;
	fprintf(stderr, "\n\n\n");
#if DEBUG != 1
	/*PortAudio setup*/
	PaStreamParameters outputParams;
	PaStreamParameters inputParams;

	/* Initializing PortAudio */
	err = Pa_Initialize();
	if (err != paNoError) {
		printf("PortAudio error: %s\n", Pa_GetErrorText(err));
		printf("\nExiting.\n");
		fprintf(stderr, "PortAudio error: %s\n", Pa_GetErrorText(err));
		fprintf(stderr, "\nExiting.\n");
		exit(1);
	}

	/* Input stream parameters */
	inputParams.device = Pa_GetDefaultInputDevice();
	inputParams.channelCount = 1;
	inputParams.sampleFormat = paFloat32;
	inputParams.suggestedLatency =
		Pa_GetDeviceInfo(inputParams.device)->defaultLowInputLatency;
	inputParams.hostApiSpecificStreamInfo = NULL;

	/* Ouput stream parameters */
	outputParams.device = Pa_GetDefaultOutputDevice();
	outputParams.channelCount = 2;
	outputParams.sampleFormat = paFloat32;
	outputParams.suggestedLatency =
		Pa_GetDeviceInfo(outputParams.device)->defaultLowOutputLatency;
	outputParams.hostApiSpecificStreamInfo = NULL;

	/* Open audio stream */
	err = Pa_OpenStream(&stream,
		&inputParams, /* no input */
		&outputParams,
		fs, FRAMES_PER_BUFFER,
		paNoFlag, /* flags */
		paCallback,
		&data);

	if (err != paNoError) {
		printf("PortAudio error: open stream: %s\n", Pa_GetErrorText(err));
		printf("\nExiting.\n");
		fprintf(stderr, "PortAudio error: open stream: %s\n", Pa_GetErrorText(err));
		fprintf(stderr, "\nExiting.\n");
		exit(1);
	}

	/* Start audio stream */
	err = Pa_StartStream(stream);
	if (err != paNoError) {
		printf("PortAudio error: start stream: %s\n", Pa_GetErrorText(err));
		printf("\nExiting.\n");
		fprintf(stderr, "PortAudio error: start stream: %s\n", Pa_GetErrorText(err));
		fprintf(stderr, "\nExiting.\n");
		exit(1);
	}
#endif

}

void closePA() {
	PaError err;
#if DEBUG != 1
	/* Stop stream */
	err = Pa_StopStream(stream);
	if (err != paNoError) {
		printf("PortAudio error: stop stream: %s\n", Pa_GetErrorText(err));
		printf("\nExiting.\n");
		fprintf(stderr, "PortAudio error: stop stream: %s\n", Pa_GetErrorText(err));
		fprintf(stderr, "\nExiting.\n");
		exit(1);
	}

	/* Close stream */
	err = Pa_CloseStream(stream);
	if (err != paNoError) {
		printf("PortAudio error: close stream: %s\n", Pa_GetErrorText(err));
		printf("\nExiting.\n");
		fprintf(stderr, "PortAudio error: close stream: %s\n", Pa_GetErrorText(err));
		fprintf(stderr, "\nExiting.\n");
		exit(1);
	}

	/* Terminate PortAudio */
	err = Pa_Terminate();
	if (err != paNoError) {
		printf("PortAudio error: terminate: %s\n", Pa_GetErrorText(err));
		printf("\nExiting.\n");
		fprintf(stderr, "PortAudio error: terminate: %s\n", Pa_GetErrorText(err));
		fprintf(stderr, "\nExiting.\n");
		exit(1);
	}
#endif
}

static int paCallback(const void *inputBuffer, void *outputBuffer,
	unsigned long framesPerBuffer,
	const PaStreamCallbackTimeInfo* timeInfo,
	PaStreamCallbackFlags statusFlags,
	void *userData)
{
	/* Cast data passed through stream to our structure. */
	Data *p = (Data *)userData;
	float *output = (float *)outputBuffer;
	//float *input = (float *)inputBuffer; /* input not used in this code */
	float *px;
	unsigned int i;
	float *buf = (float*)malloc(sizeof(float) * 2 * framesPerBuffer - HRTF_LEN);

	/*CPU/RAM Copy data loop*/
	for (int i = 0; i < framesPerBuffer; i++) {
		p->x[HRTF_LEN - 1 + i] = p->buf[p->count];
		p->count++;
		if (p->count == p->length) {
			p->count = 0;
		}
	}
	/*convolve with HRTF on CPU*/
	convolve_hrtf(&p->x[HRTF_LEN], p->hrtf_idx, output, framesPerBuffer, p->gain);

	/*Enable pausing of audio*/
	if (p->pauseStatus == true) {
		for (i = 0; i < framesPerBuffer; i++) {
			output[2 * i] = 0;
			output[2 * i + 1] = 0;
		}
		return 0;
	}

	////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////
	/*NOTE: GPU Convolution was not fast enough because of the large overhead
	of FFT and IFFT. Keeping the code here for future purposes*/
	/*CUDA Copy*/
	//cudaThreadSynchronize();
	//int blockSize = 256;
	//int numBlocks = (framesPerBuffer + blockSize - 1) / blockSize;
	//if(p->count + framesPerBuffer <= p->length) {
	//	copyMe << < numBlocks, blockSize >> > (framesPerBuffer, p->d_x, &p->dbuf[p->count]);
	//	cudaThreadSynchronize();
	//	p->count += framesPerBuffer;
	//}
	//
	//else {
	//	int remainder = p->length - p->count - framesPerBuffer;
	//	copyMe << < numBlocks, blockSize >> > (p->length - p->count, p->d_x, &p->dbuf[p->count]);
	//	p->count = 0;
	//	copyMe << < numBlocks, blockSize >> > (remainder, p->d_x, &p->dbuf[p->count]);
	//	p->count += remainder;
	//}
	/*Convolve on GPU*/
	//GPUconvolve_hrtf(p->d_x, framesPerBuffer, p->hrtf_idx, output, framesPerBuffer, p->gain);
	////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////


	/* copy last HRTF_LEN-1 samples of x data to "history" part of x for use next time */
	px = p->x;
	for (i = 0; i<HRTF_LEN - 1; i++) {
		px[i] = px[framesPerBuffer + i];
	}
	//sf_writef_float(p->sndfile, output, framesPerBuffer);
	return 0;
}


#include "Audio.cuh"
#include <cuda.h>
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
void callback_func(float *output, Data *p){
	// printf("%i\n", p->count);
	// for(int i = 0; i < 8; i++){
	// 	fprintf(stderr, "Stream No:%i - %s\n", i, cudaStreamQuery(p->streams[i]) ? "Not Finished" : "Finished");
	// }
	// fprintf(stderr, "\n");
	checkCudaErrors(cudaStreamSynchronize(p->streams[(p->blockNo - 2) % 5 * 2]));
	/*Copy into p->x pinned memory*/
	if (p->count + FRAMES_PER_BUFFER < p->length){
		memcpy(p->x + HRTF_LEN - 1, p->buf + p->count, FRAMES_PER_BUFFER * sizeof(float));
		p->count += FRAMES_PER_BUFFER;
	}
	else{
		int rem = p->length - p->count;
		memcpy(p->x + HRTF_LEN - 1, p->buf + p->count, rem * sizeof(float));
		memcpy(p->x + HRTF_LEN - 1 + rem, p->buf, (FRAMES_PER_BUFFER - rem) * sizeof(float));
		p->count = FRAMES_PER_BUFFER - rem;
	}

	// if (p->blockNo == 5) {
	// 	printf("%i\n", cuCtxPushCurrent(0));
	// }
	// fprintf(stderr, "Stream %i %s\n", p->blockNo % 5, cudaStreamQuery(p->streams[p->blockNo % 5 * 2]) ? "Unfinished":"Finished");
	/*Enable pausing of audio*/
	if (p->pauseStatus == true) {
		for (int i = 0; i < FRAMES_PER_BUFFER; i++) {
			output[2 * i] = 0;
			output[2 * i + 1] = 0;
		}
		return;
	}
	memcpy(output, p->intermediate, FRAMES_PER_BUFFER * 2 * sizeof(float));
	// fprintf(stderr, "%i %i %i %i %i\n", p->blockNo % 5, (p->blockNo - 1) % 5, (p->blockNo - 2) % 5, (p->blockNo - 3) % 5, (p->blockNo - 4) % 5);
	/*Send*/
	checkCudaErrors(cudaMemcpyAsync(p->d_input[p->blockNo % 5], p->x, COPY_AMT * sizeof(float), cudaMemcpyHostToDevice, p->streams[(p->blockNo) % 5 * 2]));
	/*Process*/
	GPUconvolve_hrtf(p->d_input[(p->blockNo - 1) % 5] + HRTF_LEN, p->hrtf_idx, p->d_output[(p->blockNo - 1) % 5], FRAMES_PER_BUFFER, p->gain, &(p->streams[(p->blockNo - 1) % 5 * 2]));
	/*Idle blockNo - 2*/
	/*Idle blockNo - 3*/
	/*Return & fill intermediate*/
	checkCudaErrors(cudaMemcpyAsync(p->intermediate, p->d_output[(p->blockNo - 4) % 5], FRAMES_PER_BUFFER * 2 * sizeof(float), cudaMemcpyDeviceToHost, p->streams[(p->blockNo - 3) % 5 * 2]));

	
	/*Overlap-save*/
	memcpy(p->x, p->x + FRAMES_PER_BUFFER, (HRTF_LEN - 1) * sizeof(float));
	p->blockNo++;

	//sf_writef_float(p->sndfile, output, framesPerBuffer);
	return;
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
	callback_func(output, p);
	return 0;
}


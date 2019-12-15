#include "Audio.cuh"
#include "pa_asio.h"
#include <cuda.h>
PaStream *stream;
extern Data data;

void initializePA(int fs) {
	PaError err;
	fprintf(stderr, "\n\n\n");
#if DEBUG != 1
	/*PortAudio setup*/
	PaStreamParameters outputParams;

	/* Initializing PortAudio */
	err = Pa_Initialize();
	if (err != paNoError) {
		printf("PortAudio error: %s\n", Pa_GetErrorText(err));
		printf("\nExiting.\n");
		fprintf(stderr, "PortAudio error: %s\n", Pa_GetErrorText(err));
		fprintf(stderr, "\nExiting.\n");
		exit(1);
	}

	/* Ouput stream parameters */
	outputParams.device = Pa_GetDefaultOutputDevice();
	outputParams.channelCount = 2;
	outputParams.sampleFormat = paFloat32;
	outputParams.suggestedLatency =
		Pa_GetDeviceInfo(outputParams.device)->defaultLowOutputLatency;
	outputParams.hostApiSpecificStreamInfo = NULL;

	/* Open audio stream */
	err = Pa_OpenStream(&stream,
		NULL, /* no input */
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
	for(int i = 0; i < FRAMES_PER_BUFFER * 2; i++){
		output[i] = 0.0f;
	}
	for (int source_no = 0; source_no < p->num_sources; source_no++){

		SoundSource* curr_source = &(p->all_sources[source_no]);
		/*Enable pausing of audio*/
		if (p->pauseStatus == true) {
			for (int i = 0; i < FRAMES_PER_BUFFER; i++) {
				output[2 * i] = 0;
				output[2 * i + 1] = 0;
			}
			break;
		}

		
		
		/*Copy into curr_source->x pinned memory*/
		if (curr_source->count + FRAMES_PER_BUFFER < curr_source->length) {
			memcpy(
				curr_source->x[p->blockNo % FLIGHT_NUM] + (PAD_LEN - FRAMES_PER_BUFFER),  /*Go to the end and work backwards*/
				curr_source->buf + curr_source->count, 
				FRAMES_PER_BUFFER * sizeof(float));
			curr_source->count += FRAMES_PER_BUFFER;
		}
		else {
			int rem = curr_source->length - curr_source->count;
			memcpy(
				curr_source->x[p->blockNo % FLIGHT_NUM] + (PAD_LEN - FRAMES_PER_BUFFER),
				curr_source->buf + curr_source->count, 
				rem * sizeof(float));
			memcpy(
				curr_source->x[p->blockNo % FLIGHT_NUM] + (PAD_LEN - FRAMES_PER_BUFFER) + rem, 
				curr_source->buf, 
				(FRAMES_PER_BUFFER - rem) * sizeof(float));
			curr_source->count = FRAMES_PER_BUFFER - rem;
		}

		//memcpy(output, curr_source->intermediate[p->blockNo % FLIGHT_NUM], FRAMES_PER_BUFFER * 2 * sizeof(float));
		for (int i = 0; i < FRAMES_PER_BUFFER * 2; i++) {
			output[i] += curr_source->intermediate[p->blockNo % FLIGHT_NUM][i];
			if (output[i] > 1.0) {
				fprintf(stderr, "ALERT! CLIPPING AUDIO!\n");
			}
		}

		/*Send*/
		checkCudaErrors(cudaMemcpyAsync(
			curr_source->d_input[p->blockNo % FLIGHT_NUM], 
			curr_source->x[p->blockNo % FLIGHT_NUM],
			PAD_LEN * sizeof(float), 
			cudaMemcpyHostToDevice, 
			curr_source->streams[p->blockNo % FLIGHT_NUM * STREAMS_PER_FLIGHT])
		);
		
		/*Process*/
		//curr_source->fftConvolve((p->blockNo- 1) % FLIGHT_NUM);
		curr_source->interpolateConvolve((p->blockNo - 1) % FLIGHT_NUM);
		/*Return*/
		checkCudaErrors(cudaMemcpyAsync(
			curr_source->intermediate[(p->blockNo - 2) % FLIGHT_NUM],
			curr_source->d_output[(p->blockNo - 2) % FLIGHT_NUM] + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
			FRAMES_PER_BUFFER * 2 * sizeof(float),
			cudaMemcpyDeviceToHost,
			curr_source->streams[(p->blockNo - 2) % FLIGHT_NUM * STREAMS_PER_FLIGHT])
		);
		/*Overlap-save*/
		memmove(
			curr_source->x[(p->blockNo + 1) % FLIGHT_NUM],
			curr_source->x[p->blockNo % FLIGHT_NUM] + FRAMES_PER_BUFFER,
			sizeof(float) * (PAD_LEN - FRAMES_PER_BUFFER)
		);
		
	}
	p->blockNo++;
	if (p->blockNo > FLIGHT_NUM * 2) {
		p->blockNo -= FLIGHT_NUM;
	}
	sf_writef_float(p->sndfile, output, FRAMES_PER_BUFFER);
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


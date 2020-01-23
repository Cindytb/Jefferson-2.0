#include "Audio.cuh"

#include <cuda.h>
PaStream* stream;
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
	outputParams.suggestedLatency = float(FRAMES_PER_BUFFER) / float(fs);
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
#if DEBUGMODE == 0 || DEBUGMODE == 2
	PaError err;
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
void callback_func(float* output, Data* p, bool write) {
	for (int i = 0; i < FRAMES_PER_BUFFER * 2; i++) {
		output[i] = 0.0f;
	}
	for (int source_no = 0; source_no < p->num_sources; source_no++) {

		/*Enable pausing of audio*/
		if (p->pauseStatus == true)
			break;

		if (p->type == processes::GPU_FD_COMPLEX || p->type == processes::GPU_FD_BASIC || p->type == processes::GPU_TD) {
			GPUSoundSource* source = &(p->all_sources[source_no]);
			if (p->type == processes::GPU_FD_COMPLEX) {
				checkCudaErrors(cudaStreamSynchronize(source->streams[0]));

				for (int i = 0; i < FRAMES_PER_BUFFER * 2; i++) {
					output[i] += source->intermediate[i];
					if (output[i] > 1.0) {
						fprintf(stderr, "ALERT! CLIPPING AUDIO!\n");
					}
				}
				source->process(p->type);
			}
		}
		else {
			CPUSoundSource* source = (CPUSoundSource*)&(p->all_sources[source_no]);
			/*Copy into curr_source->x pinned memory*/
			if (source->count + FRAMES_PER_BUFFER < source->length) {
				memcpy(
					source->x + (PAD_LEN - FRAMES_PER_BUFFER),  /*Go to the end and work backwards*/
					source->buf + source->count,
					FRAMES_PER_BUFFER * sizeof(float));
				source->count += FRAMES_PER_BUFFER;
			}
			else {
				int rem = source->length - source->count;
				memcpy(
					source->x + (PAD_LEN - FRAMES_PER_BUFFER),
					source->buf + source->count,
					rem * sizeof(float));
				memcpy(
					source->x + (PAD_LEN - FRAMES_PER_BUFFER) + rem,
					source->buf,
					(FRAMES_PER_BUFFER - (size_t)rem) * sizeof(float));
				source->count = FRAMES_PER_BUFFER - rem;
			}

			/*Process*/
			source->process(p->type);

			/*Write to output*/
			for (int i = 0; i < FRAMES_PER_BUFFER * 2; i++) {
				output[i] += ((float*)source->intermediate)[i + 2 * (PAD_LEN - FRAMES_PER_BUFFER)];
				if (output[i] > 1.0) {
					fprintf(stderr, "ALERT! CLIPPING AUDIO!\n");
				}
			}

			/*Overlap-save*/
			memmove(
				source->x,
				source->x + FRAMES_PER_BUFFER,
				sizeof(float) * (PAD_LEN - FRAMES_PER_BUFFER)
			);
		}
	}
	if(write)
		sf_writef_float(p->sndfile, output, FRAMES_PER_BUFFER);
	return;
}
static int paCallback(const void* inputBuffer, void* outputBuffer,
	unsigned long framesPerBuffer,
	const PaStreamCallbackTimeInfo* timeInfo,
	PaStreamCallbackFlags statusFlags,
	void* userData)
{
	/* Cast data passed through stream to our structure. */
	Data* p = (Data*)userData;
	float* output = (float*)outputBuffer;
	callback_func(output, p, true);
	return 0;
}


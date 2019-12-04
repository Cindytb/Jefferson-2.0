#include "Audio.cuh"
#include <cuda.h>
PaStream *stream;
extern Data data;

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
	for(int i = 0; i < FRAMES_PER_BUFFER * 2; i++){
		output[i] = 0.0f;
	}
	for (int source_no = 0; source_no < p->num_sources; source_no++){
		/*Enable pausing of audio*/
		if (p->pauseStatus == true) {
			for (int i = 0; i < FRAMES_PER_BUFFER; i++) {
				output[2 * i] = 0;
				output[2 * i + 1] = 0;
			}
			break;
		}
		
		//SoundSource curr_source = p->all_sources[source_no];
		//fprintf(stderr, "%s\n", cudaStreamQuery(p->all_sources[source_no].streams[p->blockNo % FLIGHT_NUM * 2]) ? "Unfinished" : "Finished");
		checkCudaErrors(cudaStreamSynchronize(p->all_sources[source_no].streams[p->blockNo % FLIGHT_NUM * 2]));
		checkCudaErrors(cudaStreamSynchronize(p->all_sources[source_no].streams[p->blockNo % FLIGHT_NUM * 2 + 1]));
		/*Copy into p->all_sources[source_no].x pinned memory*/
		if (p->all_sources[source_no].count + FRAMES_PER_BUFFER < p->all_sources[source_no].length) {
			memcpy(p->all_sources[source_no].x[p->blockNo % FLIGHT_NUM] + HRTF_LEN - 1, p->all_sources[source_no].buf + p->all_sources[source_no].count, FRAMES_PER_BUFFER * sizeof(float));
			p->all_sources[source_no].count += FRAMES_PER_BUFFER;
		}
		else {
			int rem = p->all_sources[source_no].length - p->all_sources[source_no].count;
			memcpy(p->all_sources[source_no].x[p->blockNo % FLIGHT_NUM] + HRTF_LEN - 1, p->all_sources[source_no].buf + p->all_sources[source_no].count, rem * sizeof(float));
			memcpy(p->all_sources[source_no].x[p->blockNo % FLIGHT_NUM] + HRTF_LEN - 1 + rem, p->all_sources[source_no].buf, (FRAMES_PER_BUFFER - rem) * sizeof(float));
			p->all_sources[source_no].count = FRAMES_PER_BUFFER - rem;
		}

		for (int i = 0; i < FRAMES_PER_BUFFER * 2; i++) {
			output[i] += p->all_sources[source_no].intermediate[p->blockNo % FLIGHT_NUM][i];
		}
		/*Send*/
		checkCudaErrors(cudaMemcpyAsync(
			p->all_sources[source_no].d_input[p->blockNo % FLIGHT_NUM],
			p->all_sources[source_no].x[p->blockNo % FLIGHT_NUM],
			COPY_AMT * sizeof(float),
			cudaMemcpyHostToDevice,
			p->all_sources[source_no].streams[(p->blockNo) % FLIGHT_NUM * 2])
		);
		/*Process*/
		GPUconvolve_hrtf(
			p->all_sources[source_no].d_input[(p->blockNo - 1) % FLIGHT_NUM] + HRTF_LEN,
			p->all_sources[source_no].hrtf_idx,
			p->all_sources[source_no].d_output[(p->blockNo - 1) % FLIGHT_NUM],
			FRAMES_PER_BUFFER,
			p->all_sources[source_no].gain,
			p->all_sources[source_no].streams + (p->blockNo - 1) % FLIGHT_NUM * 2
		);
		/*Return*/
		checkCudaErrors(cudaMemcpyAsync(
			p->all_sources[source_no].intermediate[(p->blockNo - 2) % FLIGHT_NUM],
			p->all_sources[source_no].d_output[(p->blockNo - 2) % FLIGHT_NUM],
			FRAMES_PER_BUFFER * 2 * sizeof(float),
			cudaMemcpyDeviceToHost,
			p->all_sources[source_no].streams[(p->blockNo - 2) % FLIGHT_NUM * 2])
		);
		/*Overlap-save*/
		memcpy(p->all_sources[source_no].x[(p->blockNo + 1) % FLIGHT_NUM], p->all_sources[source_no].x[p->blockNo % FLIGHT_NUM] + FRAMES_PER_BUFFER, (HRTF_LEN - 1) * sizeof(float));
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


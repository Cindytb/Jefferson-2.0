#include "main.cuh"

int main(int argc, char *argv[]){
	if (argc < 1) {
		printf("Usage: %s input.wav reverb.wav", argv[0]);
		return 0;
	}
	/*Initialize data structure*/
	Data data, *p = &data;
	data.count = 0;
	data.length = 0;
	data.gain = 0.99074;
	data.hrtf_idx = -151;
	#if(DEBUGMODE != 1)
		/*Initialize & read files*/
		cudaFFT(argc, argv, p);
		SNDFILE *test;
		SF_INFO test1;
		test = sf_open("scrap.wav", SFM_READ, &test1);
	
	////////////////////////////////////////////////////////////////////////////////
	///*NOTE: GPU Convolution was not fast enough because of the large overhead
	//of FFT and IFFT. Keeping the code here for future purposes*/
	//
	//checkCudaErrors(cudaMalloc((void**)&p->d_x, HRTF_LEN - 1 + FRAMES_PER_BUFFER));
	////////////////////////////////////////////////////////////////////////////////
	/*Open & read hrtf files*/
	if (read_hrtf_signals() != 0) {
		exit(EXIT_FAILURE);
	}
	p->hrtf_idx = 0;
	for (int i = 0; i < sizeof(p->x) / sizeof(*p->x); i++) {
		p->x[i] = 0.0;
	}
	p->osfinfo.channels = 2;
	p->osfinfo.samplerate = 44100;
	p->osfinfo.format = test1.format;
	p->sndfile = sf_open("ofile.wav", SFM_WRITE, &p->osfinfo);
	sf_close(test);
	#endif

	PaStream *stream = NULL;
	PaError err = NULL;
	

#if(DEBUGMODE != 1)
	PaStreamParameters outputParams;
	PaStreamParameters inputParams;
	
	/* Initializing PortAudio */
	err = Pa_Initialize();
	if (err != paNoError) {
		printf("PortAudio error: %s\n", Pa_GetErrorText(err));
		printf("\nExiting.\n");
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
		SAMPLE_RATE, FRAMES_PER_BUFFER,
		paNoFlag, /* flags */
		paCallback,
		&data);

	if (err != paNoError) {
		printf("PortAudio error: open stream: %s\n", Pa_GetErrorText(err));
		printf("\nExiting.\n");
		exit(1);
	}

	/* Start audio stream */
	err = Pa_StartStream(stream);
	if (err != paNoError) {
		printf("PortAudio error: start stream: %s\n", Pa_GetErrorText(err));
		printf("\nExiting.\n");
		exit(1);
	}

	printf("Starting playout\n");

#endif
	///////////////////////////////////////////////////////////////////////////////////////////////////
	/*MAIN FUNCTIONAL LOOP*/
	/*Here to debug without graphics*/
#if DEBUGMODE == 2
	char merp = getchar();
#else
	graphicsMain(argc, argv, p, err, stream);
#endif
	
	///////////////////////////////////////////////////////////////////////////////////////////////////
	
	/*THIS SECTION WILL NOT RUN IF GRAPHICS IS TURNED ON*/
	/*Placed here to properly close files when debugging without graphics*/
	
	/*Close output file*/
	sf_close(p->sndfile);

	/* Stop stream */
	err = Pa_StopStream(stream);
	if (err != paNoError) {
		printf("PortAudio error: stop stream: %s\n", Pa_GetErrorText(err));
		printf("\nExiting.\n");
		exit(1);
	}

	/* Close stream */
	err = Pa_CloseStream(stream);
	if (err != paNoError) {
		printf("PortAudio error: close stream: %s\n", Pa_GetErrorText(err));
		printf("\nExiting.\n");
		exit(1);
	}

	/* Terminate PortAudio */
	err = Pa_Terminate();
	if (err != paNoError) {
		printf("PortAudio error: terminate: %s\n", Pa_GetErrorText(err));
		printf("\nExiting.\n");
		exit(1);
	}

	return 0;
}

/* This routine will be called by the PortAudio engine when audio is needed.
* It may called at interrupt level on some machines so don't do anything
* in the routine that requires significant resources.
*/
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
	int i;
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

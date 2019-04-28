#include "Audio.cuh"

snd_pcm_t *handle;
std::string device = "plughw:0,0";                     /* playback device */
static snd_pcm_format_t format = SND_PCM_FORMAT_FLOAT;    /* sample format */
static unsigned int rate = 48000;                       /* stream rate */
static unsigned int channels = 2;                       /* count of channels */
// static unsigned int period_time = 100000;               
static unsigned int period_time =  1000000.0 * 512.0 / 48000.0; /* period time in us */
static unsigned int buffer_time = period_time * channels;               /* ring buffer length in us */
static int verbose = 1;                                 /* verbose flag */
static int resample = 1;                                /* enable alsa-lib resampling */
static int period_event = 0;                            /* produce poll event after each period */
static snd_pcm_sframes_t buffer_size;
static snd_pcm_sframes_t period_size;
static snd_output_t *output = NULL;
float *samples;
_snd_pcm_state state ;

static int set_hwparams(snd_pcm_t *handle,
                        snd_pcm_hw_params_t *params,
                        snd_pcm_access_t access)
{
        unsigned int rrate;
        snd_pcm_uframes_t size;
        int err, dir;
        /* choose all parameters */
        err = snd_pcm_hw_params_any(handle, params);
        if (err < 0) {
                printf("Broken configuration for playback: no configurations available: %s\n", snd_strerror(err));
                return err;
        }
        /* set hardware resampling */
        err = snd_pcm_hw_params_set_rate_resample(handle, params, resample);
        if (err < 0) {
                printf("Resampling setup failed for playback: %s\n", snd_strerror(err));
                return err;
        }
        /* set the interleaved read/write format */
        err = snd_pcm_hw_params_set_access(handle, params, access);
        if (err < 0) {
                printf("Access type not available for playback: %s\n", snd_strerror(err));
                return err;
        }
        /* set the sample format */
        err = snd_pcm_hw_params_set_format(handle, params, format);
        if (err < 0) {
                printf("Sample format not available for playback: %s\n", snd_strerror(err));
                return err;
        }
        /* set the count of channels */
        err = snd_pcm_hw_params_set_channels(handle, params, channels);
        if (err < 0) {
                printf("Channels count (%i) not available for playbacks: %s\n", channels, snd_strerror(err));
                return err;
        }
        /* set the stream rate */
        rrate = rate;
        err = snd_pcm_hw_params_set_rate_near(handle, params, &rrate, 0);
        if (err < 0) {
                printf("Rate %iHz not available for playback: %s\n", rate, snd_strerror(err));
                return err;
        }
        if (rrate != rate) {
                printf("Rate doesn't match (requested %iHz, get %iHz)\n", rate, err);
                return -EINVAL;
        }
        /* set the buffer time */
        err = snd_pcm_hw_params_set_buffer_time_near(handle, params, &buffer_time, &dir);
        if (err < 0) {
                printf("Unable to set buffer time %i for playback: %s\n", buffer_time, snd_strerror(err));
                return err;
        }
        err = snd_pcm_hw_params_get_buffer_size(params, &size);
        if (err < 0) {
                printf("Unable to get buffer size for playback: %s\n", snd_strerror(err));
                return err;
        }
        buffer_size = size;
        /* set the period time */
        err = snd_pcm_hw_params_set_period_time_near(handle, params, &period_time, &dir);
        if (err < 0) {
                printf("Unable to set period time %i for playback: %s\n", period_time, snd_strerror(err));
                return err;
        }
        err = snd_pcm_hw_params_get_period_size(params, &size, &dir);
        if (err < 0) {
                printf("Unable to get period size for playback: %s\n", snd_strerror(err));
                return err;
        }
        period_size = size;
        /* write the parameters to device */
        err = snd_pcm_hw_params(handle, params);
        if (err < 0) {
                printf("Unable to set hw params for playback: %s\n", snd_strerror(err));
                return err;
        }
        return 0;
}
static int set_swparams(snd_pcm_t *handle, snd_pcm_sw_params_t *swparams)
{
        int err;
        /* get the current swparams */
        err = snd_pcm_sw_params_current(handle, swparams);
        if (err < 0) {
                printf("Unable to determine current swparams for playback: %s\n", snd_strerror(err));
                return err;
        }
        /* start the transfer when the buffer is almost full: */
        /* (buffer_size / avail_min) * avail_min */
        err = snd_pcm_sw_params_set_start_threshold(handle, swparams, (buffer_size / period_size) * period_size);
        if (err < 0) {
                printf("Unable to set start threshold mode for playback: %s\n", snd_strerror(err));
                return err;
        }
        /* allow the transfer when at least period_size samples can be processed */
        /* or disable this mechanism when period event is enabled (aka interrupt like style processing) */
        err = snd_pcm_sw_params_set_avail_min(handle, swparams, period_event ? buffer_size : period_size);
        if (err < 0) {
                printf("Unable to set avail min for playback: %s\n", snd_strerror(err));
                return err;
        }
        /* enable period events when requested */
        if (period_event) {
                err = snd_pcm_sw_params_set_period_event(handle, swparams, 1);
                if (err < 0) {
                        printf("Unable to set period event: %s\n", snd_strerror(err));
                        return err;
                }
        }
        /* write the parameters to the playback device */
        err = snd_pcm_sw_params(handle, swparams);
        if (err < 0) {
                printf("Unable to set sw params for playback: %s\n", snd_strerror(err));
                return err;
        }
        return 0;
}

void callback_func(float *output, Data *p){
	/*CPU/RAM Copy data loop*/
	for (int i = 0; i < period_size; i++) {
		p->x[HRTF_LEN - 1 + i] = p->buf[p->count];
		p->count++;
		if (p->count == p->length) {
			p->count = 0;
		}
	}
	/*convolve with HRTF on CPU*/
	convolve_hrtf(&p->x[HRTF_LEN], p->hrtf_idx, output, period_size, p->gain);

	/*Enable pausing of audio*/
	if (p->pauseStatus == true) {
		for (int i = 0; i < period_size; i++) {
			output[2 * i] = 0;
			output[2 * i + 1] = 0;
		}
	}
	


	/* copy last HRTF_LEN-1 samples of x data to "history" part of x for use next time */
	float *px = p->x;
	for (int i = 0; i<HRTF_LEN - 1; i++) {
		px[i] = px[period_size + i];
	}
}
static void my_callback(snd_async_handler_t *ahandler)
{
	/* Cast data passed through stream to our structure. */
	Data *p = (Data*) snd_async_handler_get_callback_private(ahandler);
	float *output = p->samples;
	int err;
	snd_pcm_sframes_t avail;
	avail = snd_pcm_avail_update(handle);
	if (avail == -EPIPE){
		printf("WARNING: Buffer Underrun\n");
		avail = snd_pcm_recover(handle, avail, 0);
		if (avail == -EPIPE){
			printf("ERROR: UNRECOVERABLE\n");
		}
	}
    while (avail >= period_size) {
		callback_func(output, p);
		err = snd_pcm_writei(handle, output, period_size);
		if (err < 0) {
			printf("Write error: %s\n", snd_strerror(err));
			exit(EXIT_FAILURE);
		}

		if (err != period_size) {
			printf("Write error: written %i expected %li\n", err, period_size);
			exit(EXIT_FAILURE);
		}
		if (err == -EPIPE){
			printf("WARNING: Buffer Underrun\n");
			err = snd_pcm_recover(handle, err, 0);
			if (err == -EPIPE){
				printf("ERROR: UNRECOVERABLE\n");
			}
		}
		avail = snd_pcm_avail_update(handle);
	}
	//sf_writef_float(p->sndfile, output, framesPerBuffer);
	return;
        
}

void initializeAlsa(int fs, Data*p){
	
	int err;
	snd_pcm_hw_params_t *hwparams;
	snd_pcm_sw_params_t *swparams;
	float *samples;

	snd_pcm_hw_params_alloca(&hwparams);
	snd_pcm_sw_params_alloca(&swparams);

	err = snd_output_stdio_attach(&output, stdout, 0);
	if (err < 0) {
			printf("Output failed: %s\n", snd_strerror(err));
			exit(EXIT_FAILURE);;
	}
	printf("Playback device is %s\n", device.c_str());
	printf("Stream parameters are %iHz, %s, %i channels\n", rate, snd_pcm_format_name(format), channels);
	if ((err = snd_pcm_open(&handle, device.c_str(), SND_PCM_STREAM_PLAYBACK, 0)) < 0) {
		printf("Playback open error: %s\n", snd_strerror(err));
		exit(EXIT_FAILURE);;
	}
	
	if ((err = set_hwparams(handle, hwparams, SND_PCM_ACCESS_RW_INTERLEAVED) < 0)) {
		printf("Setting of hwparams failed: %s\n", snd_strerror(err));
		exit(EXIT_FAILURE);
	}
	if ((err = set_swparams(handle, swparams)) < 0) {
			printf("Setting of swparams failed: %s\n", snd_strerror(err));
			exit(EXIT_FAILURE);
	}
	if (verbose > 0)
		snd_pcm_dump(handle, output);
	samples = new float[period_size * channels];

	for(int i = 0; i<period_size * channels; i++){
		samples[i] = 0.0f;
	}
	snd_pcm_dump(handle, output);
	p->samples = samples;
    snd_async_handler_t *ahandler;
	err = snd_async_add_pcm_handler(&ahandler, handle, my_callback, &data);
	if (err < 0) {
		printf("Unable to register async handler\n");
		exit(EXIT_FAILURE);
	}
	for (int count = 0; count < 2; count++) {
		callback_func(samples, p);
		err = snd_pcm_writei(handle, samples, period_size);
		if (err < 0) {
			printf("Write error: %s\n", snd_strerror(err));
			exit(EXIT_FAILURE);
		}
		if (err != period_size) {
			printf("Write error: written %i expected %li\n", err, period_size);
			exit(EXIT_FAILURE);
		}
		if (err == -EPIPE){
			printf("WARNING: Buffer Underrun\n");
			err = snd_pcm_recover(handle, err, 0);
			if (err == -EPIPE){
				printf("ERROR: UNRECOVERABLE\n");
			}
		}
	}
	state = snd_pcm_state(handle);
	if (state == SND_PCM_STATE_PREPARED) {
		err = snd_pcm_start(handle);
		if (err < 0) {
			printf("Start error: %s\n", snd_strerror(err));
			exit(EXIT_FAILURE);
		}
	}
	// for(int i = 0; i < 1000000000; i++){
	// 	state = snd_pcm_state(handle);
	// 	std::cout << state << std::endl;
	// }
}

void closeAlsa(){
	int err;
#if DEBUG != 1
	delete[] samples;
	err = snd_pcm_close(handle);
	if (err < 0) {
		printf("Close error: %s\n", snd_strerror(err));
		exit(EXIT_FAILURE);
	}
#endif

}
#if 0
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

#endif
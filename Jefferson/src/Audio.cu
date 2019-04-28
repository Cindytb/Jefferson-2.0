#include "Audio.cuh"
#include <cuda.h>

snd_pcm_t *handle;
std::string device = "plughw:0,0";                     /* playback device */
static snd_pcm_format_t format = SND_PCM_FORMAT_FLOAT;    /* sample format */
static unsigned int rate = 44100;                       /* stream rate */
static unsigned int channels = 2;                       /* count of channels */
// static unsigned int period_time = 100000;               
static unsigned int period_time =  1000000.0 * 512.0 / 44100.0; /* period time in us */
static unsigned int buffer_time = period_time * channels;               /* ring buffer length in us */
static int verbose = 0;                                 /* verbose flag */
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
	// printf("%i\n", p->count);
	// for(int i = 0; i < 8; i++){
	// 	fprintf(stderr, "Stream No:%i - %s\n", i, cudaStreamQuery(p->streams[i]) ? "Not Finished" : "Finished");
	// }
	// fprintf(stderr, "\n");
	checkCudaErrors(cudaStreamSynchronize(p->streams[(p->blockNo - 2) % 5 * 2]));
	/*Copy into p->x pinned memory*/
	if (p->count + period_size < p->length){
		memcpy(p->x + HRTF_LEN - 1, p->buf + p->count, period_size * sizeof(float));
		p->count += period_size;
	}
	else{
		int rem = p->length - p->count;
		memcpy(p->x + HRTF_LEN - 1, p->buf + p->count, rem * sizeof(float));
		memcpy(p->x + HRTF_LEN - 1 + rem, p->buf, (period_size - rem) * sizeof(float));
		p->count = period_size - rem;
	}

	// if (p->blockNo == 5) {
	// 	printf("%i\n", cuCtxPushCurrent(0));
	// }
	// fprintf(stderr, "Stream %i %s\n", p->blockNo % 5, cudaStreamQuery(p->streams[p->blockNo % 5 * 2]) ? "Unfinished":"Finished");
	/*Enable pausing of audio*/
	if (p->pauseStatus == true) {
		for (int i = 0; i < period_size; i++) {
			output[2 * i] = 0;
			output[2 * i + 1] = 0;
		}
		return;
	}
	memcpy(output, p->intermediate, period_size * 2 * sizeof(float));
	// fprintf(stderr, "%i %i %i %i %i\n", p->blockNo % 5, (p->blockNo - 1) % 5, (p->blockNo - 2) % 5, (p->blockNo - 3) % 5, (p->blockNo - 4) % 5);
	/*Send*/
	checkCudaErrors(cudaMemcpyAsync(p->d_input[p->blockNo % 5], p->x, COPY_AMT * sizeof(float), cudaMemcpyHostToDevice, p->streams[(p->blockNo) % 5 * 2]));
	/*Process*/
	GPUconvolve_hrtf(p->d_input[(p->blockNo - 1) % 5] + HRTF_LEN, p->hrtf_idx, p->d_output[(p->blockNo - 1) % 5], FRAMES_PER_BUFFER, p->gain, &(p->streams[(p->blockNo - 1) % 5 * 2]));
	/*Idle blockNo - 2*/
	/*Idle blockNo - 3*/
	/*Return & fill intermediate*/
	checkCudaErrors(cudaMemcpyAsync(p->intermediate, p->d_output[(p->blockNo - 4) % 5], period_size * 2 * sizeof(float), cudaMemcpyDeviceToHost, p->streams[(p->blockNo - 3) % 5 * 2]));

	
	/*Overlap-save*/
	memcpy(p->x, p->x + period_size, (HRTF_LEN - 1) * sizeof(float));
	p->blockNo++;

	//sf_writef_float(p->sndfile, output, framesPerBuffer);
	return;
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
		if (avail == -EPIPE){
			printf("WARNING: Buffer Underrun\n");
			avail = snd_pcm_recover(handle, avail, 0);
			if (avail == -EPIPE){
				printf("ERROR: UNRECOVERABLE\n");
			}
		}
	}
	sf_writef_float(p->sndfile, output, period_size);
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
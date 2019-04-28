#ifndef _UNIVERSAL_H_
#define _UNIVERSAL_H_


/*Audio Includes*/
#include <sndfile.h>
#include <sndfile.hh>
#include <portaudio.h>

#define HRTF_LEN	128
#define FRAMES_PER_BUFFER 512  //buffer in portaudio i/o buffer
#define HRTF_CHN    2
const int COPY_AMT = 2 * (FRAMES_PER_BUFFER + HRTF_LEN - 1);
// #define COPY_AMT  (2 * (FRAMES_PER_BUFFER + HRTF_LEN - 1));

//Value of 0 allows everything
//Value of 1 is graphics-only debugging
//Value of 2 is audio-only debugging
#define DEBUGMODE 2

struct Data_tag {
	float *samples; 		/*IO Buffer for ALSA. Unnecessary for PortAudio*/
	int hrtf_idx; 			/*Index to the correct HRTF elevation/azimuth*/
	SNDFILE *sndfile;		/*Soundfile object for output file*/
	float *buf;				/*Reverberated signal on host*/
	/*Buffer for PA output on host*/
	float *x; /* sound object buffer, mono */
	int count;
	int length;
	float gain;
	float ele;
	bool pauseStatus = false;
	////////////////////////////////////////////////////////////////////////////////
	///*NOTE: GPU Convolution was not fast enough because of the large overhead
	//of FFT and IFFT. Keeping the code here for future purposes*/
	/*2019 version*/
	float *intermediate;	/*Host data of the output*/
	float *d_input[5];		/*FRAMES_PER_BUFFER + HRTF_LEN - 1 sized for the input*/
	float *d_output[5];		/*FRAMES_PER_BUFFER * 2 sized for the output*/
	int blockNo = 0;
	cudaStream_t *streams;
	////////////////////////////////////////////////////////////////////////////////
};
typedef struct Data_tag Data;


const float ratio = 1 / (float)44100;
#endif
#ifndef _UNIVERSAL_H_
#define _UNIVERSAL_H_


/*Audio Includes*/
#include <sndfile.h>
#include <sndfile.hh>
#include <portaudio.h>

#define HRTF_LEN	128
#define FRAMES_PER_BUFFER 8192  //buffer in portaudio i/o buffer
#define HRTF_CHN    2

//Value of 0 allows everything
//Value of 1 is graphics-only debugging
//Value of 2 is audio-only debugging
#define DEBUGMODE 0

struct Data_tag {
	float *samples;
	int hrtf_idx;
	SNDFILE *sndfile;
	SF_INFO osfinfo;
	/*Reverberated signal on host*/
	float *buf;
	/*Buffer for PA output on host*/
	float x[HRTF_LEN - 1 + FRAMES_PER_BUFFER]; /* sound object buffer, mono */
	int count;
	int length;
	float gain;
	float ele;
	bool pauseStatus = false;
	////////////////////////////////////////////////////////////////////////////////
	///*NOTE: GPU Convolution was not fast enough because of the large overhead
	//of FFT and IFFT. Keeping the code here for future purposes*/
	//
	///*Convolved signal on device*/
	//float *dbuf;
	///*Buffer for PA output on device*/
	//float *d_x;
	////////////////////////////////////////////////////////////////////////////////
};
typedef struct Data_tag Data;


const float ratio = 1 / (float)44100;
#endif
#ifndef _UNIVERSAL_H_
#define _UNIVERSAL_H_


/*Audio Includes*/
#include <sndfile.h>
#include <sndfile.hh>
#include <portaudio.h>

#define HRTF_LEN	128
#define FRAMES_PER_BUFFER 512  //buffer in portaudio i/o buffer
#define HRTF_CHN    2

struct Data_tag {
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
#endif
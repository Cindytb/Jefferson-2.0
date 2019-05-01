#ifndef _UNIVERSAL_H_
#define _UNIVERSAL_H_


/*Audio Includes*/
#include <sndfile.h>
#include <sndfile.hh>
#include <portaudio.h>

const int HRTF_LEN = 128;
const int FRAMES_PER_BUFFER = 2048;  //buffer in portaudio i/o buffer
const int HRTF_CHN = 2;
const int COPY_AMT = 2 * (FRAMES_PER_BUFFER + HRTF_LEN - 1);
const int FLIGHT_NUM = 5;

/* 
	0 - Run everything, no debugging
 	1 - graphics-only
	2 - audio-only
*/
#define DEBUGMODE 0

struct Data_tag {
	float *samples; 				/*IO Buffer for ALSA. Unnecessary for PortAudio*/
	SNDFILE *sndfile;				/*Soundfile object for output file*/
	float *buf;						/*Reverberated signal on host*/
	float *x[FLIGHT_NUM];			/*Buffer for PA output on host, mono, pinned memory, FRAMES_PER_BUFFER + HRTF_LEN -1 size */
	float *intermediate[FLIGHT_NUM];/*Host data of the output*/
	float *d_input[FLIGHT_NUM];		/*FRAMES_PER_BUFFER + HRTF_LEN - 1 sized for the input*/
	float *d_output[FLIGHT_NUM];	/*FRAMES_PER_BUFFER * 2 sized for the output*/
	cudaStream_t *streams;			/*Streams associated with each block in flight*/
	/*TODO: Fix blockNo to prevent overflow. Doing it the hacky way for now*/
	unsigned long long blockNo = 0; /*Block number for number of blocks in flight for GPU processing*/
	int count;						/*Current frame count for the audio callback*/
	int hrtf_idx; 					/*Index to the correct HRTF elevation/azimuth*/
	int length;						/*Length of the input signal*/
	float gain;						/*Gain for distance away*/
	float ele;						/*Elevation of the sound source*/

	bool pauseStatus = false;
};
typedef struct Data_tag Data;
void closeEverything();

const float ratio = 1 / (float)44100;
#endif
#ifndef _UNIVERSAL_H_
#define _UNIVERSAL_H_

#define NUM_HRTF	710
/*Audio Includes*/
#include <sndfile.h>
#include <sndfile.hh>
#include <portaudio.h>
extern float* d_hrtf;
const int HRTF_LEN = 512;
const int FRAMES_PER_BUFFER = 512;  //buffer in portaudio i/o buffer
const int HRTF_CHN = 2;
const int PAD_LEN = (int) pow(2, ceil(log2(FRAMES_PER_BUFFER + HRTF_LEN - 1I)));
const int FLIGHT_NUM = 4;
#ifndef PI
#define PI 3.14159265358979323846264338327950288
#endif
/* 
	0 - Run everything, no debugging
 	1 - graphics-only
	2 - audio-only
*/
#define DEBUGMODE 2
void closeEverything();

const float ratio = 1 / (float)44100;
#endif
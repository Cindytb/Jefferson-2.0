#ifndef _UNIVERSAL_H_
#define _UNIVERSAL_H_

#define NUM_HRTF	710
/*Audio Includes*/
#include <sndfile.h>
#include <sndfile.hh>
#include <portaudio.h>
const int HRTF_LEN = 512;
const int FRAMES_PER_BUFFER = 128;  //buffer in portaudio i/o buffer
const int HRTF_CHN = 2;
const int PAD_LEN = (int) pow(2, ceil(log2(FRAMES_PER_BUFFER + HRTF_LEN - 1)));
const int STREAMS_PER_FLIGHT = 8;
#ifndef PI
#define PI 3.14159265358979323846264338327950288
#endif
/* 
	0 - Run everything, no debugging
 	1 - graphics-only
	2 - audio-only
	3 - audio functions, no realtime portaudio and manual calls to the callback function
*/
#define DEBUGMODE 3
void closeEverything();
enum processes {
	GPU_FD_COMPLEX,
	GPU_FD_BASIC,
	GPU_TD,
	CPU_FD_COMPLEX,
	CPU_FD_BASIC,
	CPU_TD
};

const float ratio = 1 / (float)44100;
#endif
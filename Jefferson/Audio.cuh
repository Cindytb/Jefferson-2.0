#pragma once
#ifndef _AUDIO_H_
#define _AUDIO_H_

#include "Universal.cuh"
#include "hrtf_signals.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/*Portaudio defines*/
#define FRAMES_PER_BUFFER 512  //buffer in portaudio i/o buffer


/*PortAudio stuff*/

extern Data data;


static int paCallback(const void *inputBuffer, void *outputBuffer,
	unsigned long framesPerBuffer,
	const PaStreamCallbackTimeInfo* timeInfo,
	PaStreamCallbackFlags statusFlags,
	void *userData);

void initializePA(int fs);
void closePA();


#endif
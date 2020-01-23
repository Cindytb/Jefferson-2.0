#pragma once
#ifndef _AUDIO_H_
#define _AUDIO_H_

#include "Universal.cuh"
#include "hrtf_signals.cuh"
#include "main.cuh"
#include "DataTag.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



/*PortAudio stuff*/
static int paCallback(const void *inputBuffer, void *outputBuffer,
	unsigned long framesPerBuffer,
	const PaStreamCallbackTimeInfo* timeInfo,
	PaStreamCallbackFlags statusFlags,
	void *userData);
void callback_func(float* output, Data* p, bool write);
void initializePA(int fs);
void closePA();


#endif
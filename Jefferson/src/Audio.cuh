#pragma once
#ifndef _AUDIO_H_
#define _AUDIO_H_


#include "Universal.cuh"
#include "hrtf_signals.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <alsa/asoundlib.h>
#include <alsa/pcm.h>


/*PortAudio stuff*/

extern Data data;

void initializeAlsa(int fs, Data *p);
void closeAlsa();

static int paCallback(const void *inputBuffer, void *outputBuffer,
	unsigned long framesPerBuffer,
	const PaStreamCallbackTimeInfo* timeInfo,
	PaStreamCallbackFlags statusFlags,
	void *userData);

// void initializePA(int fs);
// void closePA();


#endif
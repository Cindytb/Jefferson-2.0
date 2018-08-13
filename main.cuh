#pragma once
#ifndef _MAIN_H
#define _MAIN_H

#include "Universal.cuh"
#include "cudaPart.cuh"
#include "graphics.cuh"
#include "hrtf_signals.cuh"

#define SAMPLE_RATE 44100

/* Callback function protoype */
static int paCallback(const void *inputBuffer, void *outputBuffer,
	unsigned long framesPerBuffer,
	const PaStreamCallbackTimeInfo* timeInfo,
	PaStreamCallbackFlags statusFlags,
	void *userData);

//Value of 0 allows everything
//Value of 1 is graphics-only debugging
//Value of 2 is audio-only debugging
#define DEBUGMODE 0
#endif

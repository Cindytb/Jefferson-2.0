#pragma once
#ifndef _MAIN_H
#define _MAIN_H

#include "Universal.cuh"
#include "cudaPart.cuh"
#include "graphics.cuh"
#include "hrtf_signals.cuh"
#include "Audio.cuh"

#define SAMPLE_RATE 44100
/*Initialize data structure*/
Data data, *p = &data;

#endif

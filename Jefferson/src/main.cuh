#pragma once
#ifndef _MAIN_H
#define _MAIN_H

#define HELPERGL_EXTERN_GL_FUNC_IMPLEMENTATION
#include "Universal.cuh"
#include "cudaPart.cuh"
#include "graphics.cuh"
#include "hrtf_signals.cuh"
#include "Audio.cuh"
#include "DataTag.cuh"

#define SAMPLE_RATE 44100
/*Initialize data structure*/
extern Data data, *p;

#endif

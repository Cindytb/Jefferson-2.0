#pragma once
#ifndef __PRECISION_TEST__
#define __PRECISION_TEST__
#include "DataTag.cuh"
#include "Universal.cuh"
#include "hrtf_signals.cuh"
#include "kernels.cuh"
#include "Audio.cuh"
//#include "main.cuh"

//#include <fftw3.h>
void precisionTest(Data* p);
void xfadePrecisionTest(Data* p);
void xfadePrecisionCallbackTest(Data* p);
#endif
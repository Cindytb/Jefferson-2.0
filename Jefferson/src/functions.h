#pragma once
#include <fftw3.h>
#include <cmath>
void pointwiseAddition(fftwf_complex* a, fftwf_complex* b, int size);
void pointwiseAddition(fftwf_complex* a, fftwf_complex* b, fftwf_complex* c, int size);
void pointwiseMultiplication(fftwf_complex* a, fftwf_complex* b, int size);
void pointwiseMultiplication(fftwf_complex* a, fftwf_complex* b, fftwf_complex* c, int size);
void complexScaling(fftwf_complex* f_x, float scale, int size);
int precisionChecking(float* in1, float* in2, size_t size, float epsilon);
int precisionChecking(float* in1, float* in2, size_t size);
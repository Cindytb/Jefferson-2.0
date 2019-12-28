#pragma once
#include <fftw3.h>

void pointwiseAddition(fftwf_complex* a, fftwf_complex* b, int size);
void pointwiseAddition(fftwf_complex* a, fftwf_complex* b, fftwf_complex* c, int size);
void pointwiseMultiplication(fftwf_complex* a, fftwf_complex* b, int size);
void pointwiseMultiplication(fftwf_complex* a, fftwf_complex* b, fftwf_complex* c, int size);
void complexScaling(fftwf_complex* f_x, float scale, int size);
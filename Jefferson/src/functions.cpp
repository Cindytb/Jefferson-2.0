#include "functions.h"

void pointwiseAddition(fftwf_complex* a, fftwf_complex* b, int size) {
#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		a[i][0] += b[i][0];
		a[i][1] += b[i][1];
	}
}
void pointwiseAddition(fftwf_complex* a, fftwf_complex* b, fftwf_complex* c, int size) {
#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		c[i][0] = a[i][0] + b[i][0];
		c[i][1] = a[i][1] + b[i][1];
	}
}
void pointwiseMultiplication(fftwf_complex* a, fftwf_complex* b, int size) {
#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		fftwf_complex temp;
		temp[0] = a[i][0];
		temp[1] = a[i][1];
		a[i][0] = temp[0] * b[i][0] - temp[1] * b[i][1];
		a[i][1] = temp[0] * b[i][1] + temp[1] * b[i][0];
	}
}
void pointwiseMultiplication(fftwf_complex* a, fftwf_complex* b, fftwf_complex *c, int size) {
#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		c[i][0] = a[i][0] * b[i][0] - a[i][1] * b[i][1];
		c[i][1] = a[i][0] * b[i][1] + a[i][1] * b[i][0];
	}
}
void complexScaling(fftwf_complex* f_x, float scale, int size) {
#pragma omp parallel for
	for (int i = 0; i < size; i++) {
		f_x[i][0] *= scale;
		f_x[i][1] *= scale;
	}
}
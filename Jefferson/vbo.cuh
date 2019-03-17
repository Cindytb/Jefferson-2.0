#pragma once
/*OpenGL Includes*/
#include <helper_gl.h>
#include <GL/freeglut.h>

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

/*Thrust*/
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <cmath>

#include "kernels.cuh"
__global__ void averagingKernel(float4 *pos, float *d_buf, unsigned int size, double ratio, int averageSize);
void launch_new_kernel(float4 *pos, float* buf, unsigned int size, int averageNum, float ratio);


class VBO {
public:
	int averageNum;
	float ratio;

	VBO::VBO(float **a, float *b, unsigned int c, float d);
	void init();
	void create();
	void update();
	void VBO::draw(float rotateVBO_x, float rotateVBO_y, float  rotateVBO_z);
	~VBO();
private:
	GLuint vbo;
	struct cudaGraphicsResource *cuda_vbo_resource;
	unsigned int numSamples;
	unsigned int vboSize;
	float **d_buf;
	float *translate_x;
};

class SineVBO : VBO {

};
#pragma once
#ifndef _GRAPHICS_H
#define _GRAPHICS_H
#include <helper_gl.h>
#include "vbo.cuh"
#include "Universal.cuh"
#include "hrtf_signals.cuh"
#include "Audio.cuh"
#include "kernels.cuh"
#include "import_model.h"

#include <glm/glm.hpp>

#include <glm/glm.hpp>



 //includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif
/*Graphics Includes*/
// OpenGL Graphics includes


#if defined (__APPLE__) || defined(MACOSX)
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#include <GLUT/glut.h>
#ifndef glutCloseFunc
#define glutCloseFunc glutWMCloseFunc
#endif
#else
#include <GL/freeglut.h>
#endif


// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h


// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
//#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

#include <vector_types.h>

#define MAX_EPSILON_ERROR 10.0f
#define THRESHOLD          0.30f
#define REFRESH_DELAY     10 //ms

#define MAX(a,b) ((a > b) ? a : b)

////////////////////////////////////////////////////////////////////////////////
// Forward Declarations
////////////////////////////////////////////////////////////////////////////////



int graphicsMain(int argc, char **argv, Data *p);
bool runTest(int argc, char **argv, char *ref_file);
void cleanup();

// GL functionality
bool initGL(int *argc, char **argv);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
	unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void specialKeys(int key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource);

__global__ void simple_vbo_kernel(float4 *pos, unsigned int width, unsigned int height, float time);
void launch_kernel(float4 *pos, unsigned int mesh_width,
	unsigned int mesh_height, float time);

void computeFPS();
bool initGL(int *argc, char **argv);
bool runTest(int argc, char **argv, char *ref_file);
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
	unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);
void display();
void cleanup();
void keyboard(unsigned char key, int /*x*/, int /*y*/);
void specialKeys(int key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
const unsigned int window_width = 1920;
const unsigned int window_height = 1080;

const unsigned int mesh_width = 1024;
const unsigned int mesh_height = 1024;
#ifndef PI
#define PI 3.14159265358979323846264338327950288
#endif

//
//__global__ void myNewKernel(int id, float4 *pos, float *d_buf, unsigned int size, double ratio, int magicNumber);
//void launch_new_kernel(int id, float4 *pos, float* buf, unsigned int size, int magicNumber);
//class VBO {
//public:
//	
//	unsigned int numSamples = 1310720;
//	int id;
//	float **d_buf;
//	float *translate_x;
//
//	VBO::VBO(float **a, float *b);
//	void init();
//	void create(int magicNumber);
//	void VBO::draw(float rotateVBO_y, float rotateVBO_z);
//	~VBO();
//private:
//	GLuint vbo;
//	struct cudaGraphicsResource *cuda_vbo_resource;
//	unsigned int vboSize;
//};


#endif
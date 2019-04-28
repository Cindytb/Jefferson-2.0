#include "vbo.cuh"

VBO::VBO(float **a, float *b, unsigned int c, float d)
	: d_buf(a), translate_x(b), numSamples(c), ratio(d)
{}
void VBO::init() {
	assert(&vbo);

	// create buffer object
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	vboSize = numSamples * sizeof(float4);
	// initialize buffer object
	glBufferData(GL_ARRAY_BUFFER, vboSize, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard));

	SDK_CHECK_ERROR_GL();
}
void VBO::create() {
	// map OpenGL buffer object for writing from CUDA
	float4 *dptr;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
		cuda_vbo_resource));
	printf("CUDA mapped VBO: May access %zu bytes\n", num_bytes);

	fprintf(stderr, "Launching Kernel: number of samples: %d\n", numSamples);
	launch_new_kernel(dptr, *d_buf, numSamples, averageNum, ratio);

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

}
void VBO::update() {
	// map OpenGL buffer object for writing from CUDA
	float4 *dptr;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
		cuda_vbo_resource));

	launch_new_kernel(dptr, *d_buf, numSamples, averageNum, ratio);

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));
}
void VBO::draw(float rotateVBO_x, float rotateVBO_y, float rotateVBO_z) {
	glPushMatrix();

	// render from the vbo
	glRotatef(rotateVBO_x, 0.0, 1.0, 0);
	glRotatef(rotateVBO_y, 0, 0.0, 1.0);
	//glRotatef(rotateVBO_z, 0, 0, 1.0);
	glTranslatef(*translate_x, 0.0, 0.0);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(1.0, 1.0, 1.0);
	//glDrawArrays(GL_POINTS, 0, vboSize);
	glLineWidth(0.005f);
	glEnable(GL_LINE_SMOOTH);
	glDrawArrays(GL_LINES, 0, vboSize);
	glDisableClientState(GL_VERTEX_ARRAY);

	glPopMatrix();
}
VBO::~VBO() {
	// unregister this buffer object with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));

	glBindBuffer(1, vbo);
	glDeleteBuffers(1, &vbo);
	cudaFree(&d_buf);
	fprintf(stderr, "Freed the device audio buffer\n");
}

void launch_new_kernel(float4 *pos, float* buf, unsigned int size, int averageNum, float ratio) {
	unsigned const int numThreads = 1024;
	int numBlocks = size / numThreads + 1;

	int averageSize = averageNum;
	int reducedSize;
	if (averageSize < 1) {
		reducedSize = size;
	}
	else {

		reducedSize = size / averageSize;
	}
	numBlocks = reducedSize / numThreads + 1;
	averagingKernel << < numBlocks, numThreads >> > (pos, buf, size, ratio, averageSize);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
}
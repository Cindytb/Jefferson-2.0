#include "kernels.cuh"

struct is_negative {
	__host__ __device__ bool operator()(float x) {
		if (x < 0) {
			return true;
		}
		return false;
	}
};
template <typename T>
struct square
{
	__host__ __device__
		T operator()(const T& x) const
	{
		return x * x;
	}
};
///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void simple_vbo_kernel(float4 *pos, unsigned int width, unsigned int height, float time)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// calculate uv coordinates
	float u = x / (float)width;
	float v = y / (float)height;
	u = u * 2.0f - 1.0f;
	v = v * 2.0f - 1.0f;

	// calculate simple sine wave pattern
	float freq = 4.0f;
	float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

	// write output vertex
	pos[y*width + x] = make_float4(u, -0.55f, v, 1.0f);
}

// cufftComplex pointwise multiplication
__global__ void ComplexPointwiseMulAndScale(cufftComplex *a, const cufftComplex *b, int size, float scale)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadID; i < size; i += numThreads)
	{
		a[i] = cufftComplexScale(cufftComplexMul(a[i], b[i]), scale);
	}
}
__global__ void ComplexPointwiseMulAndScaleOutPlace(const cufftComplex* a, const cufftComplex* b, cufftComplex* c, int size, float scale)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadID; i < size; i += numThreads)
	{
		c[i] = cufftComplexScale(cufftComplexMul(a[i], b[i]), scale);
	}
}
/*
	R(r) = (1 / (1 + (fs / vs) (r - r0)^2) ) * e^ ((-j2PI (fs/vs) * (r - r0) *k) / N)
			|----------FRAC-----------------|	  |------------exponent--------------|

	FRAC * e^(exponent)
	FRAC * (cosine(exponent) - sine(exponent))
	R[r].x = cosine(exponent) / FRAC
	R[r].y = -sine(exponent) / FRAC
	*/
__global__ void generateDistanceFactor(cufftComplex *in, float frac, float fsvs, float r, int N){
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadID; i < N; i += numThreads)
	{
		in[i].x = cos(2 * PI * fsvs * r * i / N) / frac;
		in[i].y = -sin(2 * PI * fsvs * r * i / N) / frac;
	}
}

/*
	f[n] = n / (N - 1)
	g[n] = 1 - f[n]
*/
__global__ void crossFade(float* out1, float* out2, int numFrames){
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	for (int i = threadID; i < numFrames; i += numThreads)
	{
		out1[i * 2] = out1[i * 2] * float(i) / (numFrames - 1) + out2[i * 2] * (1 - float(i) / (numFrames - 1));
		out1[i * 2 + 1] = out1[i * 2 + 1] * float(i) / (numFrames - 1) + out2[i * 2 + 1] * (1 - float(i) / (numFrames - 1));
	}

}
// cufftComplex pointwise multiplication
__global__ void ComplexPointwiseMulInPlace(const cufftComplex* in, cufftComplex* out, int size) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadID; i < size; i += numThreads)
	{
		out[i] = cufftComplexMul(out[i], in[i]);
	}
}
__global__ void ComplexPointwiseAdd(cufftComplex* in, cufftComplex* out, int size)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadID; i < size; i += numThreads)
	{

		atomicAdd(&(out[i].x), in[i].x);
		atomicAdd(&(out[i].y), in[i].y);
		// out[i].x += in[i].x;
		// out[i].y += in[i].y;
	}
}

__global__ void timeDomainConvolutionNaive(float* ibuf, float* rbuf, float* obuf, long long oframes,
	long long rframes, int ch, float gain) {
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	float value = 0;
	for (int k = 0; k < rframes; k++) {
		value += ibuf[threadID - k] * rbuf[k];
	}
	obuf[threadID * 2 + ch] = value * gain;

}
__global__ void interleave(float* input, float* output, int size) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadID; i < size; i += numThreads) {
		output[2 * i] = input[i];
		output[2 * i + 1] = input[size + 2 + i];
	}
}

// cufftComplex pointwise multiplication
__global__ void ComplexPointwiseMul(cufftComplex* a, const cufftComplex* b, cufftComplex* c, int size){
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadID; i < size; i += numThreads)
	{
		c[i] = cufftComplexMul(a[i], b[i]);
	}
}
__global__ void MyFloatScale(float *a, int size, float scale) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadID; i < size; i += numThreads)
	{
		a[i] = a[i] * scale;
	}
}


// Pad data
int PadData(const float *signal, float **padded_signal, int signal_size,
	const float *filter_kernel, float **padded_filter_kernel, int filter_kernel_size)
{
	int minRadius = filter_kernel_size / 2;
	int maxRadius = filter_kernel_size - minRadius;
	int new_size = signal_size + maxRadius;

	// Pad signal
	float *new_data = (float *)malloc(sizeof(float) * new_size);
	memcpy(new_data + 0, signal, signal_size * sizeof(float));
	memset(new_data + signal_size, 0, (new_size - signal_size) * sizeof(float));
	*padded_signal = new_data;

	// Pad filter
	new_data = (float *)malloc(sizeof(float) * new_size);
	memcpy(new_data + 0, filter_kernel, filter_kernel_size * sizeof(float));
	memset(new_data + filter_kernel_size, 0, (new_size - filter_kernel_size) * sizeof(float));
	*padded_filter_kernel = new_data;
	return new_size;
}

// cufftComplex scale
__device__ __host__ inline cufftComplex cufftComplexScale(cufftComplex a, float s)
{
	cufftComplex c;
	c.x = s * a.x;
	c.y = s * a.y;
	return c;
}
// cufftComplex multiplication
__device__ __host__ inline cufftComplex cufftComplexMul(cufftComplex a, cufftComplex b)
{
	cufftComplex c;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;
	return c;
}


__global__ void averagingKernel(float4 *pos, float *d_buf, unsigned int size, double ratio, int averageSize) {

	unsigned long modNum = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned long samp_num = modNum * averageSize;
	if (samp_num < size) {

		int end;
		if (size < samp_num + averageSize - 1) {
			end = size;
		}
		else {
			end = samp_num + averageSize - 1;
		}
		thrust::negate<float> op;
		thrust::transform_if(thrust::device, d_buf + samp_num, d_buf + end, d_buf + samp_num, op, is_negative());
		float avg = thrust::reduce(thrust::device, d_buf + samp_num, d_buf + end, 0.0f, thrust::plus<float>());
		avg /= (float)averageSize;

		float x = (float)samp_num * ratio;
		/*Flat 2D waveform for testing*/
		pos[modNum * 2] = make_float4(x, avg, 0, 1.0f);
		pos[modNum * 2 + 1] = make_float4(x, -avg, 0, 1.0f);

	}
}

__global__ void fill_kernel(thrust::device_ptr<float> dev_ptr, long long old_size, long long new_size)
{
	thrust::fill(dev_ptr + old_size, dev_ptr + new_size, (float)0.0f);
}

__global__ void fillZeros(float* buf, int size) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadID; i < size; i += numThreads)
	{
		buf[i] = 0.0f;
	}
}
void fillWithZeroes(float** target_buf, long long old_size, long long new_size, cudaStream_t s) {
	thrust::device_ptr<float> dev_ptr(*target_buf);
	fill_kernel << <1, 1, 0, s >> > (dev_ptr, old_size, new_size);
}
void fillWithZeroes(float** target_buf, long long old_size, long long new_size) {
	thrust::device_ptr<float> dev_ptr(*target_buf);
	fill_kernel << <1, 1>> > (dev_ptr, old_size, new_size);
}
void fillWithZeroesKernel(float* buf, int size, cudaStream_t s) {
	int numThreads = 256;
	int numBlocks = (size + numThreads - 1) / numThreads;
	fillZeros << < numThreads, numBlocks, 0, s >> > (buf, size);
}
#include "cudaPart.cuh"

void cudaFFT(int argc, char **argv, Data *p){
	findCudaDevice(argc, (const char **)argv);
	int SIGNAL_SIZE = 0, FILTER_KERNEL_SIZE = 0;
	
	/*SNDFILE SETUP & READING FILES*/
	SF_INFO isfinfo, rsfinfo;
	SNDFILE *isndfile, *rsndfile;
	char *input = "Vltava.wav";
	char *reverb = "s1_r1_b_441_mono.wav";
	/*if (argc == 1) {
		input = argv[1];
	}
	if (argc == 2) {
		input = argv[1];
		reverb = argv[2];
	}*/

	/*Buffers for wave files*/
	float *ibuf, *rbuf;
	memset(&isfinfo, 0, sizeof(isfinfo));
	memset(&rsfinfo, 0, sizeof(rsfinfo));

	/*Open input file*/
	isfinfo.format = 0;
	isndfile = sf_open(input, SFM_READ, &isfinfo);
	if (isndfile == NULL) {
		fprintf(stderr, "ERROR. Cannot open %s\n", input);
		exit(1);
	}
	if (isfinfo.channels != 1) {
		printf("ERROR: Only mono sources accepted");
		exit(2);
	}
	if (isfinfo.samplerate != 44100) {
		printf("ERROR: Only 44.1k SR accepted");
		exit(3);
	}
	/*Open reverb file*/
	rsfinfo.format = 0;
	rsndfile = sf_open(reverb, SFM_READ, &rsfinfo);
	if (rsndfile == NULL) {
		printf("ERROR. Cannot open %s\n", reverb);
		exit(1);
	}
	if (rsfinfo.channels != 1) {
		printf("ERROR: Only mono sources accepted");
		exit(2);
	}
	if (rsfinfo.samplerate != 44100) {
		printf("ERROR: Only 44.1k SR accepted");
		exit(3);
	}
	

	/*Setup buffer sizes*/
	SIGNAL_SIZE = (int)(isfinfo.channels * isfinfo.frames);
	FILTER_KERNEL_SIZE = (int)(rsfinfo.channels * rsfinfo.frames);

	/*Allocate buffers*/
	ibuf = (float*)malloc(sizeof(float) * SIGNAL_SIZE);
	rbuf = (float*)malloc(sizeof(float) * FILTER_KERNEL_SIZE);

	/*Read buffers*/
	sf_read_float(isndfile, ibuf, SIGNAL_SIZE);
	sf_read_float(rsndfile, rbuf, FILTER_KERNEL_SIZE);
	printf("First data: %f\n", ibuf[0] * rbuf[0]);

	/*Pad signal and filter kernel to same length*/
	float *h_padded_signal;
	float *h_padded_filter_kernel;
	//new_size = SIGNAL_SIZE + (FILTER_KERNEL_SIZE) % 2
	int new_size = PadData(ibuf, &h_padded_signal, SIGNAL_SIZE,
		rbuf, &h_padded_filter_kernel, FILTER_KERNEL_SIZE);
	int mem_size = sizeof(float) * new_size;


	/*MOVING SIGNAL TO GPU*/
	// Allocate device memory for signal
	float *d_signal;
	checkCudaErrors(cudaMalloc((void **)&d_signal, mem_size));

	// Copy signal from host to device
	checkCudaErrors(cudaMemcpy(d_signal, h_padded_signal, mem_size,
		cudaMemcpyHostToDevice));

	/*MOVING IMPULSE TO GPU*/
	// Allocate device memory for filter kernel
	float *d_filter_kernel;
	checkCudaErrors(cudaMalloc((void **)&d_filter_kernel, mem_size));
	// Copy host memory to device
	checkCudaErrors(cudaMemcpy(d_filter_kernel, h_padded_filter_kernel, mem_size,
		cudaMemcpyHostToDevice));

	/*FIND RMS OF ORIGINAL SIGNAL*/
	/*Convert raw float pointer into a thrust device pointer*/
	thrust::device_ptr<float> thrust_d_signal(d_signal);
	/*Declare thrust operators*/
	square<float> unary_op;
	thrust::plus<float> binary_op;
	/*Perform thrust reduction to find rms*/
	float rms = std::sqrt(thrust::transform_reduce(thrust_d_signal, thrust_d_signal + new_size, unary_op, 0.0f, binary_op) / new_size);
	
	///////////////////////////////////////////////////////////////////////////////
	/*GPU PROCESSING*/
	///////////////////////////////////////////////////////////////////////////////
	
	// CUFFT plan simple API
	cufftHandle plan;
	checkCudaErrors(cufftPlan1d(&plan, new_size, CUFFT_R2C, 1));
	cufftHandle outplan;
	checkCudaErrors(cufftPlan1d(&outplan, new_size, CUFFT_C2R, 1));

	/*Create complex arrays*/
	cufftComplex *d_sig_complex;
	checkCudaErrors(cudaMalloc(&d_sig_complex, new_size * sizeof(cufftComplex)));
	cufftComplex *d_filter_complex;
	checkCudaErrors(cudaMalloc(&d_filter_complex, new_size * sizeof(cufftComplex)));

	/*FFT*/
	printf("Transforming signal cufftExecR2C\n");
	checkCudaErrors(cufftExecR2C(plan, (cufftReal *)d_signal, d_sig_complex));
	checkCudaErrors(cufftExecR2C(plan, (cufftReal *)d_filter_kernel, d_filter_complex));

	/*CONVOLUTION*/
	// Multiply the coefficients together and normalize the result
	printf("Launching ComplexPointwiseMulAndScale<<< >>>\n");
	int blockSize = 256;
	int numBlocks = (new_size + blockSize - 1) / blockSize;
	ComplexPointwiseMulAndScale << < numBlocks, blockSize >> > (d_sig_complex, d_filter_complex, new_size, 1.0f / new_size);
	// Check if kernel execution generated and error
	getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");

	/*IFFT*/
	// Transform signal back
	printf("Transforming signal back cufftExecC2R\n");
	checkCudaErrors(cufftExecC2R(outplan, d_sig_complex, d_signal));

	if (cudaDeviceSynchronize() != cudaSuccess) {
		fprintf(stderr, "Cuda error: failed to synchronize\n");
	}

	/*Find RMS of resulting signal*/
	thrust::device_ptr<float> thrust_d_output_signal2(d_signal);
	float rms2 = std::sqrt(thrust::transform_reduce(thrust_d_signal, thrust_d_signal + new_size, unary_op, 0.0f, binary_op) / new_size);
	printf("RMS1: %f RMS2: %f\n", rms, rms2);

	/*Scale resulting signal according to input signal*/
	MyFloatScale << < numBlocks, blockSize >> > (d_signal, new_size, rms / rms2);
	
	/*MOVE BACK TO CPU & STORE IN STRUCT*/
	float *obuf = (float*)malloc(sizeof(float) * new_size);
	checkCudaErrors(cudaMemcpy(obuf, d_signal, new_size * sizeof(float), cudaMemcpyDeviceToHost));
	p->buf = obuf;
	p->length = new_size;


	////////////////////////////////////////////////////////////////////////////////
	///*NOTE: GPU Convolution was not fast enough because of the large overhead
	//of FFT and IFFT. Keeping the code here for future purposes*/
	//
	/*Convolved signal on device*/
	//p->dbuf = d_signal;
	////////////////////////////////////////////////////////////////////////////////

	/*Destroy CUFFT context*/
	checkCudaErrors(cufftDestroy(plan));
	checkCudaErrors(cufftDestroy(outplan));
	isfinfo.seekable = 1;

	/*Write sound file*/
	SndfileHandle file = SndfileHandle("output.wav", SFM_WRITE, isfinfo.format, isfinfo.channels, isfinfo.samplerate);
	file.write(obuf, new_size);


	/*Free memory*/
	sf_close(isndfile);
	sf_close(rsndfile);

	free(ibuf);
	free(rbuf);

	free(h_padded_signal);
	free(h_padded_filter_kernel);

	checkCudaErrors(cudaFree(d_signal));
	checkCudaErrors(cudaFree(d_filter_kernel));
	checkCudaErrors(cudaFree(d_sig_complex));
	checkCudaErrors(cudaFree(d_filter_complex));


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

// Complex scale
static __device__ __host__ inline Complex ComplexScale(Complex a, float s)
{
	Complex c;
	c.x = s * a.x;
	c.y = s * a.y;
	return c;
}
// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b)
{
	Complex c;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;
	return c;
}
// Complex pointwise multiplication
static __global__ void ComplexPointwiseMulAndScale(Complex *a, const Complex *b, int size, float scale)
{
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadID; i < size; i += numThreads)
	{
		a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
	}
}
static __global__ void MyFloatScale(float *a, int size, float scale) {
	const int numThreads = blockDim.x * gridDim.x;
	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = threadID; i < size; i += numThreads)
	{
		a[i] = a[i] * scale;
	}
}
////////////////////////////////////////////////////////////////////////////////
///*NOTE: GPU Convolution was not fast enough because of the large overhead
//of FFT and IFFT. Keeping the code here for future purposes*/
//
//void __global__ padData(int size, float *padder) {
//	const int numThreads = blockDim.x * gridDim.x;
//	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
//
//	for (int i = threadID; i < size; i += numThreads)
//	{
//		padder[i] = 0.0f;
//	}
//}
//static __global__ void interleaveMe(float *output, float *input, int size) {
//	const int numThreads = blockDim.x * gridDim.x;
//	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
//
//	for (int i = threadID; i < size; i += numThreads)
//	{
//		output[i * 2 + 1] = input[i];
//		output[i * 2] = input[i];
//	}
//}
//
//__global__ void copyMe(int size, float *output, float *input) {
//	const int numThreads = blockDim.x * gridDim.x;
//	const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
//
//	for (int i = threadID; i < size; i += numThreads)
//	{
//		output[i] = input[i];
//	}
//}
//
//void convolveMe(float *output, float *input, int x_len, float *p_hrtf, float gain, float *d_hrtf) {
//	int outputLength = x_len * 2;
//
//	int blockSize = 256;
//	int numBlocks = (x_len * 2 + blockSize - 1) / blockSize;
//	/*Interleave the input signal*/
//	float *d_interleaved;
//	checkCudaErrors(cudaMalloc((void **)&d_interleaved, outputLength * sizeof(float)));
//	interleaveMe << < numBlocks, blockSize >> > (d_interleaved, input, x_len);
//	cudaThreadSynchronize();
//	
//
//	/*pad the HRTF signal*/
//	float *d_padded_hrtf;
//	checkCudaErrors(cudaMalloc((void **)&d_padded_hrtf, outputLength * sizeof(float)));
//	padData << < numBlocks, blockSize >> > (x_len * 2 - HRTF_LEN, &d_padded_hrtf[HRTF_LEN]);
//	cudaThreadSynchronize();
//	copyMe << < numBlocks, blockSize >> > (HRTF_LEN, d_padded_hrtf, d_hrtf);
//	cudaThreadSynchronize();
//
//
//	/*CUFFT plan simple API*/
//	cufftHandle plan;
//	checkCudaErrors(cufftPlan1d(&plan, outputLength, CUFFT_R2C, 1));
//	cufftHandle outplan;
//	checkCudaErrors(cufftPlan1d(&outplan, outputLength, CUFFT_C2R, 1));
//
//	//Create complex arrays
//	cufftComplex *d_sig_complex;
//	checkCudaErrors(cudaMalloc(&d_sig_complex, outputLength * sizeof(cufftComplex)));
//	cufftComplex *d_filter_complex;
//	checkCudaErrors(cudaMalloc(&d_filter_complex, outputLength * sizeof(cufftComplex)));
//	
//	/*FFT*/
//	checkCudaErrors(cufftExecR2C(plan, (cufftReal *)d_interleaved, d_sig_complex));
//	checkCudaErrors(cufftExecR2C(plan, (cufftReal *)d_padded_hrtf, d_filter_complex));
//	
//	/*CONVOLUTION*/
//	//Multiply the coefficients together and normalize the result
//	ComplexPointwiseMulAndScale <<< numBlocks, blockSize >> > (d_sig_complex, d_filter_complex, outputLength, 1.0f / outputLength);
//	cudaThreadSynchronize();
//	//Check if kernel execution generated and error
//	getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");
//	cudaThreadSynchronize();
//
//	/*IFFT*/
//	checkCudaErrors(cufftExecC2R(outplan, d_sig_complex, d_interleaved));
//	if (cudaDeviceSynchronize() != cudaSuccess) {
//		fprintf(stderr, "Cuda error: failed to synchronize\n");
//	}
//	
//	/*Copy result into output*/
//	checkCudaErrors(cudaMemcpy(output, d_interleaved, x_len * 2 * sizeof(float), cudaMemcpyDeviceToHost));
//
//	//Destroy CUFFT context
//	checkCudaErrors(cufftDestroy(plan));
//	checkCudaErrors(cufftDestroy(outplan));
//
//	/*Free memory*/
//	cudaFree(d_interleaved);
//	cudaFree(d_padded_hrtf);
//	cudaFree(d_sig_complex);
//	cudaFree(d_filter_complex);
//}
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////


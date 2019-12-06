#include "cudaPart.cuh"
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
typedef float2 Complex; 
int readFile(const char *name, float **buf, int &numCh) {
	SF_INFO info;
	SNDFILE *sndfile;
	memset(&info, 0, sizeof(info));
	info.format = 0;
	sndfile = sf_open(name, SFM_READ, &info);
	if (sndfile == NULL) {
		fprintf(stderr, "ERROR. Cannot open %s\n", name);
		exit(1);
	}

	int size = info.frames;
	numCh = info.channels;

	*buf = (float*)malloc(sizeof(float) * size);

	if (info.channels == 1) {
		sf_read_float(sndfile, *buf, size);
	}

	else {
		/*Sum into mono & do RMS*/
		if (info.channels == 2) {
			/*Allocate temporary memory for wave file*/
			float *temp_buf = (float*)malloc(sizeof(float) * info.frames * 2);

			/*Read wave file into temporary memory*/
			sf_read_float(sndfile, temp_buf, info.frames * 2);

			/*Sum R & L*/
			for (int i = 0; i < info.frames; i++) {
				*buf[i] = temp_buf[i * 2] / 2.0 + temp_buf[i * 2 + 1] / 2.0;
			}

			free(temp_buf);

		}
		else {
			fprintf(stderr, "ERROR: %s : Only mono or stereo accepted", name);
		}
	}
	sf_close(sndfile);
	return size;
}

void cudaFFT(int argc, char **argv, Data *p) {

	std::string input = "media/Taiklatalvi.wav";
	std::string reverb = "media/medieval_church.wav";
	if (argc == 2) {
		if (argv[1][1] != '>')
			input = argv[1];
	}
	if (argc == 3) {
		input = argv[1];
		reverb = argv[2];
	}

	float *ibuf, *rbuf;
	int SIGNAL_SIZE = 0, FILTER_KERNEL_SIZE = 0;

	fprintf(stderr, "Reading input file\n");
	int inputCh;
	SIGNAL_SIZE = readFile(input.c_str(), &ibuf, inputCh);

	fprintf(stderr, "Reading reverb file\n");
	FILTER_KERNEL_SIZE = readFile(reverb.c_str(), &rbuf, inputCh);
	if (inputCh != 1) {
		fprintf(stderr, "ERROR: Only mono reverb sources accepted");
		exit(2);
	}

	findCudaDevice(argc, (const char **)argv);

	fprintf(stderr, "Doing GPU Convolution\n");
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
	CHECK_CUFFT_ERRORS(cufftPlan1d(&plan, new_size, CUFFT_R2C, 1));
	cufftHandle outplan;
	CHECK_CUFFT_ERRORS(cufftPlan1d(&outplan, new_size, CUFFT_C2R, 1));

	/*Create complex arrays*/
	cufftComplex *d_sig_complex;
	checkCudaErrors(cudaMalloc(&d_sig_complex, new_size * sizeof(cufftComplex)));
	cufftComplex *d_filter_complex;
	checkCudaErrors(cudaMalloc(&d_filter_complex, new_size * sizeof(cufftComplex)));

	/*FFT*/
	printf("Transforming signal cufftExecR2C\n");
	CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal *)d_signal, d_sig_complex));
	CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal *)d_filter_kernel, d_filter_complex));

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
	CHECK_CUFFT_ERRORS(cufftExecC2R(outplan, d_sig_complex, d_signal));

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
	p->all_sources[0].buf = obuf;
	p->all_sources[0].length = new_size;

	fprintf(stderr, "Samples: %i\nTotal Bytes: %i\nTotal KB: %f3\nTotal MB: %f3\n\n\n", new_size, mem_size, mem_size / (float)1024, mem_size / (float)1024 / (float)1024);

	/*Write reverberated sound file*/
	//SndfileHandle file = SndfileHandle("output.wav", SFM_WRITE, isfinfo.format, isfinfo.channels, isfinfo.samplerate);
	//file.write(obuf, new_size);

	/*Destroy CUFFT context*/
	CHECK_CUFFT_ERRORS(cufftDestroy(plan));
	CHECK_CUFFT_ERRORS(cufftDestroy(outplan));

	/*Free memory*/

	free(ibuf);
	free(rbuf);

	free(h_padded_signal);
	free(h_padded_filter_kernel);

	checkCudaErrors(cudaFree(d_signal));
	checkCudaErrors(cudaFree(d_filter_kernel));
	checkCudaErrors(cudaFree(d_sig_complex));
	checkCudaErrors(cudaFree(d_filter_complex));


}

void transform_hrtfs() {

	cufftHandle plan;

	int n = PAD_LEN;
	/*cufftResult cufftPlanMany(cufftHandle *plan, int rank, int *n,
		int *inembed, int istride, int idist,
		int *onembed, int ostride, int odist,
		cufftType type, int batch);*/
		/*stride = skip length. Ex 1 = every element, 2 = every other element*/
			/*use for interleaving???*/
		/*idist/odist is space between batches of transforms*/
			/*need to check if odist is in terms of complex numbers or floats*/
		/*inembed/onembed are for 2D/3D, num elements per dimension*/
	/*CHECK_CUFFT_ERRORS(
		cufftPlanMany(&plan, 1, &n,
			&n, 1, n + 2,
			&n, 1, n / 2 + 1,
			CUFFT_R2C, NUM_HRTF * 2)
	);
	CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal*)d_hrtf, (cufftComplex*)d_hrtf));*/

	CHECK_CUFFT_ERRORS(cufftPlan1d(&plan, PAD_LEN, CUFFT_R2C, NUM_HRTF * 2));
	size_t size = sizeof(float) * NUM_HRTF * HRTF_CHN * (PAD_LEN + 2);
	float* d_dummy,* d_dummy2;
	checkCudaErrors(cudaMalloc((void**)&d_dummy, size));
	checkCudaErrors(cudaMalloc((void**)&d_dummy2, size));
	for (int i = 0; i < NUM_HRTF; i++) {
		cufftReal* lbuf = d_dummy2 + i * HRTF_CHN * (PAD_LEN + 2);
		cufftReal* rbuf = d_dummy2 + i * HRTF_CHN * (PAD_LEN + 2) + PAD_LEN + 2;
		CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal*) lbuf, (cufftComplex*)lbuf));
		checkCudaErrors(cudaDeviceSynchronize());
		CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal*) rbuf, (cufftComplex*)rbuf));
		checkCudaErrors(cudaDeviceSynchronize());
	}
	
	CHECK_CUFFT_ERRORS(cufftDestroy(plan));
}
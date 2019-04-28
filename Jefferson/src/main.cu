#include "main.cuh"
#include <chrono>
#include <thread>
#include <cuda.h>
#include <cuda_profiler_api.h>
int main(int argc, char *argv[]){
	if (argc > 3 ) {
		fprintf(stderr, "Usage: %s input.wav reverb.wav", argv[0]);
		return 0;
	}
	
	data.count = 0;
	data.length = 0;
	data.gain = 0.99074;
	data.hrtf_idx = 151;
	#if(DEBUGMODE != 1)
		/*Initialize & read files*/
		cudaFFT(argc, argv, p);
	
		fprintf(stderr, "Opening and Reading HRTF signals\n");
		/*Open & read hrtf files*/

		if (read_hrtf_signals() != 0) {
			exit(EXIT_FAILURE);
		}
		p->hrtf_idx = 0;


		fprintf(stderr, "Opening output file\n");
		SF_INFO osfinfo;
		osfinfo.channels = 2;
		osfinfo.samplerate = 44100;
		osfinfo.format = SF_FORMAT_PCM_24 | SF_FORMAT_WAV;
		p->sndfile = sf_open("ofile.wav", SFM_WRITE, &osfinfo);
		p->count = 0;

		p->streams = new cudaStream_t[FLIGHT_NUM * 2];
		for (int i = 0; i < FLIGHT_NUM; i++){
			/*Allocating memory for the inputs*/
			checkCudaErrors(cudaMalloc(&(p->d_input[i]), COPY_AMT * sizeof(float)));
			/*Allocating memory for the outputs*/
			checkCudaErrors(cudaMalloc(&(p->d_output[i]), FRAMES_PER_BUFFER * HRTF_CHN * sizeof(float)));
			/*Creating the streams*/
			checkCudaErrors(cudaStreamCreate(&(p->streams[i * 2])));
			checkCudaErrors(cudaStreamCreate(&(p->streams[i * 2 + 1])));
		}
		/*Allocating pinned memory for incoming transfer*/
		checkCudaErrors(cudaMallocHost(&(p->x), COPY_AMT * sizeof(float)));
		/*Allocating pinned memory for outgoing transfer*/
		checkCudaErrors(cudaMallocHost(&(p->intermediate), FRAMES_PER_BUFFER * HRTF_CHN * sizeof(float)));

		/*Setting initial input to 0*/
		for (int i = 0; i < HRTF_LEN - 1; i++){
			p->x[i] = 0.0f;
		}
		p->blockNo = 0;
		cudaProfilerStart();
		for (int i = 0; i < FLIGHT_NUM; i++) {
			/*Copy new input chunk into pinned memory*/
			memcpy(p->x + HRTF_LEN - 1, p->buf + p->count, FRAMES_PER_BUFFER * sizeof(float));
			p->count += FRAMES_PER_BUFFER;

			/*Send*/
			checkCudaErrors(cudaMemcpyAsync(
				p->d_input[p->blockNo],
				p->x,
				COPY_AMT * sizeof(float),
				cudaMemcpyHostToDevice,
				p->streams[p->blockNo * 2])
			);
			if (i == 0) {
				goto end;
			}
			/*Process*/
			GPUconvolve_hrtf(
				p->d_input[p->blockNo - 1] + HRTF_LEN,
				p->hrtf_idx,
				p->d_output[(p->blockNo - 1) % FLIGHT_NUM],
				FRAMES_PER_BUFFER,
				p->gain,
				p->streams + (p->blockNo - 1) * 2
			);
			if (i < FLIGHT_NUM - 1) {
				goto end;
			}
			/*Idle*/
			/*Idle*/
			/*Return*/
			checkCudaErrors(cudaMemcpyAsync(
				p->intermediate,
				p->d_output[(p->blockNo - FLIGHT_NUM + 1) % FLIGHT_NUM],
				FRAMES_PER_BUFFER * 2 * sizeof(float),
				cudaMemcpyDeviceToHost,
				p->streams[(p->blockNo - FLIGHT_NUM + 1) % FLIGHT_NUM * 2])
			);
			end: /*overlap-save*/
			memcpy(p->x, p->x + FRAMES_PER_BUFFER, (HRTF_LEN - 1) * sizeof(float));
			p->blockNo++;
		}
		checkCudaErrors(cudaDeviceSynchronize());
	#endif
	

#if(DEBUGMODE != 1)
	fprintf(stderr, "\n\n\n\nInitializing PortAudio\n\n\n\n");
	initializePA(SAMPLE_RATE);
	printf("\n\n\n\nStarting playout\n");
#endif
	///////////////////////////////////////////////////////////////////////////////////////////////////
	/*MAIN FUNCTIONAL LOOP*/
	/*Here to debug without graphics*/
#if DEBUGMODE == 2
	std::this_thread::sleep_for(std::chrono::seconds((p->length)/44100));
	//char merp = getchar();
#else
	graphicsMain(argc, argv, p);
#endif
	
	/*THIS SECTION WILL NOT RUN IF GRAPHICS IS TURNED ON*/
	/*Placed here to properly close files when debugging without graphics*/
	cudaProfilerStop();
	
	closeEverything();

	return 0;
}

void closeEverything(){
	closePA();
	sf_close(p->sndfile);
	for(int i = 0; i < FLIGHT_NUM; i++){
		checkCudaErrors(cudaFree(p->d_input[i]));
		checkCudaErrors(cudaFree(p->d_output[i]));
		checkCudaErrors(cudaStreamSynchronize(p->streams[i * 2]));
		checkCudaErrors(cudaStreamSynchronize(p->streams[i * 2 + 1]));
		checkCudaErrors(cudaStreamDestroy(p->streams[i * 2]));
		checkCudaErrors(cudaStreamDestroy(p->streams[i * 2 + 1]));
	}
	checkCudaErrors(cudaFreeHost(p->x));
	checkCudaErrors(cudaFreeHost(p->intermediate));
	free(p->buf);
}
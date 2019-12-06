#include "main.cuh"
#include <chrono>
#include <thread>
#include <cuda.h>
#include <cuda_profiler_api.h>

struct Data data;
struct Data* p = &data;
int main(int argc, char *argv[]){
	if (argc > 3 ) {
		fprintf(stderr, "Usage: %s input.wav reverb.wav", argv[0]);
		return 0;
	}
	p->num_sources = 1;
	p->all_sources = new SoundSource[p->num_sources];
	for (int i = 0; i < p->num_sources; i++) {
		p->all_sources[i].count = 0;
		p->all_sources[i].length = 0;
		p->all_sources[i].gain = 0.99074;
		p->all_sources[i].hrtf_idx = 314;
		//p->all_sources[i].hrtf_idx = 100;
	}
	#if(DEBUGMODE != 1)
		/*Initialize & read files*/
		cudaFFT(argc, argv, p);
	
		
		fprintf(stderr, "Opening and Reading HRTF signals\n");
		/*Open & read hrtf files*/

		if (read_hrtf_signals() != 0) {
			exit(EXIT_FAILURE);
		}

		transform_hrtfs();

		fprintf(stderr, "Opening output file\n");
		SF_INFO osfinfo;
		osfinfo.channels = 2;
		osfinfo.samplerate = 44100;
		osfinfo.format = SF_FORMAT_PCM_24 | SF_FORMAT_WAV;
		p->sndfile = sf_open("ofile.wav", SFM_WRITE, &osfinfo);


		printf("Blocks in flight: %i\n", FLIGHT_NUM);
		cudaProfilerStart();
		for (int i = 0; i < p->num_sources; i++) {

			p->all_sources[i].streams = new cudaStream_t[FLIGHT_NUM * 2];
		}

		for (int i = 0; i < FLIGHT_NUM; i++){
			for (int j = 0; j < p->num_sources; j++) {
				SoundSource* curr_source = &(p->all_sources[j]);
				/*Allocating pinned memory for incoming transfer*/
				checkCudaErrors(cudaMallocHost(&(curr_source->x[i]), (PAD_LEN + 2) * sizeof(float)));
				/*Allocating memory for the inputs*/
				checkCudaErrors(cudaMalloc(&(curr_source->d_input[i]), (PAD_LEN + 2) * sizeof(float)));
				/*Allocating memory for raw, uncropped convolution output*/
				//checkCudaErrors(cudaMalloc(&(curr_source->d_uninterleaved[i]), 2 * (PAD_LEN + 2) * sizeof(float)));
				/*Allocating memory for the outputs*/
				checkCudaErrors(cudaMalloc(&(curr_source->d_output[i]), 2 * (PAD_LEN + 2) * sizeof(float)));
				/*Creating the streams*/
				checkCudaErrors(cudaStreamCreate(&(curr_source->streams[i * 2])));
				checkCudaErrors(cudaStreamCreate(&(curr_source->streams[i * 2 + 1])));
				/*Allocating pinned memory for outgoing transfer*/
				checkCudaErrors(cudaMallocHost(&(curr_source->intermediate[i]), (FRAMES_PER_BUFFER * HRTF_CHN) * sizeof(float)));

				
			}
		}
		for (int i = 0; i < FLIGHT_NUM; i++) {
			for (int j = 0; j < p->num_sources; j++) {
				SoundSource* curr_source = &(p->all_sources[j]);
				for (int k = 0; k < PAD_LEN + 2; k++) {
					curr_source->x[i][k] = 0.0f;
				}
			}
		}
		
		p->blockNo = 0;
		for (int i = 0; i < FLIGHT_NUM; i++) {
			for(int j = 0; j < p->num_sources; j++){
				SoundSource* curr_source = &(p->all_sources[j]);
				/*Copy new input chunk into pinned memory*/
				memcpy(
					curr_source->x[p->blockNo] + (PAD_LEN - FRAMES_PER_BUFFER),  /*Go to the end and work backwards*/
					curr_source->buf + curr_source->count, 
					FRAMES_PER_BUFFER * sizeof(float)
				);
				curr_source->count += FRAMES_PER_BUFFER;

				/*Send*/
				checkCudaErrors(cudaMemcpyAsync(
					curr_source->d_input[p->blockNo],
					curr_source->x[p->blockNo],
					PAD_LEN * sizeof(float),
					cudaMemcpyHostToDevice,
					curr_source->streams[p->blockNo * 2])
				);
				if (i == 0) {
					goto end;
				}
				/*Process*/
				curr_source->fftConvolve(p->blockNo - 1);
				/*GPUconvolve_hrtf(
					curr_source->d_input[p->blockNo - 1],
					curr_source->hrtf_idx,
					curr_source->d_output[(p->blockNo - 1) % FLIGHT_NUM],
					FRAMES_PER_BUFFER,
					curr_source->gain,
					curr_source->streams + (p->blockNo - 1) * 2
				);*/
				if (i == 1) {
					goto end;
				}
				checkCudaErrors(cudaMemcpyAsync(
					curr_source->intermediate[(p->blockNo - 2) % FLIGHT_NUM],
					curr_source->d_output[(p->blockNo - 2) % FLIGHT_NUM] + 2 * (PAD_LEN - FRAMES_PER_BUFFER),
					FRAMES_PER_BUFFER * 2 * sizeof(float),
					cudaMemcpyDeviceToHost,
					curr_source->streams[(p->blockNo - 2) % FLIGHT_NUM * 2])
				);
				
				end: /*overlap-save*/
				memcpy(
					curr_source->x[(p->blockNo + 1) % FLIGHT_NUM], 
					curr_source->x[p->blockNo] + (PAD_LEN - FRAMES_PER_BUFFER),
					(HRTF_LEN - 1) * sizeof(float)
				);
			}
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
	std::this_thread::sleep_for(std::chrono::seconds((p->all_sources[0].length)/44100));
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
	for(int source_no = 0; source_no < p->num_sources; source_no++){
		for(int i = 0; i < FLIGHT_NUM; i++){
			checkCudaErrors(cudaFree(p->all_sources[source_no].d_input[i]));
			checkCudaErrors(cudaFree(p->all_sources[source_no].d_output[i]));
			checkCudaErrors(cudaFreeHost(p->all_sources[source_no].intermediate[i]));
			checkCudaErrors(cudaFreeHost(p->all_sources[source_no].x[i]));
			checkCudaErrors(cudaStreamSynchronize(p->all_sources[source_no].streams[i * 2]));
			checkCudaErrors(cudaStreamSynchronize(p->all_sources[source_no].streams[i * 2 + 1]));
			checkCudaErrors(cudaStreamDestroy(p->all_sources[source_no].streams[i * 2]));
			checkCudaErrors(cudaStreamDestroy(p->all_sources[source_no].streams[i * 2 + 1]));
		}

		free(p->all_sources[source_no].buf);
	}
	
}
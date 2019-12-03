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
	data.num_sources = 1;
	data.all_sources = new source_info[data.num_sources];
	for (int i = 0; i < data.num_sources; i++) {

		data.all_sources[i].count = 0;
		data.all_sources[i].length = 0;
		data.all_sources[i].gain = 0.99074;
		data.all_sources[i].hrtf_idx = 151;
	}
	#if(DEBUGMODE != 1)
		/*Initialize & read files*/
		cudaFFT(argc, argv, p);
	
		fprintf(stderr, "Opening and Reading HRTF signals\n");
		/*Open & read hrtf files*/

		if (read_hrtf_signals() != 0) {
			exit(EXIT_FAILURE);
		}


		fprintf(stderr, "Opening output file\n");
		SF_INFO osfinfo;
		osfinfo.channels = 2;
		osfinfo.samplerate = 44100;
		osfinfo.format = SF_FORMAT_PCM_24 | SF_FORMAT_WAV;
		p->sndfile = sf_open("ofile.wav", SFM_WRITE, &osfinfo);


		printf("Blocks in flight: %i\n", FLIGHT_NUM);
		cudaProfilerStart();
		for (int i = 0; i < data.num_sources; i++) {

			p->all_sources[i].streams = new cudaStream_t[FLIGHT_NUM * 2];
		}

		for (int i = 0; i < FLIGHT_NUM; i++){
			for (int j = 0; j < data.num_sources; j++) {
				/*Allocating memory for the inputs*/
				checkCudaErrors(cudaMalloc(&(p->all_sources[j].d_input[i]), COPY_AMT * sizeof(float)));
				/*Allocating memory for the outputs*/
				checkCudaErrors(cudaMalloc(&(p->all_sources[j].d_output[i]), FRAMES_PER_BUFFER * HRTF_CHN * sizeof(float)));
				/*Creating the streams*/
				checkCudaErrors(cudaStreamCreate(&(p->all_sources[j].streams[i * 2])));
				checkCudaErrors(cudaStreamCreate(&(p->all_sources[j].streams[i * 2 + 1])));
				/*Allocating pinned memory for outgoing transfer*/
				checkCudaErrors(cudaMallocHost(&(p->all_sources[j].intermediate[i]), FRAMES_PER_BUFFER * HRTF_CHN * sizeof(float)));

				/*Allocating pinned memory for incoming transfer*/
				checkCudaErrors(cudaMallocHost(&(p->all_sources[j].x[i]), COPY_AMT * sizeof(float)));
			}
		}
		for (int i = 0; i < FLIGHT_NUM; i++) {
			for (int j = 0; j < data.num_sources; j++) {
				for (int k = 0; k < FRAMES_PER_BUFFER + HRTF_LEN - 1; k++) {
					p->all_sources[j].x[i][k] = 0.0f;
				}
			}
		}
		
		p->blockNo = 0;
		for (int i = 0; i < FLIGHT_NUM; i++) {
			for(int j = 0; j < data.num_sources; j++){
				/*Copy new input chunk into pinned memory*/
				memcpy(p->all_sources[j].x[p->blockNo] + HRTF_LEN - 1, p->all_sources[j].buf + p->all_sources[j].count, FRAMES_PER_BUFFER * sizeof(float));
				p->all_sources[j].count += FRAMES_PER_BUFFER;

				/*Send*/
				checkCudaErrors(cudaMemcpyAsync(
					p->all_sources[j].d_input[p->blockNo],
					p->all_sources[j].x[p->blockNo],
					COPY_AMT * sizeof(float),
					cudaMemcpyHostToDevice,
					p->all_sources[j].streams[p->blockNo * 2])
				);
				if (i == 0) {
					goto end;
				}
				/*Process*/
				GPUconvolve_hrtf(
					p->all_sources[j].d_input[p->blockNo - 1] + HRTF_LEN,
					p->all_sources[j].hrtf_idx,
					p->all_sources[j].d_output[(p->blockNo - 1) % FLIGHT_NUM],
					FRAMES_PER_BUFFER,
					p->all_sources[j].gain,
					p->all_sources[j].streams + (p->blockNo - 1) * 2
				);
				if (i == 1) {
					goto end;
				}
				checkCudaErrors(cudaMemcpyAsync(
					p->all_sources[j].intermediate[(p->blockNo - 2) % FLIGHT_NUM],
					p->all_sources[j].d_output[(p->blockNo - 2) % FLIGHT_NUM],
					FRAMES_PER_BUFFER * 2 * sizeof(float),
					cudaMemcpyDeviceToHost,
					p->all_sources[j].streams[(p->blockNo - 2) % FLIGHT_NUM * 2])
				);
				
				end: /*overlap-save*/
				memcpy(p->all_sources[j].x[(p->blockNo + 1) % FLIGHT_NUM], p->all_sources[j].x[p->blockNo] + FRAMES_PER_BUFFER, (HRTF_LEN - 1) * sizeof(float));
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
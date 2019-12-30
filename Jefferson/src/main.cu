#include "main.cuh"
#include <chrono>
#include <thread>
#include <cuda.h>
#include <cuda_profiler_api.h>

void printSize() {
	size_t free = 0, total = 0;
	checkCudaErrors(cudaMemGetInfo(&free, &total));
	fprintf(stderr, "GPU Global Memory Stats: Size Free: %.2fMB\tSize Total: %.2fMB\tSize Used: %.2fMB\n", free / 1048576.0f, total / 1048576.0f, (total - free) / 1048576.0f);
}
struct Data data;
struct Data* p = &data;
int main(int argc, char *argv[]){
	//if (argc > 3 ) {
		//fprintf(stderr, "Usage: %s input.wav reverb.wav", argv[0]);
		//return 0;
	//}
	p->num_sources = 1;
	p->all_sources = new SoundSource[p->num_sources]; /*Moving all allocation & initialization into the constructor*/
	printSize();
#if(DEBUGMODE != 1)
		/*Initialize & read files*/
		cudaFFT(argc, argv, p);
			
		fprintf(stderr, "Opening and Reading HRTF signals\n");
		/*Open & read hrtf files*/

		if (read_hrtf_signals() != 0) {
			exit(EXIT_FAILURE);
		}
	#if defined RT_GPU && !defined RT_GPU_TD 
		transform_hrtfs();
	#endif
		fprintf(stderr, "Opening output file\n");
		SF_INFO osfinfo;
		osfinfo.channels = 2;
		osfinfo.samplerate = 44100;
		osfinfo.format = SF_FORMAT_PCM_24 | SF_FORMAT_WAV;
		p->sndfile = sf_open("ofile.wav", SFM_WRITE, &osfinfo);
		
	#ifdef RT_GPU
		printf("Blocks in flight: %i\n", FLIGHT_NUM);
		
		float* output = new float[FRAMES_PER_BUFFER * 2];
		p->blockNo = 0;
		for (int i = 0; i < FLIGHT_NUM; i++) {
			for (int j = 0; j < p->num_sources; j++) {
				callback_func(output, p);
			}			
		}
		delete[] output;
	#endif

#endif
#if(DEBUGMODE % 2 == 0)
	fprintf(stderr, "\n\n\n\nInitializing PortAudio\n\n\n\n");
	initializePA(SAMPLE_RATE);
	printf("\n\n\n\nStarting playout\n");
#endif
	///////////////////////////////////////////////////////////////////////////////////////////////////
	/*MAIN FUNCTIONAL LOOP*/
#if DEBUGMODE == 1 || DEBUGMODE == 0
	graphicsMain(argc, argv, p);
#endif
#if DEBUGMODE == 2
	cudaProfilerStart();

	int counter = 1;
	while (p->all_sources[0].count < (counter * 44100) % p->all_sources[0].length) {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
	counter++;
	p->all_sources[0].azi = 2;
	p->all_sources[0].ele = 4;
	p->all_sources[0].updateFromSpherical();

	while (p->all_sources[0].count < (counter * 44100) % p->all_sources[0].length) {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
	counter++;
	p->all_sources[0].azi = 1;
	p->all_sources[0].ele = 3;
	p->all_sources[0].updateFromSpherical();
	//std::this_thread::sleep_for(std::chrono::seconds(1));
	while (p->all_sources[0].count < (counter * 44100) % p->all_sources[0].length) {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
	counter++;
	p->all_sources[0].azi = 4;
	p->all_sources[0].ele = 2;
	p->all_sources[0].updateFromSpherical();
	while (p->all_sources[0].count < (counter * 44100) % p->all_sources[0].length) {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
	counter++;
	p->all_sources[0].azi = 7;
	p->all_sources[0].ele = 9;
	p->all_sources[0].updateFromSpherical();
	while (p->all_sources[0].count < (counter * 44100) % p->all_sources[0].length) {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
	counter++;
	p->all_sources[0].azi = 0;
	p->all_sources[0].ele = 0;
	p->all_sources[0].updateFromSpherical();
	std::this_thread::sleep_for(std::chrono::seconds(2));
	//char merp = getchar();

	/*THIS SECTION WILL NOT RUN IF GRAPHICS IS TURNED ON*/
	/*Placed here to properly close files when debugging without graphics*/
	cudaProfilerStop();

	closeEverything();
#endif
#if DEBUGMODE == 3
	benchmarkTesting();
	fprintf(stderr, "Number of function calls: %llu\n", p->all_sources[0].num_calls);
#endif
	return 0;
}

void closeEverything(){
	closePA();
	checkCudaErrors(cudaDeviceSynchronize());
	sf_close(p->sndfile);
	delete[] hrtf;
#ifdef CPU_FD_BASIC
	fftwf_free(fft_hrtf);
#endif
	checkCudaErrors(cudaFree(d_hrtf));	
}

void benchmarkTesting(){
	cudaProfilerStart();
	float* output = new float[FRAMES_PER_BUFFER * 2];
	int num_iterations = 172;
	fprintf(stderr, "Testing no interpolation\n");
	for (int repeats = 0; repeats < 2; repeats++) {
		p->all_sources[0].azi = 0;
		p->all_sources[0].ele = 0;
		int ele_idx = 4;
		for (int i = 0; i < num_iterations; i++) {
			callback_func(output, p);
		}
		for (int i = 0; i < 36; i++) {
			p->all_sources[0].azi += azimuth_inc[ele_idx];
			if (i % 4 == 0) {
				p->all_sources[0].ele += 10;
				ele_idx++;
			}
			p->all_sources[0].updateFromSpherical();
			for (int j = 0; j < num_iterations; j++) {
				callback_func(output, p);
			}
		}
	}
	fprintf(stderr, "Testing azimuth interpolation\n");
	for (int repeats = 0; repeats < 2; repeats++) {
		p->all_sources[0].azi = 1;
		p->all_sources[0].ele = 0;
		for (int i = 0; i < num_iterations; i++) {
			callback_func(output, p);
		}
		for (int i = 0; i < 72; i++) {
			p->all_sources[0].azi += 2.5;
			if (i % 8 == 0) {
				p->all_sources[0].ele += 10;
			}
			p->all_sources[0].updateFromSpherical();
			for (int j = 0; j < num_iterations / 2; j++) {
				callback_func(output, p);
			}
		}
	}
	/*fprintf(stderr, "Testing Elevation interpolation\n");
	p->all_sources[0].azi = 0;
	p->all_sources[0].ele = 5;
	p->all_sources[0].updateFromSpherical();
	for (int repeats = 0; repeats < 10; repeats++) {
		for (int i = 0; i < num_iterations; i++) {
			callback_func(output, p);
		}
		p->all_sources[0].ele = 15;
		p->all_sources[0].updateFromSpherical();
		for (int i = 0; i < num_iterations; i++) {
			callback_func(output, p);
		}
		p->all_sources[0].ele = 25;
		p->all_sources[0].updateFromSpherical();
		for (int i = 0; i < num_iterations; i++) {
			callback_func(output, p);
		}
		p->all_sources[0].ele = 35;
		p->all_sources[0].updateFromSpherical();
		for (int i = 0; i < num_iterations; i++) {
			callback_func(output, p);
		}
		p->all_sources[0].ele = 45;
		p->all_sources[0].updateFromSpherical();
		for (int i = 0; i < num_iterations; i++) {
			callback_func(output, p);
		}
		p->all_sources[0].ele = 55;
		p->all_sources[0].updateFromSpherical();
		for (int i = 0; i < 1000; i++) {
			callback_func(output, p);
		}
		p->all_sources[0].ele = 65;
		p->all_sources[0].updateFromSpherical();
		for (int i = 0; i < num_iterations; i++) {
			callback_func(output, p);
		}
		p->all_sources[0].ele = 75;
		p->all_sources[0].updateFromSpherical();
	}
	fprintf(stderr, "Testing both interpolation\n");
	p->all_sources[0].azi = 3;
	p->all_sources[0].ele = 5;
	p->all_sources[0].updateFromSpherical();
	for (int repeats = 0; repeats < 10; repeats++) {
		for (int i = 0; i < num_iterations; i++) {
			callback_func(output, p);
		}
		p->all_sources[0].azi = 6;
		p->all_sources[0].ele = 15;
		p->all_sources[0].updateFromSpherical();
		for (int i = 0; i < num_iterations; i++) {
			callback_func(output, p);
		}
		p->all_sources[0].azi = 11;
		p->all_sources[0].ele = 25;
		p->all_sources[0].updateFromSpherical();
		for (int i = 0; i < num_iterations; i++) {
			callback_func(output, p);
		}
		p->all_sources[0].azi = 18;
		p->all_sources[0].ele = 35;
		p->all_sources[0].updateFromSpherical();
		for (int i = 0; i < num_iterations; i++) {
			callback_func(output, p);
		}
		p->all_sources[0].azi = 23;
		p->all_sources[0].ele = 45;
		p->all_sources[0].updateFromSpherical();
		for (int i = 0; i < num_iterations; i++) {
			callback_func(output, p);
		}
		p->all_sources[0].azi = 28;
		p->all_sources[0].ele = 55;
		p->all_sources[0].updateFromSpherical();
		for (int i = 0; i < num_iterations; i++) {
			callback_func(output, p);
		}
		p->all_sources[0].ele = 65;
		p->all_sources[0].updateFromSpherical();
		for (int i = 0; i < num_iterations; i++) {
			callback_func(output, p);
		}
		p->all_sources[0].azi = 33;
		p->all_sources[0].ele = 75;
		p->all_sources[0].updateFromSpherical();
	}*/
	
	for (int i = 0; i < num_iterations; i++) {
		callback_func(output, p);
	}
	delete[] output;
}
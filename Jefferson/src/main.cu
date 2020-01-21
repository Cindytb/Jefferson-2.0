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
Data data;
Data* p = &data;

int main(int argc, char* argv[]) {
	std::string input = "media/Castanets-441.wav";
	std::string reverb = "media/s1_r1_b_441_mono.wav";
	std::string out = "ofile.wav";
	p->type = GPU_FD_COMPLEX;
	/*Parse Input*/

	for (int i = 1; i < argc; i++) {
		if (argv[i][0] == '-') {
			if (argv[i][1] == 't' || argv[i][1] == 'T') {
				switch (argv[i + 1][0]) {
				case '0':
					p->type = GPU_FD_COMPLEX;
					break;
				case '1':
					p->type = GPU_FD_BASIC;
					break;
				case '2':
					p->type = GPU_TD;
					break;
				case '3':
					p->type = CPU_FD_COMPLEX;
					break;
				case '4':
					p->type = CPU_FD_BASIC;
					break;
				case '5':
					p->type = CPU_TD;
					break;
				default:
					printf("ERROR: Invalid type. Defaulting to GPU_FD_COMPLEX");
				}
			}
			else if (argv[i][1] == 'i') {
				input = argv[i + 1];
			}
			else if (argv[i][1] == 'r') {
				reverb = argv[i + 1];
			}
			else if (argv[i][1] == 'o') {
				out = argv[i + 1];
			}
		}
	}
	std::cout << "Input file: " << input << std::endl << "Reverb file: " << reverb << std::endl << "Output file: " << out << std::endl;
	std::cout << "Type: " << p->type << std::endl;
	p->num_sources = 1;
	p->all_sources = new GPUSoundSource[p->num_sources]; /*Moving all allocation & initialization into the constructor*/
	printSize();
#if (DEBUGMODE != 1)
	/*Initialize & read files*/
	cudaFFT(input, reverb, p);

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

	p->sndfile = sf_open(out.c_str(), SFM_WRITE, &osfinfo);

	/*precisionTest(p);
	xfadePrecisionTest(p);
	xfadePrecisionCallbackTest(p);*/
	if (p->type == GPU_FD_BASIC || p->type == GPU_FD_COMPLEX || p->type == GPU_TD) {
		printf("Blocks in flight: %i\n", FLIGHT_NUM);

		float* output = new float[FRAMES_PER_BUFFER * 2];
		p->blockNo = 0;
		for (int j = 0; j < p->num_sources; j++) {
			callback_func(output, p);
		}

		delete[] output;
	}



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
	while (curr_source->count < (counter * 44100) % curr_source->length) {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
	counter++;
	curr_source->azi = 2;
	curr_source->ele = 4;
	curr_source->updateFromSpherical();

	while (curr_source->count < (counter * 44100) % curr_source->length) {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
	counter++;
	curr_source->azi = 1;
	curr_source->ele = 3;
	curr_source->updateFromSpherical();
	//std::this_thread::sleep_for(std::chrono::seconds(1));
	while (curr_source->count < (counter * 44100) % curr_source->length) {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
	counter++;
	curr_source->azi = 4;
	curr_source->ele = 2;
	curr_source->updateFromSpherical();
	while (curr_source->count < (counter * 44100) % curr_source->length) {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
	counter++;
	curr_source->azi = 7;
	curr_source->ele = 9;
	curr_source->updateFromSpherical();
	while (curr_source->count < (counter * 44100) % curr_source->length) {
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
	counter++;
	curr_source->azi = 0;
	curr_source->ele = 0;
	curr_source->updateFromSpherical();
	std::this_thread::sleep_for(std::chrono::seconds(2));
	//char merp = getchar();

	/*THIS SECTION WILL NOT RUN IF GRAPHICS IS TURNED ON*/
	/*Placed here to properly close files when debugging without graphics*/
	cudaProfilerStop();

#endif
#if DEBUGMODE == 3
	benchmarkTesting();
	SoundSource* curr_source = (SoundSource*)&(p->all_sources[0]);
	fprintf(stderr, "Number of function calls: %llu\n", curr_source->num_calls);
#endif
	closeEverything();
	return 0;
}

void closeEverything() {
	closePA();
	checkCudaErrors(cudaDeviceSynchronize());
	sf_close(p->sndfile);
	cleanup_hrtf_buffers();
}

void benchmarkTesting() {

	cudaProfilerStart();
	float* output = new float[FRAMES_PER_BUFFER * 2];
	int num_iterations = 10;
	int num_rounds = 1;
	SoundSource* curr_source = (SoundSource*)&(p->all_sources[0]);
	float* gpu_output = new float[FRAMES_PER_BUFFER * 2 * num_iterations * (num_rounds + 1)];
	float* cpu_output = new float[FRAMES_PER_BUFFER * 2 * num_iterations * (num_rounds + 1)];
	GPUSoundSource* gsrc = &(p->all_sources[0]);
	CPUSoundSource* csrc = (CPUSoundSource*)&(p->all_sources[0]);
	fprintf(stderr, "Testing no interpolation\n");
	for (int i = 0; i < PAD_LEN; i++) {
		gsrc->x[i] = 0.0f;
	}
	int count = 0;
	curr_source->old_azi = 0.0f;
	curr_source->azi = 0;
	curr_source->ele = 0;
	curr_source->updateFromSpherical();
	for (int i = 0; i < num_iterations; i++) {
		callback_func(gpu_output + FRAMES_PER_BUFFER * 2 * count++, p);
	}
	for (int i = 0; i < num_rounds; i++) {
		curr_source->azi += 5;
		if (curr_source->azi >= 360) {
			curr_source->azi -= 360;
		}
		curr_source->updateFromSpherical();
		for (int j = 0; j < num_iterations; j++) {
			callback_func(gpu_output + FRAMES_PER_BUFFER * 2 * count++, p);
		}
	}
	p->type = CPU_FD_COMPLEX;
	count = 0;
	curr_source->count = 0;
	for (int i = 0; i < PAD_LEN; i++) {
		csrc->x[i] = 0.0f;
	}
	curr_source->old_azi = 0.0f;
	curr_source->azi = 0.0f;
	for (int i = 0; i < num_iterations; i++) {
		callback_func(cpu_output + FRAMES_PER_BUFFER * 2 * count++, p);
	}
	for (int i = 0; i < num_rounds; i++) {
		curr_source->azi += 5;
		if (curr_source->azi >= 360) {
			curr_source->azi -= 360;
		}
		curr_source->updateFromSpherical();
		for (int j = 0; j < num_iterations; j++) {
			callback_func(cpu_output + FRAMES_PER_BUFFER * 2 * count++, p);
		}
	}
	if (precisionChecking(cpu_output, gpu_output, FRAMES_PER_BUFFER * 2 * num_iterations * (num_rounds + 1))) {
		printf("ERROR: INACCURATE CPU VS GPU BUFFERS\n");
	}
	//fprintf(stderr, "Testing azimuth interpolation\n");
	//p->type = GPU_FD_COMPLEX;
	//count = 0;
	//curr_source->count = 0;
	//curr_source->old_azi = 0;
	//curr_source->old_ele = 0;
	//curr_source->azi = 1;
	//curr_source->ele = 0;
	//for (int i = 0; i < PAD_LEN; i++) {
	//	gsrc->x[i] = 0.0f;
	//	csrc->x[i] = 0.0f;
	//}
	//callback_func(gpu_output, p);
	//callback_func(gpu_output, p);

	//for (int i = 0; i < num_iterations; i++) {
	//	callback_func(gpu_output + FRAMES_PER_BUFFER * 2 * count++, p);
	//}
	//for (int i = 0; i < num_rounds; i++) {
	//	curr_source->azi += 5;
	//	if (curr_source->azi >= 360) {
	//		curr_source->azi -= 360;
	//	}
	//	for (int j = 0; j < num_iterations; j++) {
	//		callback_func(gpu_output + FRAMES_PER_BUFFER * 2 * count++, p);
	//	}
	//}
	//count = 0;
	//curr_source->count = 0;
	//curr_source->old_azi = 0;
	//curr_source->old_ele = 0;
	//curr_source->azi = 1;
	//curr_source->ele = 0;
	//p->type = CPU_FD_COMPLEX;
	//for (int i = 0; i < num_iterations; i++) {
	//	callback_func(cpu_output + FRAMES_PER_BUFFER * 2 * count++, p);
	//}
	//for (int i = 0; i < 72; i++) {
	//	curr_source->azi += 5;
	//	if (curr_source->azi >= 360) {
	//		curr_source->azi -= 360;
	//	}
	//	for (int j = 0; j < num_iterations; j++) {
	//		callback_func(cpu_output + FRAMES_PER_BUFFER * 2 * count++, p);
	//	}
	//}
	//if (precisionChecking(cpu_output, gpu_output, (num_rounds + 1) * num_iterations * FRAMES_PER_BUFFER * 2)) {
	//	printf("ERROR: INACCURATE CPU VS GPU BUFFERS\n");
	//}

	/*fprintf(stderr, "Testing Elevation interpolation\n");
	curr_source->updateFromSpherical();
	curr_source->azi = 0;
	curr_source->ele = 5;
	for (int i = 0; i < num_iterations; i++) {
		callback_func(output, p);
	}
	for (int i = 0; i < 72; i++) {
		curr_source->azi += 5;
		if (curr_source->azi >= 360) {
			curr_source->azi -= 360;
		}
		curr_source->updateFromSpherical();
		for (int j = 0; j < num_iterations; j++) {
			callback_func(output, p);
		}
	}

	fprintf(stderr, "Testing both interpolation\n");
	curr_source->updateFromSpherical();
	curr_source->azi = 1;
	curr_source->ele = 5;
	for (int i = 0; i < num_iterations; i++) {
		callback_func(output, p);
	}
	for (int i = 0; i < 72; i++) {
		curr_source->azi += 5;
		if (curr_source->azi >= 360) {
			curr_source->azi -= 360;
		}
		curr_source->updateFromSpherical();
		for (int j = 0; j < num_iterations; j++) {
			callback_func(output, p);
		}
	}*/



	delete[] output;
	delete[] gpu_output;
	delete[] cpu_output;
}
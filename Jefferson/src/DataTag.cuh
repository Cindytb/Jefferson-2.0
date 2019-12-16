#pragma once
#ifndef __DATA_TAG__
#define __DATA_TAG__
#include "Universal.cuh"
#include "SoundSource.cuh"

class Data {
public:
	SNDFILE *sndfile;				/*Soundfile object for output file*/
	SoundSource *all_sources;		/*Array of structs for all sound sources*/
	int blockNo;
	int num_sources;
	bool pauseStatus;
};

#endif
#pragma once
#ifndef __SOUND_SOURCE__
#define __SOUND_SOURCE__

#include "Universal.cuh"
#include "VBO.cuh"
#include "hrtf_signals.cuh"

class SoundSource {
public:
	float* buf;									/*Reverberated signal on host*/
	VBO* waveform;
	SoundSource();
	~SoundSource();
	/*Function to process each buffer. Must be implemented in derived classes*/
	void drawWaveform(); /*Renders the VBO onto the screen. Called by display()*/
	
	/*Functions below called by the graphics callback functions*/
	void updateFromCartesian(float3 point);
	void updateFromCartesian();
	void updateFromSpherical(float azi, float ele, float r);
	void updateFromSpherical();
	//virtual void process(int blockNo, processes type);

	float ele;						/*Elevation of the sound source*/
	float azi;						/*Azimuth of the sound source*/
	float3 coordinates;				/*3D Coordinates of the source*/
	int count;						/*Current frame count for the audio callback*/
	/*Inut file playback information*/
	int length;						/*Length of the input signal*/
	unsigned long long num_calls = 0;
//protected:
	/*Graphics Writable & Audio Readable*/
	float gain;						/*Gain for distance away*/
	float r;						/*Sound source distance*/
	int hrtf_idx;
	int hrtf_indices[4];
	float omegas[6];
	/*Store prior elevation & azimuth for switching method*/
	float old_ele;
	float old_azi;
	float sum_ms = 0;
	float avg_ms = 0;
	int num_iterations = 0;
	
	
	
	void interpolationCalculations(float ele, float azi, int* hrtf_indices, float* omegas); /*Determine omegas and hrtf indices from azi/ele*/
	
	virtual void calculateDistanceFactor();

};
#endif
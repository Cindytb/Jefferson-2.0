#include "SoundSource.cuh"

SoundSource::SoundSource() {
	count = 0;
	length = 0;
	gain = 0.99074;
	hrtf_idx = 314;
	coordinates.x = 0;
	coordinates.y = 0;
	coordinates.z = 0.5;
	azi = 0;
	ele = 0;
	r = 0.5;	
	old_azi = azi;
	old_ele = ele;
}
void SoundSource::updateFromCartesian() {
	updateFromCartesian(coordinates);
}
void SoundSource::updateFromCartesian(float3 point) {
	coordinates.x = point.x;
	coordinates.y = point.y;
	coordinates.z = point.z;
	/*Calculate the radius, distance, elevation, and azimuth*/
	r = std::sqrt(coordinates.x * coordinates.x + coordinates.z * coordinates.z + coordinates.y * coordinates.y);
	float horizR = std::sqrt(coordinates.x * coordinates.x + coordinates.z * coordinates.z);
	ele = (float)atan2(coordinates.y, horizR) * 180.0f / PI;

	azi = atan2(-coordinates.x / r, -coordinates.z / r) * 180.0f / PI;
	if (azi < 0.0f) {
		azi += 360;
	}
	ele = round(ele);
	azi = round(azi);
	hrtf_idx = pick_hrtf(ele, azi);
}
void SoundSource::updateFromSpherical() {
	updateFromSpherical(ele, azi, r);
}

void SoundSource::updateFromSpherical(float ele, float azi, float r) {
	this->ele = round(ele);
	this->azi = round(azi);
	this->r = r;
	ele = this->ele;
	azi = this->azi;
	coordinates.x = r * sin(azi * PI / 180.0f);
	coordinates.z = r * -cos(azi * PI / 180.0f);
	coordinates.y = r * sin(ele * PI / 180.0f);
	/*coordinates.x = r * sin(ele * PI / 180.0f) * cos((azi + 90) * PI / 180.0f);
	coordinates.y = r * sin(ele * PI / 180.0f); 
	coordinates.z = r * sin(ele * PI / 180.0f) * cos((azi + 90) * PI / 180.0f);*/
	hrtf_idx = pick_hrtf(ele, azi);
}
void SoundSource::drawWaveform() {
	float rotateVBO_y = atan2(-coordinates.z, coordinates.x) * 180.0f / PI;

	if (rotateVBO_y < 0) {
		rotateVBO_y += 360;
	}
	waveform->averageNum = 100;
	waveform->update();
	waveform->draw(rotateVBO_y, ele, 0.0f);
}
void SoundSource::interpolationCalculations(float ele, float azi, int* hrtf_indices, float* omegas) {
	float omegaA, omegaB, omegaC, omegaD, omegaE, omegaF;
	int phi[2];
	int theta[4];
	float deltaTheta1, deltaTheta2;
	phi[0] = int(ele) / 10 * 10; /*hard coded 10 because the elevation increments in the KEMAR HRTF database is 10 degrees*/
	phi[1] = int(ele + 9) / 10 * 10;
	omegaE = (ele - phi[0]) / 10.0f;
	omegaF = (phi[1] - ele) / 10.0f;

	for (int i = 0; i < NUM_ELEV; i++) {
		if (phi[0] == elevation_pos[i]) {
			deltaTheta1 = azimuth_inc[i];
		}
		if (phi[1] == elevation_pos[i]) {
			deltaTheta2 = azimuth_inc[i];
			break;
		}
	}
	theta[0] = int(azi / deltaTheta1) * deltaTheta1;
	theta[1] = int((azi + deltaTheta1 - 1) / deltaTheta1) * deltaTheta1;
	theta[2] = int(azi / deltaTheta2) * deltaTheta2;
	theta[3] = int((azi + deltaTheta2 - 1) / deltaTheta2) * deltaTheta2;
	omegaA = (azi - theta[0]) / deltaTheta1;
	omegaB = (theta[1] - azi) / deltaTheta1;
	omegaC = (azi - theta[2]) / deltaTheta2;
	omegaD = (theta[3] - azi) / deltaTheta2;

	hrtf_indices[0] = pick_hrtf(phi[0], theta[0]);
	hrtf_indices[1] = pick_hrtf(phi[0], theta[1]);
	hrtf_indices[2] = pick_hrtf(phi[1], theta[2]);
	hrtf_indices[3] = pick_hrtf(phi[1], theta[3]);

	omegas[0] = omegaA;
	omegas[1] = omegaB;
	omegas[2] = omegaC;
	omegas[3] = omegaD;
	omegas[4] = omegaE;
	omegas[5] = omegaF;

}
void SoundSource::calculateDistanceFactor() {
	;
}

SoundSource::~SoundSource() {
	free(buf);
}

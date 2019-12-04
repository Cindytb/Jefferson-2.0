#include "SoundSource.cuh"

void SoundSource::updateInfo() {
	/*Calculate the radius, distance, elevation, and azimuth*/
	float r = std::sqrt(coordinates.x * coordinates.x + coordinates.z * coordinates.z + coordinates.y * coordinates.y);
	float horizR = std::sqrt(coordinates.x * coordinates.x + coordinates.z * coordinates.z);
	ele = (float)atan2(coordinates.y, horizR) * 180.0f / PI;

	azi = atan2(-coordinates.x / r, coordinates.z / r) * 180.0f / PI;
	if (azi  < 0) {
		azi += 360;
	}
	float newR = r / 100 + 1;
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

# Jefferson

## Binaural Spatializer in OpenGL and CUDA

This project uses OpenGL to create a 3D visualization of a sound source and a listener. The audio is processed using 3D Audio techniques to match the distance and angles of the sound source compared to the listener. The project utilizes the HRTF interpolation and distance delay algorithms found in Jose Belloch's paper.

## Program Flow
### Preprocessing
- Read input & reverb file
- (optional) do convolution reverb on input
- Read HRIRs
- Transform (all 710 * 2) HRIRs to HRTFs
- Create and allocate all audio buffers
- Create FFT Plans
- Import 3D model
- Create mesh for floor

### Runtime
- Graphics side of the program updates the X, Y, and Z coordinates of the sound source
    - Write to the sound source class
    - Computes azimuth and elevation with each frame refresh
- Audio side buffer size ~128 or 256 at 44.1k sample rate
    - 128 samples = 2.8 milliseconds
    - GPU computation takes ~0.3 milliseconds in the worst case scenario

A purple cartoon character indicates the listener, which remains is movable around the space. The green indicates the sound source which is stationary in the middle. Different keys listed below will move the sound source in the X, Y, and Z axes. The visualization can also be rotated by left clicking and dragging the visualization which helps to better visualize the 3D space. The user can also zoom in and out by clicking and dragging the right arrow key or by using the scroll wheel. The R key will reset back to the default perspective and position. My program also optionally writes the output to a sound file.

The cartoon character, which I’ve fondly named Jefferson, was created by Vinnie Huynh in Blender. I exported the model to an FBX file and imported it that way.

## Tools list:
**OpenGL** - short for Open Graphics Library. It is an API/library in several programming languages to draw 2D and 3D images. It’s portable and it’s implemented primarily in each computer’s hardware.

**CUDA** - acronym for Compute Unified Device Architecture. It’s “a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs)” (Nvidia). It is a proprietary but free API in several different programming languages to speak directly to NVIDIA hardware and utilize parallel processing.

[General Knowledge](https://developer.nvidia.com/cuda-zone)

[Download link](https://developer.nvidia.com/cuda-zone)

[Documentation](https://docs.nvidia.com/cuda/)

**cuFFT** – NVIDIA CUDA Fast Fourier Transform library.

[General knowledge](https://developer.nvidia.com/cufft)

[Documentation](https://docs.nvidia.com/cuda/cufft/index.html)


**Thrust** – “Thrust is a C++ template library for CUDA based on the Standard Template Library (STL)” (Nvidia). It’s a library within CUDA that utilizes parallel processing for algorithms that already exist in C++’s standard library such as summing, reducing, and sorting.

[NVIDIA Documentation](https://docs.nvidia.com/cuda/thrust/index.html)

[GitHub Documentation](https://thrust.github.io/)

**HRTF** – acronym for Head Related Transfer Function. Several short audio filters depending on angle (azimuth) and elevation that can make a sound come from different locations in 3D space. The ones I used were the compact set from [MIT and KEMAR](http://sound.media.mit.edu/resources/KEMAR.html)

**PortAudio** - Portable audio library used to connect to the computer’s sound device

[Download link](http://www.portaudio.com/)

**ALSA** - acronym for Advanced Linux Sound Architecture. One of the libraries that PortAudio uses under the hood on Linux.

[More Info](https://www.alsa-project.org/wiki/Main_Page)

**ASIO** - acronym for Audio Stream Input/Output. Soundcard driver protocol by Steinberg for low-latency audio. Proprietary software that is freely available by Steinberg, but it can not be re-distributed. PortAudio on Windows can be configured to use ASIO.

[More Info](https://www.steinberg.net/en/company/developers.html)


**libsndfile** – Portable audio library used to read contents of wave files

[Download link](http://www.mega-nerd.com/libsndfile/)

**Blender** – Open source, free 3D creation suite

[Download link](https://blender.org/)

**ASSIMP** – Acronym for Open Asset Import Library. It “is a portable Open Source library to import various well known 3D model formats in a uniform manner.” This was used to import an FBX file into OpenGL.

[Documentation](http://www.assimp.org)

## Sources
Belloch, J. A., Ferrer, M., Gonzalez, A., Martinez-Zaldivar, F. J., & Vidal, A. M. (2013). Headphone-based virtual spatialization of sound with a GPU accelerator. Journal of the Audio Engineering Society, 61 (7/8), 546-561.
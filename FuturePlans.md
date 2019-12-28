# Future Plans:
# High Priority Bug Fixes
- [ ] GPU Frequency Domain HRTF Interpolation has sporadic clicking sounds when crossfading between two locations
- [ ] GPU FD sounds staticky even when it's not clicking

# Overhauls
- [ ] Revamp the OpenGL implementation to modern OpenGL
- [ ] Be able to include more sound objects  (audio-wise, it's ready. graphics-wise, no)
- [x] Implement GPU HRTF Interpolation
    
    Needs fixing and optimizing
    
    Sporadically, an audio buffer will have values ~10-15 and cause the sound to clip
- [ ] Implement CPU HRTF Interpolation

    Needs fixing
- [ ] Attempt to use NVIDIA's ray-tracing technology for audio  
- [ ] Include a waveform that passes through Jefferson's head  

    Have one, but need to find a way to allow longer waveforms
- [ ] Allow the code to accept any HRTF database using the 

# Objective Testing
- [ ] Do benchmarking on the HRTF interpolation implementations for CPU and GPU

# Subjective Testing
- [ ] Perform subjective testing on GPU interpolation algorithm

    Possibly test out different interpolation algorithms




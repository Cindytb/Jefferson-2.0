#%%
import soundfile as sf
import numpy as np

cpu = sf.read("Jefferson/CPU_FD_COMPLEX.wav")[0]
gpu = sf.read("Jefferson/ofile.wav")[0]

#%%
maximum = 0
max_idx = 0
if (cpu.shape[0] != gpu.shape[0] - 256):
    print("ERROR: Two wave files are different lengths")
for i in range(cpu.shape[0]):
    diff = np.abs(cpu[i][0] - gpu[i+256][0])
    diff2 = np.abs(cpu[i][1] - gpu[i+256][1])
    if diff > maximum:
        maximum = diff
        max_idx = i
    if diff2 > maximum:
        maximum = diff2
        max_idx = i
if diff > 1e-8:
    print("PRECISION ERROR")
print("Maximum Error:", diff)
print("Index:", max_idx)
print("Percentage into the piece", max_idx / cpu.shape[0])

# %%

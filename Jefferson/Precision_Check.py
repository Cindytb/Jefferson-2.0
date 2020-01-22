#%%
import soundfile as sf
import numpy as np

cpu = sf.read("CPU_FD_COMPLEX.wav")[0]
gpu = sf.read("ofile.wav")[0]

#%%
maximum = 0
max_idx = 0
if (cpu.shape[0] != gpu.shape[0] - 128):
    print(cpu.shape)
    print(gpu.shape)
    print("ERROR: Two wave files are different lengths")
for i in range(cpu.shape[0]):
    diff = np.abs(cpu[i][0] - gpu[i+128][0])
    diff2 = np.abs(cpu[i][1] - gpu[i+128][1])
    if diff > maximum:
        maximum = diff
        max_idx = i
    if diff2 > maximum:
        maximum = diff2
        max_idx = i
if maximum > 1e-7:
    print("PRECISION ERROR")
print("Maximum Error:", maximum / 1e-8, "1e-8", maximum)
print("Index:", max_idx)
print("Percentage into the piece", max_idx / cpu.shape[0])

# %%

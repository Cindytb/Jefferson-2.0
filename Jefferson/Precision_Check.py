#%%
import soundfile as sf
import numpy as np

cpu = sf.read("CPU_FD_COMPLEX.wav")[0]
gpu = sf.read("ofile.wav")[0]

#%%
diff = cpu - gpu
maximum = np.max(np.abs(diff))
max_idx = np.argmax(np.abs(diff))
if maximum > 2e-7:
    print("PRECISION ERROR")
print("Maximum Error:", maximum / 1e-8, "1e-8", maximum)
print("Index:", max_idx)
print("Percentage into the piece", max_idx / cpu.shape[0] / cpu.shape[1])

# %%

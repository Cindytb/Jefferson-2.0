
FILENAMES=("graphics" "hrtf_signals" "kernels" "main" "vbo" "Audio" "cudaPart" )
INC="-I/usr/local/cuda/samples/common/inc"
CUDA_LIBS="-lcufft -lcudart_static"
C_LIBS="-lglut -lGL -lGLU -lGLEW -lsndfile -lportaudio -lrt -lasound -lm -lpthread "
C_FILENAMES=("mat_matr" "mat_vect" "object" "load_3ds")
for i in ${C_FILENAMES[@]}; do
    printf "Compiling %s\n" $i
    g++ -c $i.cpp -g -o $i.o $INC
done
for i in ${FILENAMES[@]}; do
    printf "Compiling %s\n" $i
    /usr/local/cuda-10.1/bin/nvcc -ccbin g++ -g -c $i.cu -o $i.o $INC -gencode arch=compute_60,code=sm_60
done
/usr/local/cuda-10.1/bin/nvcc -ccbin g++ -g -pg -link mat_matr.o mat_vect.o object.o load_3ds.o \
    main.o hrtf_signals.o Audio.o graphics.o  kernels.o vbo.o cudaPart.o \
    -L/home/cindy/NVIDIA_CUDA-10.1_Samples/common/lib/linux/x86_64 \
    $C_LIBS $CUDA_LIBS $INC -gencode arch=compute_60,code=sm_60
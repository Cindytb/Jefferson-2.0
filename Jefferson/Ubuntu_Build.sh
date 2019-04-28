
FILENAMES=("graphics" "hrtf_signals" "kernels" "main" "vbo" "Audio" "cudaPart" )
INC="-I/usr/local/cuda/samples/common/inc"
CUDA_LIBS="-lcufft -lcudart_static"
C_LIBS="-lglut -lGL -lGLU -lGLEW -lsndfile -lportaudio -lrt -lm -ljack -lasound -lpthread"
C_FILENAMES=("mat_matr" "mat_vect" "object" "load_3ds")

if [[ $# != 0 ]]; then
    if [[ $1 == "clean" ]]; then
        printf "Cleaning object files\n"
        rm obj/*
    fi
fi


for i in ${C_FILENAMES[@]}; do
    if [[ ! -f "obj/$i.o"  ||  "src/$i.cpp" -nt "obj/$i.o" ]]; then
        printf "Compiling %s\n" $i
        g++ -c src/$i.cpp -g -o obj/$i.o $INC
    fi
done
for i in ${FILENAMES[@]}; do
    if [[ ! -f "obj/$i.o"  ||  "src/$i.cu" -nt "obj/$i.o" ]]; then
        printf "Compiling %s\n" $i
        /usr/local/cuda-10.1/bin/nvcc -ccbin g++ -g -c src/$i.cu -o obj/$i.o $INC -gencode arch=compute_60,code=sm_60
    fi
done
cd obj
printf "Linking files into executable\n"
/usr/local/cuda-10.1/bin/nvcc -ccbin g++ -g -pg -o ../bin/Jefferson \
    -link mat_matr.o mat_vect.o object.o load_3ds.o \
    main.o hrtf_signals.o Audio.o graphics.o  kernels.o vbo.o cudaPart.o \
    -L/home/cindy/NVIDIA_CUDA-10.1_Samples/common/lib/linux/x86_64 \
    $C_LIBS $CUDA_LIBS $INC -gencode arch=compute_60,code=sm_60
cd ..
all:
	nvcc kernel.cu -o kernel

clean: rm -rf *.o
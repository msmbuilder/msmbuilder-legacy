# This makefile will build a swigged python module swGPURMSD.py

_swGPURMSD.so: RMSD.o swGPURMSD.o
	g++ -shared RMSD.o swGPURMSD.o -L/usr/local/cuda/lib64 -lcudart -o _swGPURMSD.so

RMSD.o:
	nvcc -arch=sm_20 --ptxas-options=-v -c --compiler-options '-fPIC' -o RMSD.o RMSD.cu -I/usr/local/cuda/include

swGPURMSD.cpp:
	swig -c++ -python -o swGPURMSD.cpp swig.i

swGPURMSD.o: swGPURMSD.cpp
	gcc -fPIC -c swGPURMSD.cpp -o swGPURMSD.o `python-config --cflags`

all: _swGPURMSD.so

clean:
	rm *.o  swGPURMSD.*
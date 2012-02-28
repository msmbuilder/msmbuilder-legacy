#This Makefile should now be obsolete, as python setup.py should take care of compiling this python extension.

#Linux (Ubuntu 10.04)
#PYINCLUDEPATH = /usr/include/python2.6
#NUMPYINCLUDEPATH = /usr/include/python2.6/numpy

#Linux Enthought Python
#PYINCLUDEPATH = /home/kyleb/opt/epd/include/python2.7
#NUMPYINCLUDEPATH = /home/kyleb/opt/epd/lib/python2.7/site-packages/numpy/core/include/numpy/

#MacOSX - native python
#NUMPYINCLUDEPATH = /Library/Python/2.6/site-packages/numpy-2.0.0.dev_20101022-py2.6-macosx-10.6-universal.egg/numpy/core/include/numpy/

#MaxOSX - Enthought Python Distribution
PYINCLUDEPATH=/Library/Frameworks/EPD64.framework/Versions/7.1/include/python2.7
NUMPYINCLUDEPATH=/Library/Frameworks/EPD64.framework/Versions/7.1/lib/python2.7/site-packages/numpy/core/include/numpy

#You may need to zero out the following if your env variables are polluted with other python installations
#export LD_LIBRARY_PATH=''
#export DYLD_LIBRARY_PATH=''

CC = gcc
CPU=core2
# The above CPU line does not work for gcc 4.0 (OS X 10.5); use CPU=nocona for that (or pentium-m?)
CFLAGS = --std=c99 -O2 -mtune=$(CPU) -fPIC -shared -msse2 -msse3 -fopenmp
#CFLAGS = --std=c99 -O2 -mtune=$(CPU) -fPIC -shared -msse2 -msse3

all: rmsdcalc.so

theobald_rmsd.o: theobald_rmsd.c
	$(CC) -c $(CFLAGS) theobald_rmsd.c

rmsdcalc.so: rmsd_numpy_array.c theobald_rmsd.o
#	$(CC) -shared rmsd_numpy_array.c $(CFLAGS) -I $(PYINCLUDEPATH) -I $(NUMPYINCLUDEPATH) -lpython theobald_rmsd.o -o rmsdcalc.so # use this line for OSX with native python
	$(CC) -shared rmsd_numpy_array.c $(CFLAGS) -I$(PYINCLUDEPATH) -I$(NUMPYINCLUDEPATH) -L/Library/Frameworks/EPD64.framework/Versions/7.1/lib -lpython2.7 theobald_rmsd.o -o rmsdcalc.so # Used this line for OSX with EPD
#	$(CC) -shared rmsd_numpy_array.c $(CFLAGS) -I $(PYINCLUDEPATH) -I $(NUMPYINCLUDEPATH) theobald_rmsd.o -o rmsdcalc.so

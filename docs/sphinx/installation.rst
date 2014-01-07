Installation
============

MSMBuilder should run on most modern computers equiped with a scientific python installation. But, in the interest of being explicit, here the requirements

- A CPU with SSE3 support, which has been tandard on all x86 processors produce after 2006.
- A working C compiler, such as GCC 4.2 or later, clang, or MSVC.
- Python, with some scientific modules installed (see below)

MSMBuilder is written in the python programming language, and uses a
variety of tools from the wider scientific python ecosystem, which may
need to be installed separately. They include


Python Prerequisites
--------------------
-  MDTraj
-  Numpy
-  Scipy
-  PyTables
-  numexpr
-  fastcluster (for hierarchical clustering)
-  matplotlib (optional for plotting)
-  ipython (optional for interactive mode)
-  pymol (optional for visualization)

Two companies, Enthought and Continuum Analytics, produce python
distributions which bundle many of these packages in with the python
interpreter into a single binary installer, available for all major
operating systems. These are the Enthought Canopy python distribution
and Continuum’s Anaconda.


Install Python and Python Packages
----------------------------------

Rather than individually install the many python dependencies, we
recommend that you download the Python2.7 version of the Enthought
Canopy or Continuum Anaconda, which contain almost all python
dependencies required to run MSMBuilder. If you have a 64 bit platform,
please use the 64 bit versions, as this will give higher performance.

Note for OSX users: Enthought represents the easiest way to obtain a
working Python installation. The OSX system Python install is broken and
cannot properly build Python extensions, which are required for
MSMBuilder installation. Also, see FAQ question 11 for a known issue
with OSX Lion and OpenMP.

Note: if you are unable to use Canopy or Anaconda, there are other
pre-compiled Python distributions available, although they might not be
as fast as Enthought. Options include Python(x,y) and the Scipy
Superpack (OSX). Finally, most Linux users can install most
prerequisites using their package manager. In Ubuntu, the following will
install most of the prerequisites:

::

    $ sudo apt-get install libhdf5-serial-dev python-dev python-numpy \
    python-scipy python-setuptools python-nose python-tables \
    python-matplotlib python-yaml swig ipython

Neither Canopy nor Anaconda include MDTraj nor fastcluster. They can be installed be installed using python’s package manager, ``pip``.

::

    $ pip install -r requirements.txt

Download and Install MSMBuilder
-------------------------------

Download MSMBuilder, unzip, move to the msmbuilder directory. Install
using setup.py:

::

    $ python setup.py install

You may need root privileges during the install step; alternatively, you
can specify an alternative install path via ``–prefix=XXX``. If you
performed the install step with ``–prefix=XXX``, you need to ensure that

#. XXX/bin is included in your PATH

#. XXX/lib/python2.7/site-packages/ is included in your PYTHONPATH

Step (1) ensures that you can run MSMBuilder scripts without specifying
their location. Step (2) ensures that your Python can locate the
MSMBuilder libraries.

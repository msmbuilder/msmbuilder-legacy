Installation
============

MSMBuilder is a python package that uses a number of components from the "scientific python" stack. These packages include `numpy and scipy <http://scipy.org/getting-started.html>`_ for array manipulation and numerical linear algebra, `PyTables <http://www.pytables.org/moin>`_ for storing binary data, and others.

.. note::
    MSMBuilder is runs on Python 2.7 and Python 3.3. Use whichever you prefer. Other version like Python 2.6 and Python 3.2 probably will work, but are not explicitly supported .

Easily with ``conda``
---------------------

The easiest way to install MSMBuilder is with the python package manager ``conda``. ``conda`` is an open-source cross-platform binary package manager integrated with the scientific python stack. It's built into the `Anaconda python distribution <http://docs.continuum.io/anaconda/>`_ produced by Continuum Analytics, which is a python installer that comes shipped with many of the python packages needed for science.

.. warning::

   We **strongly** recommend using Anaconda. Installing the scientific stack by hand can be quite tricky.

If you don't want to get Anaconda, you can also install ``conda`` into an existing python interpreter. Once you have ``conda``, install MSMBuilder with ::

   conda config --add channels http://conda.binstar.org/omnia
   conda install msmbuilder


.. note::
    For windows users, we recommend using the 32-bit version of Anaconda.


Medium With ``pip``
-------------------

MSMBuilder can be instaleld with ``pip``, but ``pip`` is not fantastic at installing the dependencies. If you've already got the dependencies installed (see below), then you can download and install MSMBuilder::

    pip install msmbuilder

Hard Way by Hand
----------------

If you use conda, all of this will be done automatically. If you prefer to do things by hand, keep reading.

You'll need to get the following python packages:

-  `mdtraj >= 0.8 <https://pypi.python.org/pypi/mdtraj>`_
-  `numpy >= 1.6 <https://pypi.python.org/pypi/numpy>`_
-  `scipy >= 0.11 <https://pypi.python.org/pypi/scipy>`_
-  `tables >= 2.4.0 <https://pypi.python.org/pypi/tables>`_
-  `pyyaml <https://pypi.python.org/pypi/PyYAML>`_
-  `fastcluster (for hierarchical clustering) <https://pypi.python.org/pypi/fastcluster>`_
-  matplotlib (optional for plotting)
-  ipython (optional for interactive mode)

On a debian-based linux, you can get most of them with ::

    $ sudo apt-get install libhdf5-serial-dev python-dev python-numpy \
    python-scipy python-setuptools python-nose python-tables \
    python-matplotlib python-yaml swig ipython python-pip


Download MSMBuilder, unzip, move to the msmbuilder directory. Then, use ``pip``
to install any remaining dependencies ::

    $ pip install -r requirements.txt

Then install MSMBuilder with ``setup.py`` ::

    $ python setup.py install

You may need root privileges during the install step; alternatively, you
can specify an alternative install path via ``–prefix=XXX``. If you
performed the install step with ``–prefix=XXX``, you need to ensure that

#. ``XXX/bin`` is included in your ``PATH``

#. ``XXX/lib/python2.7/site-packages/`` is included in your ``PYTHONPATH``

Step (1) ensures that you can run MSMBuilder scripts without specifying
their location. Step (2) ensures that your Python can locate the
MSMBuilder libraries.

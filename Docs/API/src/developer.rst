Developer Zone
==============

PEP8
~~~~
  - Functions should be lower_case_with_underscores_separating_words
  - local variables should_also_be_like_functions
  - ClassesShouldUseThisKindOfStyle

Docstrings
~~~~~~~~~~
Numpy docstring convention. We're going to use autodoc as much as possible, which means that the contents of the docs are built from the docstrings.

This means that the docstring need to be COMPLETE.

Furthermore, in order to display in a consistent way, the doctrings need to be in a specific format.

-  `numpy docstring standard <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
- `docstring example <https://github.com/numpy/numpy/blob/master/doc/example.py>`_

Building the docs
-----------------
You need eed to make sure numpydoc is installed. ::
  $ easy_install numpydoc

Then you can make the docs with the supplied makefile ::
  $ make html 


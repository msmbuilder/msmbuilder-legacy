============
Contributing
============

This is primarily an academic project, and everyone is welcome to contribute.
The project is hosted on http://github.com/SimTk/msmbuilder

Submitting a bug report
-----------------------
If you experience issues using this package, do not hesitate to submit an issue
to the [bug tracker](https://github.com/SimTk/msmbuilder/issues). You're also
invited to post feature requests or links to pull requests.

Retrieving the latest code
==========================

We use `Git <http://git-scm.com/>`_ for version control and
`GitHub <http://github.com/>`_ for hosting our main repository.

You can check out the latest sources with the command::

    git clone git://github.com/SimTk/msmbuilder.git

or if you have write privileges::

    git clone git@github.com:SimTk/msmbuilder.git

If you run the development version, it is cumbersome to reinstall the
package each time you update the sources. It is thus preferred that
you add the msmbuilder directory to your ``PYTHONPATH`` and build the
extension in place::

    python setup.py build_ext --inplace

Contributing code
=================

.. note::

  To avoid duplicating work, it is highly advised that you contact the
  developers on the `issue tracker <https://github.com/SimTk/msmbuilder/issues>`_ before starting work on a
  non-trivial feature.


How to contribute
-----------------

The preferred way to contribute to msmbuilder is to fork the `main
repository <http://github.com/SimTk/msmbuilder/>`__ on GitHub:

 1. `Create an account <https://github.com/signup/free>`_ on
    GitHub if you do not already have one.

 2. Fork the `project repository
    <http://github.com/SimTk/msmbuilder>`__: click on the 'Fork'
    button near the top of the page. This creates a copy of the code under your
    account on the GitHub server.

 3. Clone this copy to your local disk::

        $ git clone git@github.com:YourLogin/msmbuilder.git

 4. Create a branch to hold your changes::

        $ git checkout -b my-feature

    and start making changes. Never work in the ``master`` branch!

 5. Work on this copy, on your computer, using Git to do the version
    control. When you're done editing, do::

        $ git add modified_files
        $ git commit

    to record your changes in Git, then push them to GitHub with::

        $ git push -u origin my-feature

Finally, go to the web page of the your fork of the msmbuilder repo,
and click 'Pull request' to send your changes to the maintainers for review.
request. This will send an email to the committers.

(If any of the above seems like magic to you, then look up the
`Git documentation <http://git-scm.com/documentation>`_ on the web.)

It is recommended to check that your contribution complies with the following
rules before submitting a pull request:

    * Follow the `coding-guidelines`_ (see below).

    * All public methods should have informative docstrings with sample
      usage presented as doctests when appropriate.

    * All other tests pass when everything is rebuilt from scratch. On
      Unix-like systems, check with::
      
        $ nosetests
    
You can also check for common programming errors with the following tools:

  * No pyflakes warnings, check with::

      $ pip install pyflakes
      $ pyflakes path/to/module.py

  * No PEP8 warnings, check with::

      $ pip install pep8
      $ pep8 path/to/module.py

  * AutoPEP8 can help you fix some of the easy redundant errors::

      $ pip install autopep8
      $ autopep8 path/to/pep8.py
      
.. note::

  The current state of the scikit-learn code base is not compliant with
  all of those guidelines, but we expect that enforcing those constraints
  on all new contributions will get the overall code base quality in the
  right direction.

.. note::

   For two very well documented and more detailed guides on development
   workflow, please pay a visit to the `Scipy Development Workflow
   <http://docs.scipy.org/doc/numpy/dev/gitwash/development_workflow.html>`_ -
   and the `Astropy Workflow for Developers <http://astropy.readthedocs.org/en/latest/development/workflow/development_workflow.html>`_
   sections.


Coding guidelines
=================

The following are some guidelines on how new code should be written. Of
course, there are special cases and there will be exceptions to these
rules. However, following these rules when submitting new code makes
the review easier so new code can be integrated in less time.

Uniformly formatted code makes it easier to share code ownership. The
msmbuilder project tries to closely follow the official Python guidelines
detailed in `PEP8 <http://www.python.org/dev/peps/pep-0008/>`_ that
detail how code should be formatted and indented. Please read it and
follow it.

In addition, we add the following guidelines:

   * Use underscores to separate words in non class names: ``n_samples``
     rather than ``nsamples``.

   * Avoid multiple statements on one line. Prefer a line return after
     a control flow statement (``if``/``for``).

   * **Please don't use `import *` in any case**. It is considered harmful
     by the `official Python recommendations
     <http://docs.python.org/howto/doanddont.html#from-module-import>`_.
     It makes the code harder to read as the origin of symbols is no
     longer explicitly referenced, but most important, it prevents
     using a static analysis tool like `pyflakes
     <http://www.divmod.org/trac/wiki/DivmodPyflakes>`_ to automatically
     find bugs in scikit-learn.

   * Use the `numpy docstring standard
     <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_
     in all your docstrings.


A good example of code that we like can be found `here <https://svn.enthought.com/enthought/browser/sandbox/docs/coding_standard.py>`_.

Building the docs
-----------------
You need to make sure numpydoc is installed.::

  $ easy_install numpydoc

Then you can make the docs with the supplied makefile::

  $ make html 


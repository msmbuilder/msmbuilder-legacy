.. currentmodule:: msmbuilder.arglib

arglib: :mod:`msmbuilder.arglib`
==================================

Support for command line scripts, making it easier to parse arguments
  
Parsers
-------

.. autoclass:: ArgumentParser

  .. autosummary::
    :toctree: generated/

    ArgumentParser.add_argument
    ArgumentParser.add_argument_group
    ArgumentParser.add_subparsers
    ArgumentParser.parse_args

.. autosummary::
  :toctree: generated/
  
  add_argument

Path Manipulation
-----------------

.. autosummary::
  :toctree: generated/
  
  die_if_path_exists
  ensure_path_exists
